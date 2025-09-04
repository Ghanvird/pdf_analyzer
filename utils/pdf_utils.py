# utils/pdf_utils.py
import io, os, re, subprocess, tempfile, shutil
from typing import Dict, List, Tuple

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

import pytesseract
from paddleocr import PaddleOCR

from .tesseract_path import ensure_tesseract, auto_set_poppler_cmd

# ------------- binaries (offline) -------------
# poppler for pdf2image
_POPPLER_BIN = None
try:
    _POPPLER_BIN = auto_set_poppler_cmd()
except Exception:
    _POPPLER_BIN = None

# tesseract for fallback
try:
    ensure_tesseract()
except Exception:
    pass

# kraken model (optional)
_KRAKEN_MODEL = os.getenv("KRAKEN_MODEL")
def _kraken_ocr_on_image(img: Image.Image) -> str:
    if not _KRAKEN_MODEL:
        return ""
    # require 'kraken' CLI
    exe = shutil.which("kraken")
    if not exe:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        cmd = [exe, "ocr", "-m", _KRAKEN_MODEL, "-i", tmp.name, "-"]
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return (out.stdout or "").strip()
        except Exception:
            return ""
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

# PaddleOCR (lazy)
# PaddleOCR (lazy)
_PADDLE = None
def _paddle():
    """
    Create a PaddleOCR pipeline that uses local/cached models and **does not**
    force the angle classifier (which requires PaddleX-format CLS bundles).
    """
    global _PADDLE
    if _PADDLE is not None:
        return _PADDLE
    try:
        # Pass only portable kwargs; no show_log, no use_gpu, no cls_model_dir
        from paddleocr import PaddleOCR
        _PADDLE = PaddleOCR(lang="en")
    except Exception:
        _PADDLE = None
    return _PADDLE


# ------------- core extractors -------------
def extract_text_with_pdfplumber(pdf_bytes: bytes) -> Tuple[str, List[List[List[str]]]]:
    text_parts: List[str] = []
    tables: List[List[List[str]]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if txt:
                text_parts.append(txt)
            try:
                page_tables = page.extract_tables() or []
                for tbl in page_tables:
                    tables.append(tbl)
            except Exception:
                pass
    return "\n".join(text_parts).strip(), tables

def _tesseract_ocr(img: Image.Image) -> str:
    g = ImageOps.autocontrast(img.convert("L"))
    g = g.filter(ImageFilter.SHARPEN)
    try:
        return pytesseract.image_to_string(g, config="--oem 1 --psm 6 -l eng").strip()
    except Exception:
        return ""

def extract_text_with_paddle(pdf_bytes: bytes, conf_threshold: float = 0.70) -> str:
    poppler_path = _POPPLER_BIN
    images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png", poppler_path=poppler_path)
    o = _paddle()
    out_texts: List[str] = []

    for img in images:
        if o is None:
            # Just fall back to Tesseract whole-page
            out_texts.append(_tesseract_ocr(img))
            continue

        result = o.ocr(ImageOps.exif_transpose(img), cls=False)
        # result structure: [[ [box, (text, score)], ... ]]
        blocks = result[0] if result else []
        lines: List[str] = []
        for block in blocks:
            box, (txt, score) = block
            txt = (txt or "").strip()
            if not txt:
                continue
            if score < conf_threshold:
                # try kraken, else tesseract on the cropped region
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                l, t, r, b = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                crop = img.crop((l, t, r, b))
                htr = _kraken_ocr_on_image(crop) or _tesseract_ocr(crop)
                if htr:
                    txt = htr
            lines.append(txt)
        out_texts.append("\n".join(lines))

    return "\n\n".join(t for t in out_texts if t).strip()

def _tables_to_kv(tables: List[List[List[str]]]) -> Dict[str, str]:
    """
    Convert simple 2-column tables into key->value hints.
    """
    kv: Dict[str, str] = {}
    for tbl in tables or []:
        for row in tbl or []:
            if not row:
                continue
            cells = [(c or "").strip() for c in row]
            if len(cells) >= 2 and any(cells[1:]):
                left = cells[0]
                right = ""
                for c in reversed(cells[1:]):
                    if c:
                        right = c
                        break
                if left and right:
                    kv[left] = right
    return kv

def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Dict[str, str]]:
    """
    Returns (text, kv_candidates)
    """
    text, tables = extract_text_with_pdfplumber(pdf_bytes)
    if len(text) < 60:
        # looks like a scanned/poor PDF -> OCR
        text = extract_text_with_paddle(pdf_bytes)
    kv = _tables_to_kv(tables)
    return text, kv

# utils/pdf_utils.py
import contextlib, io as _io
import io, os, re, subprocess, tempfile, shutil
from typing import Dict, List, Tuple

import numpy as np

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

import pytesseract
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

_PADDLE = None

def _local_model_ok(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "inference.pdmodel")) and
        os.path.exists(os.path.join(path, "inference.pdiparams"))
    )

def _paddle():
    """Create a PaddleOCR that uses only local det/rec models (no network)."""
    global _PADDLE
    if _PADDLE is not None:
        return _PADDLE

    # Tell PaddleX/PaddleOCR we’re offline
    os.environ.setdefault("PADDLEX_OFFLINE", "1")
    os.environ.setdefault("PPOCR_OFFLINE", "1")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "paddleocr"))
    det_dir = os.environ.get("PADDLE_DET_DIR", os.path.join(root, "en_PP-OCRv3_det_infer"))
    rec_dir = os.environ.get("PADDLE_REC_DIR", os.path.join(root, "en_PP-OCRv3_rec_infer"))

    if not (_local_model_ok(det_dir) and _local_model_ok(rec_dir)):
        # Models missing -> skip Paddle and let Tesseract handle it
        _PADDLE = None
        return _PADDLE

    # Import inside a stdout/stderr redirect to hide hoster banners.
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
        from paddleocr import PaddleOCR

    # Try v3 first (has extra switches), then v2 style.
    tried = [
        # v3 API – disable all orientation helpers
        dict(det_model_dir=det_dir, rec_model_dir=rec_dir,
             lang="en",
             use_angle_cls=False,          # no angle cls
             use_doc_orient=False,         # v3: disable doc orientation (if supported)
             use_textline_orientation=False # v3: disable textline orientation (if supported)
        ),
        # v2 API – only det/rec; angle cls off
        dict(det_model_dir=det_dir, rec_model_dir=rec_dir,
             lang="en", use_angle_cls=False)
    ]
    for kwargs in tried:
        try:
            _PADDLE = PaddleOCR(**kwargs)
            break
        except TypeError:
            # Some args not supported on this version – try the next set
            continue
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
    from pdf2image import convert_from_bytes
    from PIL import ImageFilter, ImageOps

    images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png",
                                poppler_path=_POPPLER_BIN)
    o = _paddle()
    out_texts: list[str] = []

    for img in images:
        if o is None:
            out_texts.append(_tesseract_ocr(img))
            continue

        # Use Paddle det+rec only (no CLS/orientation)
        np_img = np.array(ImageOps.exif_transpose(img).convert("RGB"))
        result = o.ocr(np_img, det=True, rec=True, cls=False)

        blocks = result[0] if result else []
        lines = []
        for block in blocks:
            box, (txt, score) = block
            txt = (txt or "").strip()
            if not txt:
                continue
            if score < conf_threshold:
                # fallback OCR on low-confidence chunks (kraken/tesseract)
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

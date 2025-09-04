from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

import pytesseract
from paddleocr import PaddleOCR  # 2.x API

from tesseract_path import ensure_tesseract, auto_set_poppler_cmd

# ------------------------------------------------
# Poppler (for pdf2image) — allow absence gracefully
# ------------------------------------------------
_POPPLER_BIN: str | None
try:
    _POPPLER_BIN = auto_set_poppler_cmd()
except Exception:
    _POPPLER_BIN = None

# ------------------------------------------------
# Tesseract (fallback)
# ------------------------------------------------
try:
    ensure_tesseract()
except Exception:
    # Safe to continue; we only use it as a fallback.
    pass

# ------------------------------------------------
# Optional Kraken HTR (only if environment provides it)
# ------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_KRAKEN_MODEL = os.path.join(_PROJECT_ROOT, "models", "kraken", "en_best.mlmodel")

# Use env var if set, else default; disable if file missing
_KRAKEN_MODEL = os.getenv("KRAKEN_MODEL", _DEFAULT_KRAKEN_MODEL)
if not os.path.exists(_KRAKEN_MODEL):
    _KRAKEN_MODEL = None  # silently skip Kraken if the model file isn't there

def _kraken_cmd():
    exe = shutil.which("kraken")
    if exe:
        return [exe, "ocr"]
    # Fallback to python -m kraken if the CLI shim isn't on PATH
    return [sys.executable, "-m", "kraken", "ocr"]

def _kraken_ocr_on_image(img: Image.Image) -> str:
    if not _KRAKEN_MODEL:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        cmd = _kraken_cmd() + ["-m", _KRAKEN_MODEL, "-i", tmp.name, "-"]
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

# ------------------------------------------------
# PaddleOCR (2.x) — strictly local, det/rec only
# ------------------------------------------------
PADDLE_MODELS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "paddleocr")
)
DET_DIR = os.path.join(PADDLE_MODELS_ROOT, "en_PP-OCRv3_det_infer")
REC_DIR = os.path.join(PADDLE_MODELS_ROOT, "en_PP-OCRv3_rec_infer")

def _models_present() -> bool:
    def has_pair(d: str) -> bool:
        return (
            os.path.isdir(d)
            and os.path.exists(os.path.join(d, "inference.pdmodel"))
            and os.path.exists(os.path.join(d, "inference.pdiparams"))
        )
    return has_pair(DET_DIR) and has_pair(REC_DIR)

_PADDLE: PaddleOCR | None = None

def _paddle() -> PaddleOCR | None:
    """
    Lazy-initialize PaddleOCR 2.x with **local** det/rec models only.
    No angle classifier. No internet access required.
    """
    global _PADDLE
    if _PADDLE is not None:
        return _PADDLE

    if not _models_present():
        # Don't throw here; we'll just fall back to Tesseract.
        print(
            "[WARN] PaddleOCR local models not found. "
            f"Expected:\n  {DET_DIR}\n  {REC_DIR}\nFalling back to Tesseract."
        )
        _PADDLE = None
        return _PADDLE

    try:
        # PaddleOCR 2.x accepts numpy images and works fully offline with these dirs.
        _PADDLE = PaddleOCR(
            det_model_dir=DET_DIR,
            rec_model_dir=REC_DIR,
            use_angle_cls=False,  # keep it off so we don't need the CLS model
            lang="en",
        )
    except Exception as e:
        print(f"[WARN] PaddleOCR init failed; falling back to Tesseract. Error: {e}")
        _PADDLE = None

    return _PADDLE

# ------------------------------------------------
# PDF text + tables via pdfplumber
# ------------------------------------------------
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

# ------------------------------------------------
# Tesseract OCR on a PIL image
# ------------------------------------------------
def _tesseract_ocr(img: Image.Image) -> str:
    g = ImageOps.autocontrast(img.convert("L"))
    g = g.filter(ImageFilter.SHARPEN)
    try:
        return pytesseract.image_to_string(g, config="--oem 1 --psm 6 -l eng").strip()
    except Exception:
        return ""

# ------------------------------------------------
# PaddleOCR 2.x page OCR (with low-score fallback to Kraken/Tesseract)
# ------------------------------------------------
def extract_text_with_paddle(pdf_bytes: bytes, conf_threshold: float = 0.70) -> str:
    images = convert_from_bytes(
        pdf_bytes, dpi=300, fmt="png", poppler_path=_POPPLER_BIN
    )
    ocr = _paddle()
    out_texts: List[str] = []

    for img in images:
        if ocr is None:
            # Offline Paddle not available -> whole page with Tesseract
            out_texts.append(_tesseract_ocr(img))
            continue

        # Paddle 2.x expects numpy arrays
        arr = np.array(ImageOps.exif_transpose(img).convert("RGB"))
        # We disabled CLS at init, so keep cls=False at call to use only det+rec
        result = ocr.ocr(arr, cls=False)

        # result is: [ [ [box, (text, score)], ... ] ] for a single page
        blocks = result[0] if result else []
        lines: List[str] = []
        for block in blocks:
            try:
                box, (txt, score) = block
            except Exception:
                # Defensive: if structure changes or block malformed
                continue

            txt = (txt or "").strip()
            if not txt:
                continue

            if score is not None and score < conf_threshold:
                # Try HTR/Tesseract on the bounding box
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                l, t, r, b = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                crop = img.crop((l, t, r, b))
                htr = _kraken_ocr_on_image(crop) or _tesseract_ocr(crop)
                if htr:
                    txt = htr

            lines.append(txt)

        out_texts.append("\n".join(lines))

    return "\n\n".join(t for t in out_texts if t).strip()

# ------------------------------------------------
# Convert simple 2-col tables into KV hints
# ------------------------------------------------
def _tables_to_kv(tables: List[List[List[str]]]) -> Dict[str, str]:
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

# ------------------------------------------------
# Public entry: text + table-KV
# ------------------------------------------------
def extract_text_and_tables(pdf_bytes: bytes) -> Tuple[str, Dict[str, str]]:
    """
    Returns (text, kv_candidates). Use pdfplumber; if the text is too short
    (likely scanned), fall back to OCR with Paddle 2.x offline (then Tesseract).
    """
    text, tables = extract_text_with_pdfplumber(pdf_bytes)
    if len(text) < 60:
        # Looks like a scanned/poor PDF -> OCR
        text = extract_text_with_paddle(pdf_bytes)
    kv = _tables_to_kv(tables)
    return text, kv

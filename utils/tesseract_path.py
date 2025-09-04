# utils/tesseract_path.py
import os
import shutil
import pytesseract

def ensure_tesseract() -> str:
    """
    Ensure pytesseract can find tesseract.exe
    1) PATH
    2) ./Tesseract-OCR/tesseract.exe beside app.py
    """
    exe = shutil.which("tesseract")
    if exe:
        return exe
    local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Tesseract-OCR", "tesseract.exe"))
    if os.path.exists(local):
        pytesseract.pytesseract.tesseract_cmd = local
        return local
    raise RuntimeError("Tesseract not found. Install it or drop a 'Tesseract-OCR' folder beside app.py.")

def auto_set_poppler_cmd() -> str:
    """
    Return path to Poppler 'bin' folder if present beside app.py:
    ./Poppler-24.08.0/Library/bin
    """
    local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Poppler-24.08.0", "Library", "bin"))
    if os.path.exists(local):
        return local
    raise RuntimeError("Poppler not found. Place 'Poppler-24.08.0/Library/bin' beside app.py.")

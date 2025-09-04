# PDF Field Extractor (Loan docs)

Offline-friendly extractor for loan PDFs. It:
- reads digital text via **pdfplumber**
- OCRs scans via **PaddleOCR** (if available) with fallback to **Tesseract**
- (optional) tries **Kraken** for low-confidence handwriting snippets if `KRAKEN_MODEL` is set
- parses key fields using regex + table key/value hints (+ fuzzy matching)
- shows a comparison table and can compare to an uploaded Excel "ground truth"

## Run

```powershell
# (optional) set kraken model for handwriting
$env:KRAKEN_MODEL = (Resolve-Path .\models\kraken\en_best.mlmodel).Path
streamlit run app.py

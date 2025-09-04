# utils/field_map.py
# Full, offline-friendly field rules + helpers for loan PDFs.
# Works with utils/parser.py (regex + KV + specials + fuzzy KV).

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
from dateutil import parser as dateparser

# ===============================
# ---------- HELPERS ------------
# ===============================

# ---- text cleaners ----
def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def clean_text_one_line(s: str) -> str:
    return _collapse_ws(s)

def clean_borrower(s: str) -> str:
    s = _collapse_ws(s)
    # Remove trailing dotted leaders, colons
    return re.sub(r"[:\.\-–—\s]+$", "", s)

def clean_name(s: str) -> str:
    s = _collapse_ws(s)
    # keep letters, spaces, apostrophes, hyphens
    s = re.sub(r"[^A-Za-z \-’']", "", s)
    # normalize multiple spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    # Title-case but keep all-caps words as-is
    if s.isupper():
        return s
    return s.title()

def clean_sort_or_account(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # allow digits and -
    s = re.sub(r"[^\d\-]", "", s)
    # collapse multiple dashes/spaces
    s = re.sub(r"-{2,}", "-", s)
    return s

def _first_sentence(s: str) -> str:
    s = _collapse_ws(s)
    m = re.search(r"(.+?)(?:\.|\n|$)", s)
    return (m.group(1) if m else s).strip()

def _years_to_months(s: str) -> str:
    """Convert '3 years' to '36' (months)."""
    if not s:
        return ""
    m = re.search(r"(\d+)\s*year", s, flags=re.I)
    if not m:
        return re.sub(r"\D", "", s)  # already months?
    months = int(m.group(1)) * 12
    return str(months)

# ---- robust money parsing (₹, Rs, £, $, €, spaces, thin/nbsp, comma/dot) ----
NBSP = "\u00A0"
THINSP = "\u2009"

MONEY_RE = (
    r"(?:(?:₹|Rs\.?|£|\$|€)\s*)?\d{1,3}(?:[ \u00A0\u2009\.,]\d{3})*(?:[.,]\d{2})?"
    r"|"
    r"\d+(?:[.,]\d{2})?"
)

def _money_to_float_str(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    m = re.search(MONEY_RE, s, flags=re.IGNORECASE)
    if not m:
        return ""
    num = m.group(0)
    # strip currency + odd spaces
    num = re.sub(r"[₹£$€]|(?:Rs\.?)", "", num, flags=re.IGNORECASE)
    num = num.replace(" ", "").replace(NBSP, "").replace(THINSP, "")
    if "," in num and "." in num:
        # last separator wins as decimal
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        # comma decimal only if looks like ,dd
        if re.search(r",\d{2}$", num):
            num = num.replace(",", ".")
        else:
            num = num.replace(",", "")
    elif num.count(".") > 1:
        parts = num.split(".")
        num = "".join(parts[:-1]) + "." + parts[-1]
    return num

def _money_to_float(s: str) -> Optional[float]:
    num = _money_to_float_str(s)
    try:
        return float(num)
    except Exception:
        return None

def clean_money(s: str) -> str:
    num = _money_to_float_str(s)
    if not num:
        return ""
    try:
        return f"{float(num):,.2f}"
    except Exception:
        return f"{num}"

# ---- date helpers (normalize to 'DD Mon YYYY') ----
TZ_RE    = re.compile(r"\b(?:BST|GMT|UTC|PDT|PST|EDT|EST|CET|CEST|MDT|MST|IST)\b", re.I)
TIME_RE  = re.compile(r"\b\d{1,2}:\d{2}\b")
DATE_RE1 = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
DATE_RE2 = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b")

def _normalize_date_any(s: str) -> str:
    if not s:
        return ""
    s = s.replace("|", " ")
    try:
        dt = dateparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.strftime("%d %b %Y") if dt else ""
    except Exception:
        m = DATE_RE1.search(s) or DATE_RE2.search(s)
        if m:
            try:
                return dateparser.parse(m.group(0), dayfirst=True).strftime("%d %b %Y")
            except Exception:
                return m.group(0)
        return ""

def pick_date_in_window(win: str) -> str:
    if not win:
        return ""
    m = re.search(rf"{TIME_RE.pattern}.*?{TZ_RE.pattern}", win, flags=re.I)
    if m: return _normalize_date_any(m.group(0))
    m = TIME_RE.search(win)
    if m: return _normalize_date_any(win[m.start(): m.start()+20])
    m = DATE_RE1.search(win)
    if m: return _normalize_date_any(m.group(0))
    m = DATE_RE2.search(win)
    if m: return _normalize_date_any(m.group(0))
    return _normalize_date_any(win)

def clean_date_line(s: str) -> str:
    return _normalize_date_any(s)

# ===============================
# ---- SPECIAL EXTRACTORS -------
# ===============================
# Every special has signature: fn(text: str, kv_hints: Dict[str, str]) -> str

def _kv_get(kv: Dict[str, str], *keys) -> str:
    """Simple KV exact/contains helper."""
    if not kv:
        return ""
    # exact first
    for k in keys:
        if k in kv and kv[k]:
            return str(kv[k]).strip()
    # contains match
    norm = {k.lower(): v for k, v in kv.items()}
    for key in keys:
        kk = key.lower()
        for nk, v in norm.items():
            if kk in nk and v:
                return str(v).strip()
    return ""

def _kv_find_money_sum(kv: Dict[str, str], *contains_keys) -> str:
    """Sum money values for keys whose label contains any of contains_keys."""
    if not kv:
        return ""
    total = 0.0
    found_any = False
    for k, v in kv.items():
        lk = k.lower()
        if any(sub.lower() in lk for sub in contains_keys):
            amt = _money_to_float(v)
            if amt is not None:
                total += amt
                found_any = True
    return f"{total:,.2f}" if found_any else ""

# ---- PDF1 loan amount from KV / text ----
def kft_limit_amount(text: str, kv: Dict[str, str]) -> str:
    val = _kv_get(kv, "Limit/Amount", "Limit / Amount", "Facility Amount", "Amount")
    return clean_money(val)

def loan_amount_from_limit(text: str, kv: Dict[str, str]) -> str:
    pats = [
        rf"(?mi)Limit/Amount\s*[:\-]?\s*({MONEY_RE})",
        rf"(?mi)Facility\s*Amount\s*[:\-]?\s*({MONEY_RE})",
        rf"(?mi)\bAmount\b\s*[:\-]?\s*({MONEY_RE})",
    ]
    for p in pats:
        m = re.search(p, text)
        if m:
            return clean_money(m.group(1))
    return ""

# ---- PDF1 signatory between markers ----
def signatory_between(text: str, kv: Dict[str, str]) -> str:
    # Try common anchors around signature blocks
    # e.g., "Signed by ... Print Name: <NAME>  Date: ..."
    m = re.search(r"(?s)Print\s*Name\s*:?\s*([A-Za-z’' \-]+)\s*(?:Date|Signature|Signed|on)?", text, flags=re.I)
    if m:
        return clean_name(m.group(1))
    # fallback: Name (BLOCK CAPITALS)
    m = re.search(r"Name\s*\(BLOCK\s*CAPITALS?\)\s*:?\s*([A-Za-z’' \-]+)", text, flags=re.I)
    if m:
        return clean_name(m.group(1))
    return ""

# ---- PDF1/Generic dates ----
def date_near_label_lines(text: str, kv: Dict[str, str]) -> str:
    # look after 'Date' labels
    for lab in [r"Date\s*Signed", r"Date\s*of\s*Signature", r"Date"]:
        m = re.search(rf"{lab}\s*[:\-]?\s*(.{0,100})", text, flags=re.I)
        if m:
            cand = pick_date_in_window(m.group(1))
            if cand:
                return cand
    # look before 'Date' if spread across lines
    for m in re.finditer(r"(.{0,60})\bDate\b", text, flags=re.I):
        cand = pick_date_in_window(m.group(1))
        if cand:
            return cand
    return ""

def date_before_tz(text: str, kv: Dict[str, str]) -> str:
    # find a date before a timezone/at time
    for m in re.finditer(rf"(.{{0,80}}){TZ_RE.pattern}", text, flags=re.I):
        cand = pick_date_in_window(m.group(1))
        if cand:
            return cand
    return ""

def date_before_label(text: str, kv: Dict[str, str]) -> str:
    # look in window before 'Date' word
    for m in re.finditer(r"(.{0,60})\bDate\b", text, flags=re.I):
        cand = pick_date_in_window(m.group(1))
        if cand:
            return cand
    return ""

def date_scored(text: str, kv: Dict[str, str]) -> str:
    # simple scoring around anchors
    anchors = [r"Date\s*Signed", r"Signed\s*on", r"Signed\s*by", r"Date\s*of\s*Signature", r"\bDate\b"]
    best = ""
    for a in anchors:
        for m in re.finditer(a, text, flags=re.I):
            start = m.start()
            after  = text[start:start+180]
            before = text[max(0, start-180):start]
            cand = pick_date_in_window(after) or pick_date_in_window(before)
            if cand:
                return cand
    return best

# ---- PDF2 purpose/margin/interest type/term ----
def facility_purpose(text: str, kv: Dict[str, str]) -> str:
    val = _kv_get(kv, "Purpose", "Facility Purpose")
    if val:
        return clean_text_one_line(val).rstrip(".")
    m = re.search(r"(?mi)Facility\s*Purpose\s*[:\-]?\s*([^\n\r]+)", text)
    return clean_text_one_line(m.group(1)).rstrip(".") if m else ""

def kft_purpose(text: str, kv: Dict[str, str]) -> str:
    return facility_purpose(text, kv)

def kft_margin(text: str, kv: Dict[str, str]) -> str:
    # try KV first
    val = _kv_get(kv, "Margin")
    if val:
        return clean_text_one_line(val).rstrip(",.")
    m = re.search(r"(?mi)\bMargin\b\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?%?)", text)
    return clean_text_one_line(m.group(1)).rstrip(",.") if m else ""

def kft_interest_type(text: str, kv: Dict[str, str]) -> str:
    val = _kv_get(kv, "Interest Rate Basis", "Interest Rate Type")
    if val:
        return _first_sentence(val)
    m = re.search(r"(?mi)Interest\s*Rate\s*(?:Basis|Type)\s*[:\-]?\s*([^\n\r]+)", text)
    return _first_sentence(m.group(1)) if m else ""

def kft_product_fee(text: str, kv: Dict[str, str]) -> str:
    # treat Arrangement/Product/Processing Fee interchangeably
    for key in ["Product Fee", "Arrangement Fee", "Arrangement / Product Fee", "Processing Fee", "PF"]:
        v = _kv_get(kv, key)
        if v:
            return clean_money(v)
    m = re.search(rf"(?mi)\b(Product|Arrangement)\s*Fee\s*[:\-]?\s*({MONEY_RE})", text)
    return clean_money(m.group(2)) if m else ""

def security_fee_sum(text: str, kv: Dict[str, str]) -> str:
    # sum all items whose label contains 'security fee'
    total = _kv_find_money_sum(kv, "Security Fee")
    if total:
        return total
    # fallback: any 'Fee - Security' pattern
    m_all = re.findall(rf"(?mi)Security\s*Fee\s*[:\-]?\s*({MONEY_RE})", text)
    if m_all:
        acc = 0.0
        for s in m_all:
            f = _money_to_float(s)
            if f is not None:
                acc += f
        if acc > 0:
            return f"{acc:,.2f}"
    return ""

def loan_term(text: str, kv: Dict[str, str]) -> str:
    # "the date falling 5 years from ..." -> 60
    m = re.search(r"(?mi)\bdate\s*falling\s*(\d+)\s*year", text)
    if m:
        return str(int(m.group(1)) * 12)
    return ""

def kft_term_months(text: str, kv: Dict[str, str]) -> str:
    # KV
    v = _kv_get(kv, "Term (months)", "Loan Term", "Term")
    if v:
        v = v.strip()
        if "year" in v.lower():
            return _years_to_months(v)
        return re.sub(r"\D", "", v)
    # text
    m = re.search(r"(?mi)\bTerm\s*\(months\)\s*[:\-]?\s*(\d{2,3})", text)
    if m:
        return m.group(1)
    m = re.search(r"(?mi)\bLoan\s*Term\s*[:\-]?\s*(\d{2,3})", text)
    if m:
        return m.group(1)
    return ""

def repay_freq(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "Repayment Frequency", "Repayments Frequency")
    if v:
        return clean_text_one_line(v)
    m = re.search(r"(?mi)\bRepayment\s*Frequency\s*[:\-]?\s*([A-Za-z]+)", text)
    return clean_text_one_line(m.group(1)) if m else ""

def kft_repay_freq(text: str, kv: Dict[str, str]) -> str:
    return repay_freq(text, kv)

def cca_marker(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "CCA Marker", "CCA")
    if v:
        return clean_text_one_line(v)
    m = re.search(r"(?mi)\bCCA\s*Marker\s*[:\-]?\s*([A-Za-z]+)", text)
    return clean_text_one_line(m.group(1)) if m else ""

def kft_cca_marker(text: str, kv: Dict[str, str]) -> str:
    return cca_marker(text, kv)

def product_type1(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "Product Type", "Loan Type")
    if v:
        return clean_text_one_line(v)
    m = re.search(r"(?mi)\bLoan\s*Type\s*[:\-]?\s*([^\n\r]+)", text)
    return clean_text_one_line(m.group(1)) if m else ""

def kft_product_type(text: str, kv: Dict[str, str]) -> str:
    return product_type1(text, kv)

def kft_total_rate(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "Total Rate")
    if v:
        return clean_text_one_line(v).rstrip(",.")
    m = re.search(r"(?mi)\bTotal\s*Rate\b.*?([0-9]+(?:[.,][0-9]+)?%?)", text)
    return clean_text_one_line(m.group(1)).rstrip(",.") if m else ""

def valuation_general(text: str, kv: Dict[str, str]) -> str:
    # Keep as-is; sometimes multi-line narrative required by auditors
    # Try a simple capture after the heading
    m = re.search(r"(?mi)Valuation\s*-\s*General\s*[:\-]?\s*(.+?)(?:\n\n|\r\r|$)", text)
    return (m.group(1).strip() if m else "").strip()

def sanctioner_decision(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "CREDIT DECISION", "Sanctioner Decision")
    if v:
        return clean_text_one_line(v)
    m = re.search(r"(?mi)CREDIT\s*DECISION\s*([^\n\r]+)", text)
    return clean_text_one_line(m.group(1)) if m else ""

def solicitor_org(text: str, kv: Dict[str, str]) -> str:
    v = _kv_get(kv, "Organisation Name", "Solicitor", "Solicitor Details Organisation Name")
    if v:
        return clean_text_one_line(v)
    m = re.search(r"(?mi)Solicitor\s*Details\s*Organisation\s*Name\s*([^\n\r]+)", text)
    return clean_text_one_line(m.group(1)) if m else ""

# Registry for parser
SPECIAL_DISPATCH = {
    # PDF1
    "kft_limit_amount": kft_limit_amount,
    "loan_amount_from_limit": loan_amount_from_limit,
    "signatory_between": signatory_between,
    "date_near_label_lines": date_near_label_lines,
    "date_before_tz": date_before_tz,
    "date_before_label": date_before_label,
    "date_scored": date_scored,

    # PDF2
    "facility_purpose": facility_purpose,
    "kft_purpose": kft_purpose,
    "kft_margin": kft_margin,
    "kft_interest_type": kft_interest_type,
    "kft_product_fee": kft_product_fee,
    "security_fee_sum": security_fee_sum,
    "loan_term": loan_term,
    "kft_term_months": kft_term_months,

    # PDF3
    "repay_freq": repay_freq,
    "kft_repay_freq": kft_repay_freq,
    "cca_marker": cca_marker,
    "kft_cca_marker": kft_cca_marker,
    "product_type1": product_type1,
    "kft_product_type": kft_product_type,
    "kft_total_rate": kft_total_rate,
    "valuation_general": valuation_general,
    "sanctioner_decision": sanctioner_decision,
    "solicitor_org": solicitor_org,
}

# ===============================
# ------ FIELD DEFINITIONS ------
# ===============================

DISPLAY_ORDER = [
    "file",

    # PDF 1
    "loan_amount",
    "beneficiary",
    "sort_code",
    "account_number",
    "borrower_reference",
    "signatory_name",
    "date_signed",

    # PDF 2
    "purpose",
    "expiry_date",
    "margin",
    "interest_rate_basis",
    "repayment_instalments",
    "arrangement_fee",        # unified: also holds Product Fee
    "security_fee_total",
    "existing_security",
    "new_security_required",
    "loan_term_months",

    # PDF 3
    "customer_id",
    "credit_application_id",
    "amortisation_term_months",
    "repayment_frequency",
    "cca_marker",
    "product_type",
    "interest_rate_type",
    "total_rate",
    "valuation_general",
    "sanctioner_decision",
    "solicitor_org",
]

HEADERS_FOR_EXPORT = {
    "file": "File",

    "loan_amount": "Loan Amount",
    "beneficiary": "Name of beneficiary",
    "sort_code": "Sort Code",
    "account_number": "Account Number",
    "borrower_reference": "Borrower",
    "signatory_name": "Signatory Name",
    "date_signed": "Date Signed",

    "purpose": "Purpose",
    "expiry_date": "Expiry Date",
    "margin": "Margin",
    "interest_rate_basis": "Interest Rate Basis",
    "repayment_instalments": "Repayment Instalments",
    "arrangement_fee": "Arrangement / Product Fee",
    "security_fee_total": "Security Fee (total)",
    "existing_security": "Existing Security",
    "new_security_required": "New security required",
    "loan_term_months": "Loan Term (months)",

    "customer_id": "Customer ID",
    "credit_application_id": "Credit Application ID",
    "amortisation_term_months": "Amortisation Term (months)",
    "repayment_frequency": "Repayment Frequency",
    "cca_marker": "CCA Marker",
    "product_type": "Product Type",
    "interest_rate_type": "Interest Rate Type",
    "total_rate": "Total Rate",
    "valuation_general": "Valuation - General",
    "sanctioner_decision": "Sanctioner Decision",
    "solicitor_org": "Solicitor",
}

FIELDS = [

    # -------- PDF 1 --------
    {
        "key": "loan_amount",
        "label": "Loan Amount",
        "aliases": [
            "The borrower wishes to send the amount stated in this field",
            "Facility Amount",
            "Proposed Exposure, stated in this field",
            "Limit/Amount",
            "Amount",
            "(THE BORROWER) -",
        ],
        "patterns": [
            rf"(?mi)^\s*The\s*borrower\s*wishes\s*to\s*send\s*the\s*amount\s*stated\s*in\s*this\s*field\s*[:\-]?\s*(?P<m>{MONEY_RE})",
            rf"(?mi)^\s*stated\s*in\s*this\s*field\s*[:\-]?\s*(?P<m>{MONEY_RE})",
            rf"(?mi)^\s*(?:THE\s*BORROWER)\s*.*?\s*[:\-]?\s*(?P<m>{MONEY_RE})",
            rf"(?mi)^\s*:?\s*Facility\s*Amount\s*[:\-]?\s*(?P<m>{MONEY_RE})",
            rf"(?mi)^\s*Limits?\s*/?\s*\b(?:\n|\r)\b\s*Amount\s*[:\-]?\s*(?P<m>{MONEY_RE})",
            rf"(?mi)\bLimit/Amount\b.*?(?P<m>{MONEY_RE})",
        ],
        "cleaner": clean_money,
        "special": ["kft_limit_amount", "loan_amount_from_limit"],
    },

    {
        "key": "beneficiary",
        "label": "Name of beneficiary",
        "aliases": ["Name of beneficiary"],
        "patterns": [
            r"(?mi)^\s*Name\s*of\s*beneficiary\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": clean_text_one_line,
    },

    {
        "key": "sort_code",
        "label": "Sort Code",
        "aliases": ["Sort Code"],
        "patterns": [
            r"(?mi)^\s*Sort\s*Code\s*[:\-]?\s*([0-9 \-]+)",
        ],
        "cleaner": clean_sort_or_account,
    },

    {
        "key": "account_number",
        "label": "Account Number",
        "aliases": ["Account Number"],
        "patterns": [
            r"(?mi)^\s*Account\s*Number\s*[:\-]?\s*([0-9 \-]+)",
        ],
        "cleaner": clean_sort_or_account,
    },

    {
        "key": "borrower_reference",
        "label": "Borrower",
        "aliases": ["Payment Reference (if applicable)", "Borrower(s)", "Organisation Name"],
        "patterns": [
            r"(?mi)^\s*Payment\s*Reference\s*(?:\(if\s*applicable\))?\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Borrower\(s\)\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Organisation\s*Name\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": clean_borrower,
    },

    {
        "key": "signatory_name",
        "label": "Signatory Name",
        "aliases": ["Print Name", "Name (BLOCK CAPITALS)", "Name (BLOCK CAPITAL)"],
        "patterns": [
            r"(?mi)^\s*Print\s*Name\s*[:\-]?\s*([A-Za-z’' \-]+)",
            r"(?mi)^\s*Name\s*\(BLOCK\s*CAPITALS?\)\s*[:\-]?\s*([A-Za-z’' \-]+)",
        ],
        "cleaner": clean_name,
        "special": "signatory_between",
    },

    {
        "key": "date_signed",
        "label": "Date Signed",
        "aliases": ["Date"],
        "patterns": [
            r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        ],
        "cleaner": clean_date_line,
        "special": ["date_scored", "date_near_label_lines", "date_before_tz", "date_before_label"],
        "prefer_special": True,
    },

    # -------- PDF 2 --------
    {
        "key": "purpose",
        "label": "Purpose",
        "aliases": ["Purpose"],
        "patterns": [
            r"(?mi)^\s*Facility\s*Purpose\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Purpose\s*1\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Purpose\s*2\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": lambda s: clean_text_one_line(s).rstrip("."),
        "special": ["facility_purpose", "kft_purpose"],
    },

    {
        "key": "expiry_date",
        "label": "Expiry Date",
        "aliases": ["Final Date for Drawing", "sanction expiry date", "Expiry Date"],
        "patterns": [
            r"(?mi)^\s*Final\s*Date[s]?\s*for\s*Drawing[s]?\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*sanction\s*expiry\s*date\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Expiry\s*Date[s]?\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": clean_date_line,
    },

    {
        "key": "margin",
        "label": "Margin",
        "aliases": ["Margin"],
        "patterns": [
            r"(?mi)(?:Not\s+)?MCOB\s+regulated\s+\w+.*?\bMargin\b\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?%?)",
            r"(?mi)^\s*Margin\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?%?)",
        ],
        "cleaner": lambda s: clean_text_one_line(s).rstrip(",."),
        "special": "kft_margin",
    },

    {
        "key": "interest_rate_basis",
        "label": "Interest Rate Basis",
        "aliases": ["Interest Rate Basis", "Interest Rate Type"],
        "patterns": [
            r"(?mi)^\s*Interest\s*Rate\s*Basis\s*[:\-]?\s*([^\n\r]+)",
            r"(?mi)^\s*Interest\s*Rate\s*Type\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": _first_sentence,
        "special": "kft_interest_type",
    },

    {
        "key": "repayment_instalments",
        "label": "Repayment Instalments",
        "aliases": ["Repayment Instalments"],
        "patterns": [
            r"(?mi)^\s*Repayments?\s*Instalments?\s*.*?(\d+)\s*instal",
        ],
        "cleaner": lambda s: re.sub(r"\D", "", s or ""),
    },

    {
        "key": "arrangement_fee",   # unified (also holds Product Fee)
        "label": "Arrangement / Product Fee",
        "aliases": ["Arrangement / Product Fee", "Arrangement Fee", "Product Fee", "Arrangement/Product Fee", "Processing Fee", "PF"],
        "patterns": [
            rf"(?mi)^\s*Arrangement\s*Fee\s*[:\-]?\s*({MONEY_RE})",
            rf"(?mi)^\s*Product\s*Fee\s*[:\-]?\s*({MONEY_RE})",
        ],
        "cleaner": clean_money,
        "special": "kft_product_fee",
    },

    {
        "key": "security_fee_total",
        "label": "Security Fee (total)",
        "aliases": ["Security Fee", "Security Fees"],
        "patterns": [],
        "cleaner": clean_money,
        "special": "security_fee_sum",
    },

    {
        "key": "existing_security",
        "label": "Existing Security",
        "aliases": ["Existing Security"],
        "patterns": [],
        "cleaner": clean_text_one_line,
    },

    {
        "key": "new_security_required",
        "label": "New security required",
        "aliases": ["New security required"],
        "patterns": [
            r"(?mi)^\s*New\s*security\s*required\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": clean_text_one_line,
    },

    {
        "key": "loan_term_months",
        "label": "Loan Term (months)",
        "aliases": ["The term of the loan is", "Term (months)", "Loan Term"],
        "patterns": [
            r"(?mi)\bthe\s*date\s*falling\s*(\d+)\s*year",
            r"(?mi)^\s*Term\s*[:\-]?\s*(\d+)\s*months?\b",
            r"(?mi)^\s*Term\s*\(months\)\s*[:\-]?\s*(\d+)\b",
            r"(?mi)^\s*Loan\s*Term\s*[:\-]?\s*(\d+)\b",
        ],
        "cleaner": lambda s: _years_to_months(s) if s and "year" in s.lower() else re.sub(r"\D", "", s or ""),
        "special": ["loan_term", "kft_term_months"],
    },

    # -------- PDF 3 --------
    {
        "key": "customer_id",
        "label": "Customer ID",
        "aliases": ["Customer ID"],
        "patterns": [
            r"(?mi)^\s*Customer\s*ID\s*[:\-]?\s*([0-9]{8,})",
            r"(?mi)Customer\s*ID\s*[:\-]?\s*([0-9]{8,})",
        ],
        "cleaner": lambda s: re.sub(r"\D", "", s or ""),
    },

    {
        "key": "credit_application_id",
        "label": "Credit Application ID",
        "aliases": ["Credit Application ID"],
        "patterns": [
            r"(?mi)(?:Credit\s*)?Application\s*ID\s*[:\-]?\s*([A-Za-z0-9\-]{8,20})\b",
            r"(?mi)^\s*Credit\s*Application\s*\#?\s*[:\-]?\s*([0-9]{10,20})",
            r"(?mi)^\s*Credit\s*Application\s*ID\s*[:\-]?\s*([0-9]{11,})",
        ],
        "cleaner": lambda s: re.sub(r"\s", "", s or ""),
    },

    {
        "key": "amortisation_term_months",
        "label": "Amortisation Term (months)",
        "aliases": ["Amo Term (months)", "Amortisation Term", "Amortisation Profile"],
        "patterns": [
            r"(?mi)^\s*Amo\s*Term\s*\(months\)\s*[:\-]?\s*(\d{2,3})\b",
            r"(?mi)^\s*Amortisation\s*Term\s*\(months\)\s*[:\-]?\s*(\d{2,3})\b",
        ],
        "cleaner": lambda s: re.sub(r"\D", "", s or ""),
    },

    {
        "key": "repayment_frequency",
        "label": "Repayment Frequency",
        "aliases": ["Repayment Frequency"],
        "patterns": [],
        "cleaner": clean_text_one_line,
        "special": ["repay_freq", "kft_repay_freq"],
    },

    {
        "key": "cca_marker",
        "label": "CCA Marker",
        "aliases": ["CCA Marker"],
        "patterns": [],
        "cleaner": clean_text_one_line,
        "special": ["cca_marker", "kft_cca_marker"],
    },

    {
        "key": "product_type",
        "label": "Product Type",
        "aliases": ["Product Type", "Loan Type"],
        "patterns": [
            r"(?mi)^\s*Loan\s*Type\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": clean_text_one_line,
        "special": ["product_type1", "kft_product_type"],
    },

    {
        "key": "interest_rate_type",
        "label": "Interest Rate Type",
        "aliases": ["Interest Rate Type"],
        "patterns": [
            r"(?mi)^\s*Interest\s*Rate\s*Type\s*[:\-]?\s*([^\n\r]+)",
        ],
        "cleaner": _first_sentence,
        "special": "kft_interest_type",
    },

    {
        "key": "total_rate",
        "label": "Total Rate",
        "aliases": ["Total Rate"],
        "patterns": [
            r"(?mi)\bTotal\s*Rate\b.*?([0-9]+(?:[.,][0-9]+)?%?)",
        ],
        "cleaner": lambda s: clean_text_one_line(s).rstrip(",."),
        "special": "kft_total_rate",
    },

    {
        "key": "valuation_general",
        "label": "Valuation - General",
        "aliases": ["Valuation - General"],
        "patterns": [],
        "cleaner": lambda s: s,
        "special": "valuation_general",
    },

    {
        "key": "sanctioner_decision",
        "label": "Sanctioner Decision",
        "aliases": ["Sanctioner Decision", "CREDIT DECISION"],
        "patterns": [
            r"CREDIT\s+DECISION\s*([^\n\r]+)",
        ],
        "cleaner": clean_text_one_line,
        "special": "sanctioner_decision",
    },

    {
        "key": "solicitor_org",
        "label": "Solicitor",
        "aliases": ["Organisation Name"],
        "patterns": [
            r"Solicitor\s*Details\s*Organisation\s*Name\s*([^\n\r]+)",
        ],
        "cleaner": clean_text_one_line,
        "special": "solicitor_org",
    },
]

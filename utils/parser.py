# utils/parser.py
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

from .field_map import FIELDS, SPECIAL_DISPATCH

# ---------- helpers for fuzzy KV matching ----------
def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _kv_lookup_fuzzy(aliases: List[str], kv: Dict[str, str], min_ratio: float = 0.82) -> Tuple[str, str, float]:
    """
    Try to match any alias to kv keys by exact/startswith/contains then fuzzy.
    Returns (value, matched_key, score).
    """
    alias_norms = [_norm_label(a) for a in (aliases or []) if a]
    best_val, best_key, best = "", "", 0.0
    for k, v in (kv or {}).items():
        kn = _norm_label(k)
        for an in alias_norms:
            if not an:
                continue
            if kn == an or kn.startswith(an) or an in kn:
                return v, k, 1.0
            r = SequenceMatcher(None, kn, an).ratio()
            if r > best:
                best, best_val, best_key = r, v, k
    return (best_val, best_key, best)

# ---------- small utilities ----------
def _search_patterns(text: str, patterns: List[str]) -> str:
    for pat in patterns or []:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            groups = [g for g in m.groups() if g] if m.groups() else []
            return (groups[-1] if groups else m.group(0)).strip()
    return ""

def _apply_special(special, text: str, kv_hints: Dict[str, str]) -> str:
    if not special:
        return ""
    if isinstance(special, str):
        fn = SPECIAL_DISPATCH.get(special)
        return fn(text, kv_hints) if fn else ""
    for key in special:  # list/tuple -> try in order
        fn = SPECIAL_DISPATCH.get(key)
        if fn:
            val = fn(text, kv_hints)
            if val:
                return val
    return ""

# ---------- main ----------
def parse_fields(text: str, kv_hints: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for f in FIELDS:
        prefer_special = f.get("prefer_special", False)
        raw_val = ""

        if prefer_special:
            raw_val = _apply_special(f.get("special"), text, kv_hints)

        # 1) plain regex from body
        if not raw_val:
            raw_val = _search_patterns(text, f.get("patterns", []))

        # 2) KV hints (table extraction) with fuzzy
        if not raw_val and kv_hints:
            aliases = (f.get("aliases", []) or []) + [f.get("label", "")]
            val, _k, score = _kv_lookup_fuzzy(aliases, kv_hints, min_ratio=0.82)
            if score >= 0.82 and val:
                raw_val = val

        # 3) specials as fallback (for PDF-3 cases etc.)
        if not raw_val and not prefer_special:
            raw_val = _apply_special(f.get("special"), text, kv_hints)

        cleaner = f.get("cleaner")
        cleaned = cleaner(raw_val) if callable(cleaner) else (raw_val or "")

        if not cleaned or cleaned.strip().lower() in {"n/a", "na", "none", "null", "-", "—"}:
            cleaned = ""

        out[f["key"]] = cleaned

    return out

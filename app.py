import time
import re
import streamlit as st
import pandas as pd

from utils.pdf_utils import extract_text_and_tables
from utils.parser import parse_fields
from utils.export_utils import dataframe_to_excel_bytes
from utils.field_map import FIELDS, DISPLAY_ORDER, HEADERS_FOR_EXPORT

st.set_page_config(page_title="PDF Field Extractor", page_icon="📄", layout="wide")
st.title("📄 PDF Analyzer")

with st.sidebar:
    st.subheader("Settings")
    show_text = st.checkbox("Show raw extracted text per file", value=False)

uploaded = st.file_uploader(
    "Upload your PDFs (3–4 recommended)",
    type=["pdf"],
    accept_multiple_files=True
)

# ----- compare helper -----
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _norm_val(v) -> str:
    s = "" if v is None else str(v)
    return re.sub(r"\s+", " ", s).strip().lower()

def compare_extracted_to_expected(extracted_df: pd.DataFrame, expected_df: pd.DataFrame, key_col="File"):
    if key_col not in extracted_df.columns:
        raise ValueError(f"Key column '{key_col}' not found in extracted DF")

    ex_cols_map = {_norm_col(c): c for c in extracted_df.columns}
    exp_cols_map = {_norm_col(c): c for c in expected_df.columns}
    shared_norms = [n for n in exp_cols_map.keys() if n in ex_cols_map]
    col_pairs = [(ex_cols_map[n], exp_cols_map[n]) for n in shared_norms if ex_cols_map[n] != key_col]

    # find key
    exp_key = None
    for k in [key_col, "filename", "file", "document", "doc", "name"]:
        if k in expected_df.columns:
            exp_key = k; break
        n = _norm_col(k)
        if n in exp_cols_map:
            exp_key = exp_cols_map[n]; break
        if n in exp_cols_map.values():
            exp_key = k; break
    if exp_key is None:
        raise ValueError("Expected Excel must have a 'File' (or similar) column to join on.")

    aligned = expected_df.rename(columns={exp_key: key_col})
    mismatch_rows = []
    for _, row in extracted_df.iterrows():
        key = row[key_col]
        exp_row = aligned[aligned[key_col] == key]
        if exp_row.empty:
            for (ex_c, _exp_c) in col_pairs:
                mismatch_rows.append({
                    "File": key,
                    "Field": ex_c,
                    "Extracted": row.get(ex_c, ""),
                    "Expected": "",
                    "Match": False,
                    "Reason": "missing in expected"
                })
            continue

        exp_row = exp_row.iloc[0]
        for (ex_c, exp_c) in col_pairs:
            a = _norm_val(row.get(ex_c, ""))
            b = _norm_val(exp_row.get(exp_c, ""))
            mismatch_rows.append({
                "File": key,
                "Field": ex_c,
                "Extracted": row.get(ex_c, ""),
                "Expected": exp_row.get(exp_c, ""),
                "Match": (a == b),
                "Reason": "" if a == b else "value differs"
            })

    return pd.DataFrame(mismatch_rows), aligned

expected_xlsx = st.file_uploader(
    "Optional: upload expected Excel (for comparison)",
    type=["xlsx"],
    accept_multiple_files=False
)

if uploaded:
    rows = []
    details = []
    start_all = time.time()

    for idx, up in enumerate(uploaded, start=1):
        st.write(f"**Processing:** `{up.name}`")
        file_bytes = up.read()

        t0 = time.time()
        text, kv = extract_text_and_tables(file_bytes)
        fields = parse_fields(text, kv)
        elapsed = time.time() - t0

        # build output row using your field order + pretty headers
        row = {"File": up.name}
        for key in DISPLAY_ORDER:
            if key == "file":
                continue
            header = HEADERS_FOR_EXPORT.get(key, key)
            row[header] = fields.get(key, "")
        rows.append(row)

        details.append({
            "file": up.name,
            "seconds": round(elapsed, 2),
            "kv_hints": kv,
            "raw_text": text
        })

    total_elapsed = time.time() - start_all
    st.success(f"Done! Processed {len(uploaded)} file(s) in {total_elapsed:.2f}s.")

    if show_text:
        st.subheader("Extraction Details")
        for info in details:
            with st.expander(f"🛈 {info['file']} — took {info['seconds']}s"):
                st.markdown("**Key-Value hints from tables (if any):**")
                if info["kv_hints"]:
                    st.json(info["kv_hints"])
                else:
                    st.write("_None detected_")
                st.markdown("**Raw Text:**")
                st.text_area("Extracted Text", value=info["raw_text"], height=400, key=f"text_{info['file']}")
                st.code(info["raw_text"][:2000])

    # final comparison table
    df = pd.DataFrame(rows)

    # ensure column order matches HEADERS_FOR_EXPORT in DISPLAY_ORDER
    ordered_headers = ["File"] + [HEADERS_FOR_EXPORT.get(k, k) for k in DISPLAY_ORDER if k != "file"]
    existing_cols = [c for c in ordered_headers if c in df.columns]
    df = df[existing_cols]

    st.subheader("Comparison Table")
    st.dataframe(df, use_container_width=True)

    # Download to Excel
    excel_bytes = dataframe_to_excel_bytes(df)
    st.download_button(
        label="⬇️ Download comparison as Excel",
        data=excel_bytes,
        file_name="pdf_field_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Optional: compare with expected
    if expected_xlsx:
        st.subheader("Comparison vs Expected")
        try:
            expected_df = pd.read_excel(expected_xlsx)
            mismatches, aligned_expected = compare_extracted_to_expected(df, expected_df, key_col="File")
            total = len(mismatches)
            bad = int((~mismatches["Match"]).sum())
            good = int((mismatches["Match"]).sum())
            st.write(f"Fields compared: **{total}** | Matches: ✅ **{good}** | Mismatches: ❌ **{bad}**")

            bad_df = mismatches[~mismatches["Match"]].copy()
            st.dataframe(bad_df, use_container_width=True)

            with pd.ExcelWriter("comparison_report.xlsx", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Extracted", index=False)
                aligned_expected.to_excel(writer, sheet_name="Expected (aligned)", index=False)
                mismatches.to_excel(writer, sheet_name="Diff", index=False)

            with open("comparison_report.xlsx", "rb") as f:
                st.download_button(
                    "⬇️ Download comparison report (Excel)",
                    data=f.read(),
                    file_name="comparison_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Comparison failed: {e}")

else:
    st.info("Upload 1–4 PDFs to begin.")

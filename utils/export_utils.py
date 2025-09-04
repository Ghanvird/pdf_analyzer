# utils/export_utils.py
import io
import pandas as pd

def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Comparison") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()

# Update the Streamlit app per the new requirements:
# - Remove schema section entirely
# - Use 'nama' as the identity column (if exists), no selector UI
# - Auto-load a fixed Logistic Regression pipeline model (no upload UI)
# - Keep data uploaded on Page 1 in session_state so Page 2 doesn't need re-upload
# - Also allow uploading directly on Page 2 (optional) -> triggers instant prediction

import os, pandas as pd, numpy as np, textwrap


import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
from io import BytesIO


# ============================
# Config
# ============================
st.set_page_config(page_title="Attrition Real-time Prediction", layout="wide")

# Fixed model path (already uploaded/saved on server)
MODEL_PATHS = ["logreg_tuned.pkl", "dashboard/models/logreg_tuned.pkl"]  # try both

LOW_CUTOFF = 0.66
MID_CUTOFF = 0.73

def categorize_prob(p: float) -> str:
    if p < LOW_CUTOFF:
        return "low"
    elif p <= MID_CUTOFF:
        return "medium"
    else:
        return "high"

def ensure_state():
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = None

ensure_state()

# ============================
# Load model once, globally
# ============================
def load_fixed_model():
    # Already loaded
    if st.session_state["model"] is not None:
        return st.session_state["model"]
    last_err = None
    for p in MODEL_PATHS:
        try:
            with open(p, "rb") as f:
                try:
                    model = joblib.load(f)
                except Exception:
                    f.seek(0)
                    model = pickle.load(f)
            st.session_state["model"] = model
            return model
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Tidak bisa memuat model dari path {MODEL_PATHS}. Error terakhir: {last_err}")

# ============================
# Sidebar nav
# ============================
# Sidebar nav (bisa di-switch programatis)
st.sidebar.title("Attrition Dashboard")
page = st.sidebar.radio("Navigate", ["Upload Data", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.caption("Upload data sekali di page Upload Data. Model logistic regression akan otomatis dijalankan di page Prediction.")

# ============================
# Page 1: Upload Data
# ============================
if page == "Upload Data":
    st.title("📥 Upload Data (Prediction Input)")
    st.write("Unggah file **CSV/Excel** sebagai input prediksi. Data akan **disimpan** dan dipakai di halaman Prediction.")

    file = st.file_uploader("Drag & drop di sini atau klik untuk pilih file", type=["csv","xlsx","xls"], key="upload_input_page1")

    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xl = pd.ExcelFile(file)
            sheet_name = st.selectbox("Pilih sheet", xl.sheet_names, index=0)
            df = xl.parse(sheet_name=sheet_name)

        st.session_state["uploaded_df"] = df
        st.success("Data tersimpan di memori aplikasi. Silakan buka page Prediction.")
        st.subheader("🔎 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Total rows: {len(df):,} | Total columns: {df.shape[1]}")
    else:
        if st.session_state["uploaded_df"] is not None:
            st.info("Menggunakan data yang sudah ada di memori dari upload sebelumnya.")
            st.dataframe(st.session_state["uploaded_df"].head(50), use_container_width=True)
        else:
            st.info("Belum ada data. Silakan unggah file terlebih dahulu.")

# ============================
# Page 2: Prediction
# ============================
elif page == "Prediction":
    st.title("🧮 Real-time Prediction (Logistic Regression)")

    # init aman
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None

    df_ss = st.session_state["uploaded_df"]
    if df_ss is None:
        st.info("Belum ada data. Silakan upload dulu di **Upload Data**.")
        st.stop()  # stop halamannya, nggak lanjut ke prediksi

    df = df_ss.copy()

    # Load fixed model
    try:
        model = load_fixed_model()
        st.caption("Model: Logistic Regression pipeline (auto-loaded).")
    except Exception as e:
        st.error(f"Gagal memuat model tetap: {e}")
        st.stop()

    # Run prediction
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(df)
            proba = 1/(1+np.exp(-scores))
        else:
            preds = model.predict(df)
            proba = (pd.Series(preds).astype(int).values).astype(float)
        result = pd.DataFrame({"Attrition_Probability": proba})
    except Exception as e:
        st.error(f"Prediksi gagal. Pastikan kolom & tipe data konsisten dengan data training. Error: {e}")
        st.stop()

    # Category mapping
    result["Category"] = result["Attrition_Probability"].apply(categorize_prob)

    # Identity column: use 'nama' if exists else RowID
    if "nama" in df.columns:
        result.insert(0, "nama", df["nama"].astype(str).values)
    else:
        result.insert(0, "nama", pd.Series(range(1, len(df)+1), dtype=int).astype(str).values)

    # Metrics
    st.markdown("---")
    st.subheader("📈 Ringkasan Prediksi")
    pred_rate = float((result["Attrition_Probability"] >= 0.5).mean()) if len(result) else 0.0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Attrition Rate (≥0.5)", f"{int(pred_rate*100)}%")
    with c2:
        st.metric("Low (<0.66)", int((result["Category"]=="low").sum()))
    with c3:
        st.metric("Medium (0.66–0.73)", int((result["Category"]=="medium").sum()))
    with c4:
        st.metric("High (>0.73)", int((result["Category"]=="high").sum()))

    # Table
    st.markdown("---")
    st.subheader("📋 Daftar Karyawan & Probability + Category")
    view_df = result.copy()
    sort_desc = st.checkbox("Urutkan dari probability tertinggi", value=True)
    if sort_desc:
        view_df = view_df.sort_values("Attrition_Probability", ascending=False)
    st.dataframe(view_df, use_container_width=True, height=480)

    # Download
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    excel_bytes = buffer.getvalue()

    st.download_button(
        "⬇️ Download Hasil (Excel)",
        data=excel_bytes,
        file_name="hasil.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

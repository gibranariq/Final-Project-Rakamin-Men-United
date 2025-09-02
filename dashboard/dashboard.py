# ============================================
# Attrition Dashboard (fixed)
# - No schema
# - Identity uses 'nama' if exists
# - Fixed Logistic Regression pipeline auto-loaded (no model uploader)
# - Data uploaded once (Page: Upload Data), stored in session_state
# - Filters & search ONLY on Prediction page after result is computed
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
from io import BytesIO

# ============================
# Config
# ============================
st.set_page_config(page_title="Attrition Real-time Prediction", layout="wide")

# Path model tetap (taruh file di repo: models/logreg_tuned.pkl)
MODEL_PATHS = ["models/logreg_tuned.pkl"]

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
# Load model sekali
# ============================
def load_fixed_model():
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
st.sidebar.title("Attrition Dashboard")
page = st.sidebar.radio("Navigate", ["Upload Data", "Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("Upload data sekali di page Upload Data. Model logistic regression otomatis dijalankan di page Prediction.")

# ============================
# Page 1: Upload Data
# ============================
if page == "Upload Data":
    st.title("📥 Upload Data (Prediction Input)")
    st.write("Unggah file **CSV/Excel** sebagai input prediksi. Data disimpan ke memori aplikasi dan dipakai di halaman **Prediction**.")

    file = st.file_uploader(
        "Drag & drop di sini atau klik untuk pilih file",
        type=["csv", "xlsx", "xls"],
        key="upload_input_page1"
    )

    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                xl = pd.ExcelFile(file)
                sheet_name = st.selectbox("Pilih sheet", xl.sheet_names, index=0)
                df = xl.parse(sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
        else:
            st.session_state["uploaded_df"] = df
            st.success("✅ Data tersimpan di memori aplikasi. Buka page **Prediction** untuk menjalankan prediksi.")
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

    # Pastikan ada data di session
    df_ss = st.session_state.get("uploaded_df", None)
    if df_ss is None:
        st.info("Belum ada data. Silakan upload dulu di **Upload Data**.")
        st.stop()

    df = df_ss.copy()

    # Load fixed model
    try:
        model = load_fixed_model()
        st.caption("Model: Logistic Regression pipeline (auto-loaded).")
    except Exception as e:
        st.error(f"Gagal memuat model tetap: {e}")
        st.stop()

    # Prediksi
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(df)
            proba = 1 / (1 + np.exp(-scores))
        else:
            preds = model.predict(df)
            proba = (pd.Series(preds).astype(int).values).astype(float)
        result = pd.DataFrame({"Attrition_Probability": proba})
    except Exception as e:
        st.error(f"Prediksi gagal. Pastikan kolom & tipe data konsisten dengan data training. Error: {e}")
        st.stop()

    # Tambah kategori & identitas 'nama'
    result["Category"] = result["Attrition_Probability"].apply(categorize_prob)
    if "nama" in df.columns:
        result.insert(0, "nama", df["nama"].astype(str).values)
    else:
        # fallback RowID kalau kolom 'nama' tidak ada
        result.insert(0, "nama", pd.Series(range(1, len(df) + 1), dtype=int).astype(str).values)

    # Metrics ringkas
    st.markdown("---")
    st.subheader("📈 Ringkasan Prediksi")
    pred_rate = float((result["Attrition_Probability"] >= 0.5).mean()) if len(result) else 0.0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Attrition Rate (≥0.5)", f"{int(pred_rate*100)}%")
    with c2:
        st.metric("Low (<0.66)", int((result["Category"] == "low").sum()))
    with c3:
        st.metric("Medium (0.66–0.73)", int((result["Category"] == "medium").sum()))
    with c4:
        st.metric("High (>0.73)", int((result["Category"] == "high").sum()))

    # ============================
    # Tabel + Filter/Search + Download (ONLY on Prediction)
    # ============================
    st.markdown("---")
    st.subheader("📋 Daftar Karyawan Hasil Prediksi")

    # Filter & Search UI
    with st.expander("Filter & Search", expanded=True):
        cfs1, cfs2, cfs3 = st.columns([2, 2, 3])
        with cfs1:
            search_name = st.text_input("Cari nama (case-insensitive)", "")
        with cfs2:
            cat_selected = st.multiselect(
                "Filter kategori",
                options=["low", "medium", "high"],
                default=["low", "medium", "high"],
            )
        with cfs3:
            prob_min, prob_max = st.slider(
                "Range probability",
                min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.01
            )

    # Apply filters ke hasil prediksi
    view_df = result.copy()
    if search_name:
        view_df = view_df[view_df["nama"].str.contains(search_name, case=False, na=False)]

    if cat_selected:
        view_df = view_df[view_df["Category"].isin(cat_selected)]
    else:
        view_df = view_df.iloc[0:0]  # kosong kalau semua kategori di-unselect

    view_df = view_df[
        view_df["Attrition_Probability"].between(prob_min, prob_max, inclusive="both")
    ]

    # ---- Sorting
    sort_desc = st.checkbox("Urutkan dari probability tertinggi", value=True)
    if sort_desc and not view_df.empty:
        view_df = view_df.sort_values("Attrition_Probability", ascending=False)

    # =====================================
    # Tabel interaktif (Select maksimal 1 baris)
    # =====================================
    st.write("Centang **1** baris pada kolom Select untuk melihat detail karyawan ⬇️")

    # state untuk single-select
    if "single_select_idx" not in st.session_state:
        st.session_state["single_select_idx"] = None
    if "select_map_prev" not in st.session_state:
        st.session_state["select_map_prev"] = {}

    # siapkan data tampil + kolom Select
    view_df_with_select = view_df.copy()
    view_df_with_select.insert(0, "Select", False)

    # kalau sebelumnya sudah ada pilihan, set True di baris itu
    ssi = st.session_state["single_select_idx"]
    if ssi in view_df_with_select.index:
        view_df_with_select.loc[ssi, "Select"] = True

    # render editor: hanya kolom "Select" yang bisa diubah
    edited_df = st.data_editor(
        view_df_with_select,
        use_container_width=True,
        height=480,
        hide_index=True,
        num_rows="fixed",
        disabled=[c for c in view_df_with_select.columns if c != "Select"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Maksimal 1 baris"),
            "Attrition_Probability": st.column_config.NumberColumn(format="%.4f"),
        },
        key="pred_table",
    )

    # baca baris yang dicentang user
    current_true = list(edited_df.index[edited_df["Select"] == True])
    prev_map = st.session_state["select_map_prev"]
    prev_selected = st.session_state["single_select_idx"]

    # tentukan baris yang baru diaktifkan (true sekarang tapi sebelumnya false)
    newly_on = [i for i in current_true if not prev_map.get(i, False)]

    # normalize: hasil akhir harus 0 atau 1 baris yang True
    if len(current_true) == 0:
        # tidak ada yang terpilih
        if prev_selected is not None:
            st.session_state["single_select_idx"] = None
            st.session_state["select_map_prev"] = {i: (i in current_true) for i in view_df.index}
            st.rerun()
    elif len(current_true) == 1:
        # tepat satu yang true
        if current_true[0] != prev_selected:
            st.session_state["single_select_idx"] = current_true[0]
            st.session_state["select_map_prev"] = {i: (i in current_true) for i in view_df.index}
            st.rerun()
    else:
        # lebih dari satu dicentang -> keep yang terakhir diaktifkan (atau terakhir di list)
        chosen = newly_on[-1] if newly_on else current_true[-1]
        st.session_state["single_select_idx"] = chosen
        st.session_state["select_map_prev"] = {i: (i == chosen) for i in view_df.index}
        st.rerun()

    # ---- tampilkan detail kalau ada satu yang terpilih
    sel_idx = st.session_state["single_select_idx"]
    if sel_idx is not None and sel_idx in view_df.index:
        nama_selected = view_df.loc[sel_idx, "nama"]
        st.markdown("---")
        st.subheader(f"📑 Detail Data Karyawan: {nama_selected}")

        detail = df.loc[[sel_idx]].copy()
        detail["Attrition_Probability"] = result.loc[sel_idx, "Attrition_Probability"]
        detail["Category"] = result.loc[sel_idx, "Category"]

        

        detail_view = detail.T.reset_index()
        detail_view.columns = ["Field", "Value"]
        st.dataframe(detail_view, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada karyawan yang dipilih.")



    # =====================================
    # Download buttons (CSV + Excel) – hasil terfilter (bukan edited)
    # =====================================
    st.markdown("---")
    st.subheader("⬇️ Download Hasil Prediksi (Filtered)")

    # st.download_button(
    #     "Download CSV",
    #     data=view_df.to_csv(index=False).encode("utf-8"),
    #     file_name="attrition_predictions_filtered.csv",
    #     mime="text/csv",
    # )

    buf = BytesIO()
    view_df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "Download Excel",
        data=buf.getvalue(),
        file_name="attrition_predictions_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

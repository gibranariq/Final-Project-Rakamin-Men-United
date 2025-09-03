# ======================================================
# HR Attrition App (Multipage)
# Pages:
# 1) Homepage
# 2) Dashboard IBM (default dataset + append predicted)
# 3) Upload Data (raw input for prediction)
# 4) Prediction (real-time model pipeline)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
from io import BytesIO
import altair as alt
from pathlib import Path

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Attrition App", layout="wide")

LOW_CUTOFF = 0.66
MID_CUTOFF = 0.73
MODEL_PATHS = ["models/logreg_tuned.pkl", "logreg_tuned.pkl"]
DEFAULT_DATA_PATHS = [
    "/mnt/data/data_full.xlsx",  # if running here
    "data/data_full.xlsx",       # repo path
    "data_full.xlsx",            # cwd
]

# === Theme palette (oranye) ===
CAT_PALETTE = {
    "low":    "#FFE2C2",  # muda
    "medium": "#FFB26B",  # oranye
    "high":   "#FF7A1A",  # oranye gelap
}

# -------------------------
# Utils
# -------------------------
def categorize_prob(p: float) -> str:
    if p < LOW_CUTOFF:
        return "low"
    elif p <= MID_CUTOFF:
        return "medium"
    else:
        return "high"

def ensure_state():
    for k, v in {
        "uploaded_df": None,   # raw input for prediction page
        "model": None,
        "pred_df": None,       # default + appended predicted dataset for dashboard
        "single_select_idx": None,
        "select_map_prev": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

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
    raise RuntimeError(f"Gagal load model dari {MODEL_PATHS}. Err: {last_err}")

def find_existing_path(candidates):
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def load_default_predicted():
    """Load default predicted dataset (full features + Attrition_Probability + Category)."""
    if st.session_state["pred_df"] is not None:
        return st.session_state["pred_df"]

    path = find_existing_path(DEFAULT_DATA_PATHS)
    if path is None:
        st.warning("Default dataset tidak ditemukan. Upload di Dashboard untuk memulai.")
        st.session_state["pred_df"] = pd.DataFrame()
        return st.session_state["pred_df"]

    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalisasi kolom prediksi
    if "Attrition_Probability" not in df.columns:
        for c in df.columns:
            if c.lower() == "attrition_probability":
                df.rename(columns={c: "Attrition_Probability"}, inplace=True)
                break
    if "Category" not in df.columns:
        for c in df.columns:
            if c.lower() == "category":
                df.rename(columns={c: "Category"}, inplace=True)
                break
    if "Category" not in df.columns and "Attrition_Probability" in df.columns:
        df["Category"] = df["Attrition_Probability"].apply(categorize_prob)

    # Standarisasi beberapa kolom
    renames = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "job role":
            renames[c] = "JobRole"
        if cl == "employee name":
            renames[c] = "nama"
        if cl == "educationfield":
            renames[c] = "EducationField"
    if renames:
        df.rename(columns=renames, inplace=True)

    st.session_state["pred_df"] = df
    return df

def age_group_series(s: pd.Series) -> pd.Series:
    bins = [-np.inf, 24, 34, 44, 55, np.inf]
    labels = ["<25", "25-34", "35-44", "45-55", ">55"]
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels)

def find_col(df: pd.DataFrame, *candidates: str):
    """Cari kolom di df berdasarkan kandidat nama (case-insensitive, strip spasi)."""
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in norm:
            return norm[k]
    return None

def stacked_bar(df, x_col, stack_col="Category", title=None, sort_x="ascending", height=500):
    """Stacked bar: x & stack string; NaN -> 'Unknown'; palette oranye; height adjustable."""
    x_real = find_col(df, x_col) or x_col
    stack_real = find_col(df, stack_col) or stack_col

    if df.empty or x_real not in df.columns:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text(size=14).encode(text="msg")

    data = df.copy()
    x = pd.Series(pd.array(data[x_real], dtype="string")).fillna("Unknown")
    x = x.replace({"<NA>": "Unknown", "None": "Unknown", "nan": "Unknown"})
    data[x_real] = x

    if stack_real in data.columns:
        sc = pd.Series(pd.array(data[stack_real], dtype="string")).fillna("Unknown")
        sc = sc.replace({"<NA>": "Unknown"})
        data[stack_real] = sc
    else:
        data[stack_real] = "Unknown"

    agg = data.groupby([x_real, stack_real], dropna=False).size().reset_index(name="count")

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_real}:N", sort=(sort_x if sort_x else "ascending"), title=x_col),
            y=alt.Y("count:Q"),
            color=alt.Color(
                f"{stack_real}:N",
                scale=alt.Scale(
                    domain=["low","medium","high"],
                    range=[CAT_PALETTE["low"], CAT_PALETTE["medium"], CAT_PALETTE["high"]],
                )
            ),
            tooltip=[x_real, stack_real, "count"],
        )
        .properties(height=height)
    )
    if title is not None:
        chart = chart.properties(title=title)
    return chart

# -------------------------
# Sidebar Nav
# -------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Homepage", "Dashboard IBM", "Upload Data", "Prediction"],
    index=0
)
st.sidebar.caption("• Dashboard IBM = dataset default (bisa append)\n• Upload Data = input baru untuk diprediksi\n• Prediction = prediksi real-time pakai model")

# ======================================================
# 1) HOMEPAGE
# ======================================================
if page == "Homepage":
    st.title("👋 HR Attrition App")
    st.write(
        "- **Dashboard IBM**: ringkasan & breakdown dataset prediksi default (bisa append hasil baru).\n"
        "- **Upload Data**: unggah data baru (tanpa target) untuk diprediksi.\n"
        "- **Prediction**: jalankan model Logistic Regression (pipeline) real-time.\n"
    )

    df0 = load_default_predicted()
    if not df0.empty:
        st.markdown("### Snapshot cepat")
        total = len(df0)
        high = int((df0.get("Category","").astype(str) == "high").sum())
        med  = int((df0.get("Category","").astype(str) == "medium").sum())
        low  = int((df0.get("Category","").astype(str) == "low").sum())
        rate = float((df0.get("Attrition_Probability", pd.Series([0]*total)) >= 0.5).mean()) if total else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Employees", f"{total:,}")
        c2.metric("Attrition Rate (≥0.5)", f"{int(rate*100)}%")
        c3.metric("High", high)
        c4.metric("Medium", med)
        c5.metric("Low", low)
    else:
        st.info("Belum ada dataset default. Masuk ke **Dashboard IBM** lalu upload hasil prediksi untuk dijadikan base.")

# ======================================================
# 2) DASHBOARD IBM
# ======================================================
elif page == "Dashboard IBM":
    st.title("📊 Dashboard – IBM Predicted Dataset")

    # load or init
    base_df = load_default_predicted().copy()

    with st.expander("➕ Append hasil prediksi baru (CSV/Excel) ke dashboard", expanded=False):
        up = st.file_uploader("Upload file prediksi (harus ada kolom `Attrition_Probability` / `Category`)", type=["csv","xlsx","xls"])
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    newdf = pd.read_csv(up)
                else:
                    newdf = pd.read_excel(up)
                if "Attrition_Probability" not in newdf.columns:
                    for c in newdf.columns:
                        if c.lower() == "attrition_probability":
                            newdf.rename(columns={c: "Attrition_Probability"}, inplace=True)
                            break
                if "Category" not in newdf.columns and "Attrition_Probability" in newdf.columns:
                    newdf["Category"] = newdf["Attrition_Probability"].apply(categorize_prob)

                st.session_state["pred_df"] = pd.concat([base_df, newdf], ignore_index=True)
                base_df = st.session_state["pred_df"].copy()
                st.success(f"Appended {len(newdf)} baris. Total sekarang: {len(base_df)}")
            except Exception as e:
                st.error(f"Gagal baca/append: {e}")

        colr1, colr2 = st.columns([1,1])
        if colr1.button("🔁 Reset ke dataset default (reload file)"):
            st.session_state["pred_df"] = None
            base_df = load_default_predicted().copy()
            st.success("Reset berhasil.")

        if not base_df.empty:
            buf_all = BytesIO()
            base_df.to_excel(buf_all, index=False, engine="openpyxl")
            colr2.download_button("⬇️ Download dataset dashboard (Excel)",
                                  data=buf_all.getvalue(),
                                  file_name="dashboard_dataset.xlsx",
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if base_df.empty:
        st.stop()

    # Derived fields
    if "Age" in base_df.columns and "AgeGroup" not in base_df.columns:
        base_df["AgeGroup"] = age_group_series(base_df["Age"])
    if "nama" not in base_df.columns:
        base_df["nama"] = np.arange(1, len(base_df)+1).astype(str)

    # resolve kolom dinamis
    dept_col     = find_col(base_df, "Department")
    role_col     = find_col(base_df, "JobRole")
    marital_col  = find_col(base_df, "MaritalStatus")

    # Filters (top bar)
    st.markdown("---")
    st.subheader("Filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    cats   = f1.multiselect("Category", ["low","medium","high"], default=["low","medium","high"])
    depts  = f2.multiselect("Department", sorted(base_df.get(dept_col, pd.Series(dtype=str)).dropna().unique())) if dept_col else []
    roles  = f3.multiselect("JobRole",    sorted(base_df.get(role_col, pd.Series(dtype=str)).dropna().unique())) if role_col else []
    marits = f4.multiselect("Marital Status", sorted(base_df.get(marital_col, pd.Series(dtype=str)).dropna().unique())) if marital_col else []
    ages   = f5.multiselect("AgeGroup", ["<25","25-34","35-44","45-55",">55","Unknown"])

    view = base_df.copy()
    if cats:                    view = view[view["Category"].isin(cats)]
    if dept_col and depts:      view = view[view[dept_col].isin(depts)]
    if role_col and roles:      view = view[view[role_col].isin(roles)]
    if marital_col and marits:  view = view[view[marital_col].isin(marits)]
    if ages:
        if "AgeGroup" not in view.columns and "Age" in view.columns:
            view = view.copy()
            view["AgeGroup"] = age_group_series(view["Age"])
        view = view[view.get("AgeGroup").astype("string").fillna("Unknown").isin(ages)]

    # KPI
    st.markdown("---")
    st.subheader("Overview")
    total = len(view)
    rate = float((view.get("Attrition_Probability", pd.Series([0]*total)) >= 0.5).mean()) if total else 0.0
    high = int((view["Category"] == "high").sum())
    med  = int((view["Category"] == "medium").sum())
    low  = int((view["Category"] == "low").sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Employees (filtered)", f"{total:,}")
    k2.metric("Attrition Rate (≥0.5)", f"{int(rate*100)}%")
    k3.metric("High", high)
    k4.metric("Medium", med)
    k5.metric("Low", low)

    # Layout: Department & JobRole
    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("By Department")
        st.altair_chart(
            stacked_bar(view, dept_col or "Department",
                        title="Count by Department & Category",
                        height=500),
            use_container_width=True
        )

    with c_right:
        st.subheader("Job Role — Top 5 (by High)")

        role = role_col or "JobRole"
        if role not in view.columns:
            st.info("Kolom JobRole tidak ditemukan.")
        else:
            # hitung per kategori
            grp = (
                view.groupby([role, "Category"], dropna=False)
                    .size().unstack(fill_value=0)
            )
            for k in ["low","medium","high"]:
                if k not in grp.columns:
                    grp[k] = 0
            grp["total"] = grp["low"] + grp["medium"] + grp["high"]

            # urutkan by 'high' desc (ganti ke 'total' kalau mau total)
            top5 = grp.sort_values("high", ascending=False).head(5).reset_index()

            h1, h2 = st.columns([3,1])
            with h2:
                st.caption("Top 5")

            def badge(val, bg, fg="#111"):
                return f"""
                    <div style="
                        display:inline-block;padding:8px 12px;border-radius:12px;
                        background:{bg};color:{fg};font-weight:700;text-align:center;min-width:48px;">
                        {int(val)}
                    </div>
                """

            for i, row in top5.reset_index(drop=True).iterrows():
                rnk = i + 1
                nm  = str(row[role])

                col1, col2, col3, col4, col5 = st.columns([0.5, 3.0, 0.9, 0.9, 0.9])
                with col1:
                    st.markdown(f"<div style='font-weight:600;font-size:1rem'>{rnk}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div style='font-weight:600;font-size:1rem'>{nm}</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(badge(row["low"], CAT_PALETTE["low"]), unsafe_allow_html=True)
                with col4:
                    st.markdown(badge(row["medium"], CAT_PALETTE["medium"]), unsafe_allow_html=True)
                with col5:
                    st.markdown(badge(row["high"], CAT_PALETTE["high"], fg="#fff"), unsafe_allow_html=True)

    # Demographics
    st.markdown("---")
    st.subheader("Demographics")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.caption("Marital Status")
        if marital_col:
            st.altair_chart(
                stacked_bar(view, marital_col, title="Marital Status", height=500),
                use_container_width=True
            )
        else:
            st.info("Kolom MaritalStatus tidak ditemukan.")

    with d2:
        st.caption("Age Group")
        if "AgeGroup" not in view.columns and "Age" in view.columns:
            view = view.copy()
            view["AgeGroup"] = age_group_series(view["Age"])
        st.altair_chart(
            stacked_bar(view, "AgeGroup", title="Age Group",
                        sort_x=["<25","25-34","35-44","45-55",">55","Unknown"],
                        height=500),
            use_container_width=True
        )

    with d3:
        edu_col = find_col(view, "Education", "EducationField")
        st.caption(edu_col or "Education")
        if edu_col:
            st.altair_chart(
                stacked_bar(view, edu_col, title=str(edu_col), height=500),
                use_container_width=True
            )
        else:
            st.info("Kolom Education/EducationField tidak ditemukan.")

    # Category distribution
    st.markdown("---")
    st.subheader("Category Distribution")
    cat_counts = (view["Category"].value_counts()
                  .reindex(["low","medium","high"]).fillna(0).reset_index())
    cat_counts.columns = ["Category","count"]
    st.altair_chart(
        alt.Chart(cat_counts)
          .mark_bar()
          .encode(
              x=alt.X("Category:N", sort=["low","medium","high"]),
              y="count:Q",
              color=alt.Color("Category:N",
                              scale=alt.Scale(
                                  domain=["low","medium","high"],
                                  range=[CAT_PALETTE["low"], CAT_PALETTE["medium"], CAT_PALETTE["high"]],
                              )),
              tooltip=["Category","count"]
          )
          .properties(height=500),
        use_container_width=True
    )

    # Table preview
    st.markdown("---")
    st.subheader("Preview Data (filtered)")
    st.dataframe(view.head(100), use_container_width=True)

# ======================================================
# 3) UPLOAD DATA (RAW INPUT FOR PREDICTION)
# ======================================================
elif page == "Upload Data":
    st.title("📥 Upload Data (Prediction Input)")
    st.write("Unggah file **CSV/Excel** (tanpa target). Data akan dipakai di halaman **Prediction**.")

    file = st.file_uploader("Drag & drop di sini atau klik untuk pilih file", type=["csv","xlsx","xls"], key="upload_input_page1")
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
            st.success("✅ Data tersimpan. Buka page **Prediction** untuk prediksi.")
            st.dataframe(df.head(100), use_container_width=True)
    else:
        if st.session_state["uploaded_df"] is not None:
            st.info("Menggunakan data yang sudah ada di memori dari upload sebelumnya.")
            st.dataframe(st.session_state["uploaded_df"].head(50), use_container_width=True)
        else:
            st.info("Belum ada data. Silakan unggah file terlebih dahulu.")

# ======================================================
# 4) PREDICTION (REAL-TIME)
# ======================================================
elif page == "Prediction":
    st.title("🧮 Real-time Prediction (Logistic Regression)")

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
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # Predict
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
        st.error(f"Prediksi gagal: {e}")
        st.stop()

    result["Category"] = result["Attrition_Probability"].apply(categorize_prob)
    if "nama" in df.columns:
        result.insert(0, "nama", df["nama"].astype(str).values)
    else:
        result.insert(0, "nama", pd.Series(range(1, len(df)+1), dtype=int).astype(str).values)

    # KPI
    st.markdown("---")
    st.subheader("📈 Ringkasan Prediksi")
    pred_rate = float((result["Attrition_Probability"] >= 0.5).mean()) if len(result) else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Attrition Rate (≥0.5)", f"{int(pred_rate*100)}%")
    c2.metric("Low", int((result["Category"]=="low").sum()))
    c3.metric("Medium", int((result["Category"]=="medium").sum()))
    c4.metric("High", int((result["Category"]=="high").sum()))

    # Filters
    st.markdown("---")
    st.subheader("Filter & Search")
    f1, f2, f3 = st.columns([2,2,3])
    with f1:
        search_name = st.text_input("Cari nama", "")
    with f2:
        cat_selected = st.multiselect("Filter kategori", ["low","medium","high"], default=["low","medium","high"])
    with f3:
        prob_min, prob_max = st.slider("Range probability", 0.0, 1.0, (0.0, 1.0), 0.01)

    view_df = result.copy()
    if search_name:
        view_df = view_df[view_df["nama"].str.contains(search_name, case=False, na=False)]
    if cat_selected:
        view_df = view_df[view_df["Category"].isin(cat_selected)]
    else:
        view_df = view_df.iloc[0:0]
    view_df = view_df[view_df["Attrition_Probability"].between(prob_min, prob_max, inclusive="both")]

    if st.checkbox("Urutkan dari probability tertinggi", value=True):
        view_df = view_df.sort_values("Attrition_Probability", ascending=False)

    # Single-select via checkbox column (no manual rerun)
    st.write("Centang **1** baris pada kolom Select untuk melihat detail ⬇️")
    if "single_select_idx" not in st.session_state:
        st.session_state["single_select_idx"] = None
    if "select_map_prev" not in st.session_state:
        st.session_state["select_map_prev"] = {}

    view_df_with_select = view_df.copy()
    view_df_with_select.insert(0, "Select", False)
    ssi = st.session_state["single_select_idx"]
    if ssi in view_df_with_select.index:
        view_df_with_select.loc[ssi, "Select"] = True

    edited_df = st.data_editor(
        view_df_with_select,
        use_container_width=True,
        height=420,
        hide_index=True,
        num_rows="fixed",
        disabled=[c for c in view_df_with_select.columns if c != "Select"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Maks. 1 baris"),
            "Attrition_Probability": st.column_config.NumberColumn(format="%.4f"),
        },
        key="pred_table",
    )

    current_true = list(edited_df.index[edited_df["Select"] == True])
    prev_map = st.session_state["select_map_prev"]
    newly_on = [i for i in current_true if not prev_map.get(i, False)]

    if len(current_true) == 0:
        chosen = None
    elif len(current_true) == 1:
        chosen = current_true[0]
    else:
        chosen = newly_on[-1] if newly_on else current_true[-1]

    st.session_state["single_select_idx"] = chosen
    st.session_state["select_map_prev"] = {i: (i == chosen) for i in view_df.index}

    # Detail
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

    # Download FULL features (filtered order)
    st.markdown("---")
    st.subheader("⬇️ Download Hasil (Full Features, Filtered)")
    export_df = df.loc[view_df.index].copy()
    export_df["Attrition_Probability"] = result.loc[view_df.index, "Attrition_Probability"].values
    export_df["Category"] = result.loc[view_df.index, "Category"].values
    front = []
    if "nama" in export_df.columns: front.append("nama")
    front += ["Attrition_Probability", "Category"]
    export_df = export_df[front + [c for c in export_df.columns if c not in front]]

    st.download_button(
        "Download CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="attrition_predictions_full.csv",
        mime="text/csv",
    )
    buf = BytesIO()
    export_df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "Download Excel",
        data=buf.getvalue(),
        file_name="attrition_predictions_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ======================================================
# HR Attrition App (Multipage)
# Pages:
# 1) Homepage
# 2) Dashboard IBM (default dataset + append predicted [AUTO])
# 3) Prediction (upload + real-time model pipeline)
# ======================================================

import streamlit as st

st.set_page_config(page_title="Attrition App", layout="wide")

import pandas as pd
import numpy as np
import pickle, joblib
from io import BytesIO
import altair as alt
from pathlib import Path

# --- base dir of this script (robust for deployments) ---
APP_DIR = Path(__file__).resolve().parent

# -------------------------
# Config
# -------------------------
LOW_CUTOFF = 0.66
MID_CUTOFF = 0.73
MODEL_PATHS = ["models/logreg_tuned.pkl", "logreg_tuned.pkl"]

# default data candidates (abs & rel + secrets override)
DEFAULT_DATA_PATHS = [
    APP_DIR / "data_default" / "ibm_full.csv",
    APP_DIR / "ibm_full.csv",
    Path("data_default/ibm_full.csv"),
    Path("ibm_full.csv"),
]
if "DEFAULT_DATA_PATH" in st.secrets and st.secrets["DEFAULT_DATA_PATH"]:
    DEFAULT_DATA_PATHS.insert(0, Path(st.secrets["DEFAULT_DATA_PATH"]))  # highest priority

# -------------------------
# Config
# -------------------------
# st.set_page_config(page_title="Attrition App", layout="wide")

LOW_CUTOFF = 0.66
MID_CUTOFF = 0.73
MODEL_PATHS = ["models/logreg_tuned.pkl", "logreg_tuned.pkl"]
# DEFAULT_DATA_PATHS = ["data_default/ibm_full.csv", "ibm_full.csv"]

# === Theme palette (oranye) ===
CAT_PALETTE = {
    "low":    "#FFE2C2",  # muda
    "medium": "#FFB26B",  # oranye
    "high":   "#FF7A1A",  # oranye gelap
}

# --- Columns to drop for MODEL INPUT ONLY (preview/export tetap full) ---
DROP_COLS = [
    "EmployeeCount","StandardHours","Over18","PerformanceRating",
    "EmployeeNumber","Education","JobLevel","PercentSalaryHike","Gender",
    "YearsAtCompany","YearsWithCurrManager","NumCompaniesWorked",
    "YearsSinceLastPromotion","RelationshipSatisfaction"
]

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
        "uploaded_df": None,   # raw input for prediction page (after FE)
        "model": None,
        "pred_df": None,       # default + auto-appended predicted dataset for dashboard
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

# --- DEBUG: print lokasi file yg dicoba + isi folder app ---
def _debug_paths(app_dir: Path, candidates) -> None:
    try:
        listing = "\n".join(f"- {p.name}" for p in app_dir.glob("*"))
    except Exception:
        listing = "(gagal list directory)"
    st.info(
        "Debug default dataset:\n"
        f"- APP_DIR: {app_dir}\n"
        f"- Files in APP_DIR:\n{listing}\n"
        "- Tried paths:\n" + "\n".join(f"- {str(p)}" for p in candidates)
    )

def find_existing_path(candidates):
    for p in candidates:
        try:
            if p and Path(p).exists():
                return str(Path(p).resolve())
        except Exception:
            pass
    return None


def _harmonize_cols(df1: pd.DataFrame, df2: pd.DataFrame):
    """Union columns, preserving order of first-appearance."""
    cols = list(dict.fromkeys(list(df1.columns) + list(df2.columns)))
    return df1.reindex(columns=cols), df2.reindex(columns=cols)


@st.cache_data(show_spinner=False)
def _load_file_to_df(path: str) -> pd.DataFrame:
    p = Path(path)
    # robust ke csv/xlsx
    if p.suffix.lower() == ".csv":
        # coba beberapa encoding umum
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        # fallback default
        return pd.read_csv(p)
    else:
        return pd.read_excel(p)

def load_default_predicted():
    """Load default predicted dataset (full features + Attrition_Probability + Category)."""
    if st.session_state["pred_df"] is not None:
        return st.session_state["pred_df"]

    path = find_existing_path(DEFAULT_DATA_PATHS)
    if path is None:
        st.warning("Default dataset tidak ditemukan. Kamu bisa seed dataset lewat upload di Dashboard.")
        _debug_paths(APP_DIR, DEFAULT_DATA_PATHS)  # << tampilkan diagnosa
        st.session_state["pred_df"] = pd.DataFrame()
        return st.session_state["pred_df"]

    df = _load_file_to_df(path)

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

# ---------- Feature Engineering (for uploaded data on Prediction) ----------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah 3 fitur: ExperienceRatio, IncomePerYearExp, TenureSatisfaction."""
    out = df.copy()

    # 1) ExperienceRatio = YearsInCurrentRole / TotalWorkingYears
    yicr = find_col(out, "YearsInCurrentRole")
    twy  = find_col(out, "TotalWorkingYears")
    if yicr and twy:
        num = pd.to_numeric(out[yicr], errors="coerce")
        den = pd.to_numeric(out[twy],  errors="coerce")
        ratio = num / den.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["ExperienceRatio"] = ratio
    else:
        out["ExperienceRatio"] = 0.0  # fallback aman
        missing = []
        if not yicr: missing.append("YearsInCurrentRole")
        if not twy:  missing.append("TotalWorkingYears")
        st.warning(f"Kolom {', '.join(missing)} tidak ditemukan. ExperienceRatio di-set 0.")

    # 2) IncomePerYearExp = MonthlyIncome / (TotalWorkingYears + 1)
    mi = find_col(out, "MonthlyIncome")
    twy = find_col(out, "TotalWorkingYears")  # re-find (bisa beda case)
    if mi and twy:
        inc = pd.to_numeric(out[mi],  errors="coerce")
        yrs = pd.to_numeric(out[twy], errors="coerce").fillna(0.0) + 1.0
        v = inc / yrs
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["IncomePerYearExp"] = v
    else:
        out["IncomePerYearExp"] = 0.0
        missing = []
        if not mi:  missing.append("MonthlyIncome")
        if not twy: missing.append("TotalWorkingYears")
        st.warning(f"Kolom {', '.join(missing)} tidak ditemukan. IncomePerYearExp di-set 0.")

    # 3) TenureSatisfaction = YearsInCurrentRole * JobSatisfaction
    yicr = find_col(out, "YearsInCurrentRole")
    jsat = find_col(out, "JobSatisfaction")
    if yicr and jsat:
        a = pd.to_numeric(out[yicr], errors="coerce").fillna(0.0)
        b = pd.to_numeric(out[jsat], errors="coerce").fillna(0.0)
        out["TenureSatisfaction"] = a * b
    else:
        out["TenureSatisfaction"] = 0.0
        missing = []
        if not yicr: missing.append("YearsInCurrentRole")
        if not jsat: missing.append("JobSatisfaction")
        st.warning(f"Kolom {', '.join(missing)} tidak ditemukan. TenureSatisfaction di-set 0.")

    return out


# --- put near the top ---
def _goto(page: str):
    # callback: cukup set state; button click already triggers a rerun
    st.session_state["nav"] = page

# -------------------------
# Sidebar Nav (keyed)
# -------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Homepage", "Dashboard IBM", "Prediction"],
    key="nav",  # state-owned by this key
)
st.sidebar.caption(
    "• Dashboard IBM = dataset default (auto-append dari prediksi)\n"
    "• Prediction = upload data + prediksi real-time pakai model"
)

# ======================================================
# 1) HOMEPAGE
# ======================================================
if page == "Homepage":
    st.title("👋 HR Attrition App — Guide & Shortcuts")

    # --- Color Legend (live swatches)
    st.subheader("Category Color Legend")
    low_c, med_c, high_c = st.columns(3)
    low_c.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;padding:10px;border-radius:12px;background:{CAT_PALETTE['low']};">
            <div style="font-weight:700;">Low</div>
            <div style="opacity:.75;">(probability &lt; {LOW_CUTOFF:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    med_c.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;padding:10px;border-radius:12px;background:{CAT_PALETTE['medium']};">
            <div style="font-weight:700;">Medium</div>
            <div style="opacity:.75;">({LOW_CUTOFF:.2f} – {MID_CUTOFF:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    high_c.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;padding:10px;border-radius:12px;background:{CAT_PALETTE['high']};color:#fff;">
            <div style="font-weight:700;">High</div>
            <div style="opacity:.9;">(probability &gt; {MID_CUTOFF:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("How this app works (end-to-end)")
    st.markdown(
        f"""
1) **Prediction Page**  
   • Upload CSV/Excel **tanpa target** → auto tambah 3 fitur (*ExperienceRatio*, *IncomePerYearExp*, *TenureSatisfaction*).  
   • **DROP_COLS** hanya buat **input model** (preview/export tetap full).  
   • Model: Logistic Regression pipeline → output `Attrition_Probability` + `Category` (**Low/Med/High** sesuai legend).  
   • Hasil batch **auto-append** ke Dashboard.

2) **Dashboard IBM Page**  
   • Gabungan **dataset default** + **semua hasil prediksi** (auto-append).  
   • Filter: Category, Department, JobRole, MaritalStatus, AgeGroup.  
   • Visual: By Department, Top 5 Job Role (by High), Demographics, Category Distribution.  
   • Bisa **download** dataset dashboard (Excel).

3) **Interpretation (cutoffs)**  
   • **Low**: p < **{LOW_CUTOFF:.2f}**  
   • **Medium**: **{LOW_CUTOFF:.2f}** ≤ p ≤ **{MID_CUTOFF:.2f}**  
   • **High**: p > **{MID_CUTOFF:.2f}**  

4) **Persistence**  
   • Auto-append disimpan di `session_state` (hilang kalau hard refresh).  
   • Butuh persist ke file? nanti gue tambahin tombol **Save merged** kalau lo mau.
"""
    )

    st.markdown("---")
    st.subheader("Quick Actions")
    c1, c2 = st.columns(2)
    c1.button("➡️ Go to Dashboard", on_click=_goto, args=("Dashboard IBM",))
    c2.button("➡️ Go to Prediction", on_click=_goto, args=("Prediction",))


# ======================================================
# 2) DASHBOARD IBM (AUTO-APPEND)
# ======================================================
elif page == "Dashboard IBM":
    st.title("📊 Dashboard – IBM Predicted Dataset")

    # load or init (prefer state yang sudah di-append)
    base_df = (st.session_state.get("pred_df")
               if st.session_state.get("pred_df") is not None
               else load_default_predicted()).copy()

    # tombol reset (opsional)
    colr1, colr2 = st.columns([1,1])
    if colr1.button("🔁 Reset ke dataset default (reload file)"):
        st.session_state["pred_df"] = None
        st.success("Reset berhasil.")
        st.rerun()

    cur_df = (st.session_state.get("pred_df")
              if st.session_state.get("pred_df") is not None
              else base_df)

    if not cur_df.empty:
        buf_all = BytesIO()
        cur_df.to_excel(buf_all, index=False, engine="openpyxl")
        colr2.download_button("⬇️ Download dataset dashboard (Excel)",
                              data=buf_all.getvalue(),
                              file_name="dashboard_dataset.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    base_df = (st.session_state.get("pred_df")
               if st.session_state.get("pred_df") is not None
               else base_df)
    if base_df.empty:
        st.info("Dashboard kosong. Pergi ke tab Prediction untuk menambahkan prediksi.")
        st.stop()

    # Derived fields
    if "Age" in base_df.columns and "AgeGroup" not in base_df.columns:
        base_df["AgeGroup"] = age_group_series(base_df["Age"])

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

            # urutkan by 'high' desc
            top5 = grp.sort_values("high", ascending=False).head(5).reset_index()

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
# 3) PREDICTION (UPLOAD + REAL-TIME)
# ======================================================
elif page == "Prediction":
    st.title("🧮 Prediction (Upload → FE → Logistic Regression)")

    # ---- Upload di page ini ----
    file = st.file_uploader(
        "Upload CSV/Excel (tanpa target). Begitu upload, 3 fitur baru akan otomatis ditambahkan.",
        type=["csv","xlsx","xls"],
        key="upload_input_page_prediction"
    )

    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                raw = pd.read_csv(file)
            else:
                raw = pd.read_excel(file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        # Tambah fitur engineered
        df = add_engineered_features(raw)
        st.session_state["uploaded_df"] = df
        st.success("✅ Data tersimpan & 3 fitur baru sudah ditambahkan.")
        with st.expander("Preview data (setelah feature engineering)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
    else:
        if st.session_state["uploaded_df"] is None:
            st.info("Belum ada data. Silakan upload file untuk memulai prediksi.")
            st.stop()

    # Pakai data di state (sudah FE)
    df_full_features = st.session_state["uploaded_df"].copy()  # keep full for preview/export

    # Load fixed model
    try:
        model = load_fixed_model()
        st.caption("Model: Logistic Regression pipeline (auto-loaded).")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # ===== Predict (DROP_COLS hanya untuk model input) =====
    try:
        X = df_full_features.drop(columns=DROP_COLS, errors="ignore")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            proba = 1/(1+np.exp(-scores))
        else:
            preds = model.predict(X)
            proba = (pd.Series(preds).astype(int).values).astype(float)
        result = pd.DataFrame({"Attrition_Probability": proba}, index=df_full_features.index)
    except Exception as e:
        st.error(f"Prediksi gagal: {e}")
        st.stop()

    # categories + nama
    result["Category"] = result["Attrition_Probability"].apply(categorize_prob)
    if "nama" in df_full_features.columns:
        result.insert(0, "nama", df_full_features["nama"].astype(str).values)
    else:
        result.insert(0, "nama", pd.Series(range(1, len(df_full_features)+1), dtype=int).astype(str).values)

    # ===== AUTO-APPEND to Dashboard dataset =====
    appended = df_full_features.copy()  # full features retained
    appended["Attrition_Probability"] = result["Attrition_Probability"].values
    appended["Category"] = result["Category"].values
    if "Age" in appended.columns and "AgeGroup" not in appended.columns:
        appended["AgeGroup"] = age_group_series(appended["Age"])

    base_df = load_default_predicted().copy()
    base_df, appended = _harmonize_cols(base_df, appended)
    merged = pd.concat([base_df, appended], ignore_index=True)
    st.session_state["pred_df"] = merged  # push to dashboard state immediately

    st.success(f"✅ {len(appended)} baris berhasil diprediksi & **auto-append** ke Dashboard.")

    # KPI (hasil upload ini saja)
    st.markdown("---")
    st.subheader("📈 Ringkasan Prediksi (batch saat ini)")
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

    # Detail (gunakan df_full_features yang lengkap + engineered)
    sel_idx = st.session_state["single_select_idx"]
    if sel_idx is not None and sel_idx in view_df.index:
        nama_selected = view_df.loc[sel_idx, "nama"]
        st.markdown("---")
        st.subheader(f"📑 Detail Data Karyawan: {nama_selected}")

        detail = df_full_features.loc[[sel_idx]].copy()  # full features (engineered retained)
        detail["Attrition_Probability"] = result.loc[sel_idx, "Attrition_Probability"]
        detail["Category"] = result.loc[sel_idx, "Category"]
        detail_view = detail.T.reset_index()
        detail_view.columns = ["Field", "Value"]
        st.dataframe(detail_view, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada karyawan yang dipilih.")

    # Download FULL features (filtered order) -> sudah termasuk 3 fitur baru + prediksi
    st.markdown("---")
    st.subheader("⬇️ Download Hasil (Full Features, Filtered)")
    export_df = df_full_features.loc[view_df.index].copy()
    export_df["Attrition_Probability"] = result.loc[view_df.index, "Attrition_Probability"].values
    export_df["Category"] = result.loc[view_df.index, "Category"].values
    front = []
    if "nama" in export_df.columns: front.append("nama")
    front += ["Attrition_Probability", "Category",
              "ExperienceRatio", "IncomePerYearExp", "TenureSatisfaction"]
    front = [c for c in front if c in export_df.columns]
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

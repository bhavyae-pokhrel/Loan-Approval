import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSystem – Credit Approval",
    page_icon="🏦",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
PRIMARY   = "#1A73E8"
SUCCESS   = "#2ECC71"
DANGER    = "#E74C3C"
WARN      = "#F39C12"
BG_CARD   = "#F0F4FF"
OHE_COLS  = ["Employment_Status","Marital_Status","Loan_Purpose",
             "Property_Area","Gender","Employer_Category"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.main-title {{
    text-align: center; font-size: 2.6rem; font-weight: 700;
    color: {PRIMARY}; margin-bottom: 0;
}}
.subtitle {{
    text-align: center; font-size: 1rem; color: #666;
    margin-top: 0.2rem; margin-bottom: 1.5rem;
}}
.metric-card {{
    background: {BG_CARD}; border: 1.5px solid #c5d8ff;
    border-radius: 12px; padding: 1.2rem 1rem; text-align: center;
}}
.metric-value {{ font-size: 2rem; font-weight: 700; color: {PRIMARY}; }}
.metric-label {{ font-size: 0.85rem; color: #777; margin-top: 0.2rem; }}
.section-header {{
    font-size: 1.25rem; font-weight: 700; color: #222;
    border-left: 4px solid {PRIMARY}; padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem 0;
}}
.result-approved {{
    background: #e8f8f0; border: 2px solid {SUCCESS}; border-radius: 14px;
    padding: 1.8rem; text-align: center;
}}
.result-rejected {{
    background: #fdf0f0; border: 2px solid {DANGER}; border-radius: 14px;
    padding: 1.8rem; text-align: center;
}}
.result-title {{ font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem; }}
.result-sub   {{ font-size: 1rem; color: #555; }}
div.stButton > button {{
    background: {PRIMARY}; color: white; font-size: 1.05rem;
    font-weight: 600; border: none; border-radius: 8px;
    padding: 0.65rem 2rem; cursor: pointer; transition: background 0.2s;
    width: 100%;
}}
div.stButton > button:hover {{ background: #1558b0; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; font-weight: 600; }}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 3px solid {PRIMARY} !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ML Pipeline (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_pipeline(csv_path: str):
    df = pd.read_csv(csv_path)

    # ── Impute ────────────────────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    num_cols = df.select_dtypes(include="number").columns
    num_imp = SimpleImputer(strategy="mean")
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    df_eda = df.copy()           # keep a clean copy for EDA tab

    # ── Drop ID ───────────────────────────────────────────────────────────────
    df = df.drop("Applicant_ID", axis=1)

    # ── Encode ────────────────────────────────────────────────────────────────
    le_edu  = LabelEncoder()
    le_tgt  = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le_tgt.fit_transform(df["Loan_Approved"])   # No=0, Yes=1

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc = ohe.fit_transform(df[OHE_COLS])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(OHE_COLS), index=df.index)
    df = pd.concat([df.drop(columns=OHE_COLS), enc_df], axis=1)

    # ── Feature Engineering ───────────────────────────────────────────────────
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Train all three models ─────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes":         GaussianNB(),
        "KNN":                 KNeighborsClassifier(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_train)
        yp = m.predict(X_te)
        results[name] = {
            "model":     m,
            "accuracy":  accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp),
            "recall":    recall_score(y_test, yp),
            "f1":        f1_score(y_test, yp),
            "cm":        confusion_matrix(y_test, yp),
        }

    return results, scaler, ohe, le_edu, le_tgt, feature_cols, df_eda, X_test, y_test


def build_input_vector(inputs: dict, scaler, ohe, le_edu, feature_cols) -> np.ndarray:
    """Convert form inputs → scaled numpy array aligned to training features."""
    edu_encoded = le_edu.transform([inputs["Education_Level"]])[0]

    ohe_input = pd.DataFrame([[
        inputs["Employment_Status"],
        inputs["Marital_Status"],
        inputs["Loan_Purpose"],
        inputs["Property_Area"],
        inputs["Gender"],
        inputs["Employer_Category"],
    ]], columns=OHE_COLS)
    ohe_vec = ohe.transform(ohe_input)
    ohe_names = ohe.get_feature_names_out(OHE_COLS)

    row = {
        "Applicant_Income":    inputs["Applicant_Income"],
        "Coapplicant_Income":  inputs["Coapplicant_Income"],
        "Age":                 inputs["Age"],
        "Dependents":          inputs["Dependents"],
        "Existing_Loans":      inputs["Existing_Loans"],
        "Savings":             inputs["Savings"],
        "Collateral_Value":    inputs["Collateral_Value"],
        "Loan_Amount":         inputs["Loan_Amount"],
        "Loan_Term":           inputs["Loan_Term"],
        "Education_Level":     edu_encoded,
        "DTI_Ratio_sq":        inputs["DTI_Ratio"] ** 2,
        "Credit_Score_sq":     inputs["Credit_Score"] ** 2,
    }
    for col, val in zip(ohe_names, ohe_vec[0]):
        row[col] = val

    vec = pd.DataFrame([row])[feature_cols]
    return scaler.transform(vec)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-title">🏦 LoanSystem</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Credit Approval Dashboard</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")
    uploaded = st.file_uploader("Upload loan_approval_data.csv", type=["csv"])
    st.markdown("---")
    st.markdown("## 🤖 Active Model")
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Naive Bayes", "KNN"],
        index=0,
        help="Logistic Regression gives the best accuracy (87.5%)"
    )
    st.markdown("---")
    st.caption("Built with Streamlit · Logistic Regression / Naive Bayes / KNN")

# ── Load CSV ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "loan_approval_data.csv")
if uploaded:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uploaded.read()); tmp.flush()
    CSV_PATH = tmp.name

try:
    results, scaler, ohe, le_edu, le_tgt, feature_cols, df_eda, X_test, y_test = train_pipeline(CSV_PATH)
    active = results[model_choice]
    DATA_LOADED = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    DATA_LOADED = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
if DATA_LOADED:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", "📈 EDA", "🧪 Model Comparison", "🔍 Predict Loan"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

        total   = len(df_eda)
        approved = (df_eda["Loan_Approved"] == "Yes").sum()
        rejected = (df_eda["Loan_Approved"] == "No").sum()
        avg_inc  = df_eda["Applicant_Income"].mean()
        avg_cred = df_eda["Credit_Score"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label in zip(
            [c1, c2, c3, c4, c5],
            [total, approved, rejected, f"${avg_inc:,.0f}", f"{avg_cred:.0f}"],
            ["Total Applicants", "✅ Approved", "❌ Rejected", "Avg Income", "Avg Credit Score"]
        ):
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-header">Active Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, key, label in zip(
            [m1, m2, m3, m4],
            ["accuracy", "precision", "recall", "f1"],
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ):
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{PRIMARY}">{active[key]*100:.1f}%</div>'
                f'<div class="metric-label">{label} · {model_choice}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-header">Approval Distribution</div>', unsafe_allow_html=True)
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                values=[approved, rejected],
                names=["Approved", "Rejected"],
                color_discrete_sequence=[SUCCESS, DANGER],
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            purpose_counts = df_eda.groupby(["Loan_Purpose", "Loan_Approved"]).size().reset_index(name="count")
            fig_bar = px.bar(
                purpose_counts, x="Loan_Purpose", y="count",
                color="Loan_Approved",
                color_discrete_map={"Yes": SUCCESS, "No": DANGER},
                barmode="group",
                labels={"Loan_Purpose": "Loan Purpose", "count": "Applicants"}
            )
            fig_bar.update_layout(margin=dict(t=20, b=20), legend_title="Approved")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – EDA
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Credit Score vs Loan Approval</div>', unsafe_allow_html=True)
        fig_cs = px.histogram(
            df_eda, x="Credit_Score", color="Loan_Approved",
            barmode="overlay", nbins=30, opacity=0.75,
            color_discrete_map={"Yes": SUCCESS, "No": DANGER},
        )
        fig_cs.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_cs, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">Income Distribution by Approval</div>', unsafe_allow_html=True)
            fig_inc = px.box(
                df_eda, x="Loan_Approved", y="Applicant_Income",
                color="Loan_Approved",
                color_discrete_map={"Yes": SUCCESS, "No": DANGER},
                points="outliers"
            )
            fig_inc.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_inc, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">DTI Ratio by Approval</div>', unsafe_allow_html=True)
            fig_dti = px.box(
                df_eda, x="Loan_Approved", y="DTI_Ratio",
                color="Loan_Approved",
                color_discrete_map={"Yes": SUCCESS, "No": DANGER},
                points="outliers"
            )
            fig_dti.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_dti, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Approval Rate by Employment</div>', unsafe_allow_html=True)
            emp_rate = (
                df_eda.groupby("Employment_Status")["Loan_Approved"]
                .apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            )
            emp_rate.columns = ["Employment_Status", "Approval Rate (%)"]
            fig_emp = px.bar(
                emp_rate, x="Employment_Status", y="Approval Rate (%)",
                color="Approval Rate (%)", color_continuous_scale=["#fde8e8", SUCCESS]
            )
            fig_emp.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_emp, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Approval Rate by Property Area</div>', unsafe_allow_html=True)
            area_rate = (
                df_eda.groupby("Property_Area")["Loan_Approved"]
                .apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            )
            area_rate.columns = ["Property_Area", "Approval Rate (%)"]
            fig_area = px.bar(
                area_rate, x="Property_Area", y="Approval Rate (%)",
                color="Approval Rate (%)", color_continuous_scale=["#fde8e8", PRIMARY]
            )
            fig_area.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_area, use_container_width=True)

        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_only = df_eda.select_dtypes(include="number").drop(columns=["Applicant_ID"], errors="ignore")
        corr = num_only.corr()
        fig_hm, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 7},
                    cmap="coolwarm", ax=ax, linewidths=0.4)
        ax.set_title("Feature Correlation", fontsize=12, pad=10)
        st.pyplot(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Model Comparison
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">All Model Metrics</div>', unsafe_allow_html=True)

        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Accuracy":  f"{r['accuracy']*100:.1f}%",
                "Precision": f"{r['precision']*100:.1f}%",
                "Recall":    f"{r['recall']*100:.1f}%",
                "F1 Score":  f"{r['f1']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Metric Comparison Chart</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Model":     list(results.keys()),
            "Accuracy":  [r["accuracy"]  for r in results.values()],
            "Precision": [r["precision"] for r in results.values()],
            "Recall":    [r["recall"]    for r in results.values()],
            "F1":        [r["f1"]        for r in results.values()],
        }).melt(id_vars="Model", var_name="Metric", value_name="Score")

        fig_cmp = px.bar(
            metrics_df, x="Metric", y="Score", color="Model",
            barmode="group",
            color_discrete_sequence=[PRIMARY, WARN, DANGER],
            text_auto=".2%"
        )
        fig_cmp.update_layout(yaxis_tickformat=".0%", margin=dict(t=20, b=20))
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown('<div class="section-header">Confusion Matrix — Active Model</div>', unsafe_allow_html=True)
        cm = active["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No",    "Actual: Yes"],
            color_continuous_scale=[[0,"#fff"],[1, PRIMARY]],
            aspect="auto"
        )
        fig_cm.update_layout(margin=dict(t=20, b=20))
        col_cm, _ = st.columns([1, 1])
        with col_cm:
            st.plotly_chart(fig_cm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – Predict Loan
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">Applicant Details</div>', unsafe_allow_html=True)
        st.markdown("Fill in the applicant's information and click **Check Loan Eligibility**.")

        with st.form("loan_form"):

            st.markdown("#### 👤 Personal Information")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            age            = r1c1.number_input("Age",              min_value=18, max_value=80,   value=35)
            gender         = r1c2.selectbox("Gender",              ["Male", "Female"])
            marital_status = r1c3.selectbox("Marital Status",      ["Married", "Single"])
            dependents     = r1c4.number_input("Dependents",       min_value=0,  max_value=10,   value=0)

            st.markdown("#### 🎓 Background")
            r2c1, r2c2, r2c3 = st.columns(3)
            education      = r2c1.selectbox("Education Level",     ["Graduate", "Not Graduate"])
            employment     = r2c2.selectbox("Employment Status",   ["Salaried", "Self-employed", "Contract", "Unemployed"])
            employer_cat   = r2c3.selectbox("Employer Category",   ["Private", "Government", "MNC", "Business", "Unemployed"])

            st.markdown("#### 💰 Financial Information")
            r3c1, r3c2, r3c3 = st.columns(3)
            app_income     = r3c1.number_input("Applicant Income ($)",     min_value=0, value=50000, step=500)
            coapp_income   = r3c2.number_input("Co-applicant Income ($)",  min_value=0, value=0,     step=500)
            savings        = r3c3.number_input("Savings ($)",              min_value=0, value=10000, step=500)

            r4c1, r4c2, r4c3 = st.columns(3)
            credit_score   = r4c1.number_input("Credit Score",     min_value=300, max_value=900, value=700)
            dti_ratio      = r4c2.number_input("DTI Ratio",        min_value=0.0, max_value=1.0, value=0.35, step=0.01,
                                               help="Debt-to-Income ratio (0.0 – 1.0)")
            existing_loans = r4c3.number_input("Existing Loans",   min_value=0,  max_value=20,  value=1)

            st.markdown("#### 🏠 Loan Details")
            r5c1, r5c2, r5c3, r5c4 = st.columns(4)
            loan_amount    = r5c1.number_input("Loan Amount ($)",  min_value=1000, value=20000, step=1000)
            loan_term      = r5c2.number_input("Loan Term (months)", min_value=6, max_value=360, value=60, step=6)
            loan_purpose   = r5c3.selectbox("Loan Purpose",        ["Home", "Car", "Business", "Education", "Personal"])
            property_area  = r5c4.selectbox("Property Area",       ["Urban", "Semiurban", "Rural"])

            r6c1, r6c2 = st.columns(2)
            collateral     = r6c1.number_input("Collateral Value ($)", min_value=0, value=30000, step=1000)

            submitted = st.form_submit_button("🔍 Check Loan Eligibility")

        if submitted:
            inputs = {
                "Age":                age,
                "Gender":             gender,
                "Marital_Status":     marital_status,
                "Dependents":         dependents,
                "Education_Level":    education,
                "Employment_Status":  employment,
                "Employer_Category":  employer_cat,
                "Applicant_Income":   app_income,
                "Coapplicant_Income": coapp_income,
                "Savings":            savings,
                "Credit_Score":       credit_score,
                "DTI_Ratio":          dti_ratio,
                "Existing_Loans":     existing_loans,
                "Loan_Amount":        loan_amount,
                "Loan_Term":          loan_term,
                "Loan_Purpose":       loan_purpose,
                "Property_Area":      property_area,
                "Collateral_Value":   collateral,
            }

            try:
                vec = build_input_vector(inputs, scaler, ohe, le_edu, feature_cols)
                model = active["model"]
                pred   = model.predict(vec)[0]           # 0=No, 1=Yes
                prob   = model.predict_proba(vec)[0]     # [p_no, p_yes]
                approved = (pred == 1)

                st.markdown("---")
                if approved:
                    st.markdown(
                        f'<div class="result-approved">'
                        f'<div class="result-title" style="color:{SUCCESS}">✅ Loan Approved</div>'
                        f'<div class="result-sub">Approval Confidence: <b>{prob[1]*100:.1f}%</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-rejected">'
                        f'<div class="result-title" style="color:{DANGER}">❌ Loan Rejected</div>'
                        f'<div class="result-sub">Rejection Confidence: <b>{prob[0]*100:.1f}%</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Probability Gauge ─────────────────────────────────────────
                col_g, col_f = st.columns(2)
                with col_g:
                    st.markdown("#### 📊 Approval Probability")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob[1] * 100,
                        number={"suffix": "%", "font": {"size": 36}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": SUCCESS if approved else DANGER},
                            "steps": [
                                {"range": [0,  40], "color": "#fde8e8"},
                                {"range": [40, 65], "color": "#fff3cd"},
                                {"range": [65,100], "color": "#e8f8f0"},
                            ],
                            "threshold": {
                                "line": {"color": PRIMARY, "width": 3},
                                "thickness": 0.75, "value": 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Key Factor Summary ────────────────────────────────────────
                with col_f:
                    st.markdown("#### 📋 Submitted Profile")
                    risk_color = SUCCESS if credit_score >= 680 else (WARN if credit_score >= 600 else DANGER)
                    dti_color  = SUCCESS if dti_ratio <= 0.35 else (WARN if dti_ratio <= 0.5 else DANGER)

                    factors = {
                        "Credit Score":    (credit_score, f"{credit_score}", risk_color),
                        "DTI Ratio":       (dti_ratio,    f"{dti_ratio:.2f}", dti_color),
                        "Loan Amount":     (None,         f"${loan_amount:,}", PRIMARY),
                        "Annual Income":   (None,         f"${app_income:,}", PRIMARY),
                        "Collateral":      (None,         f"${collateral:,}", PRIMARY),
                        "Loan Term":       (None,         f"{loan_term} mo", PRIMARY),
                    }
                    for label, (_, display, color) in factors.items():
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:0.4rem 0;border-bottom:1px solid #eee;">'
                            f'<span style="color:#555">{label}</span>'
                            f'<span style="font-weight:700;color:{color}">{display}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:.85rem;'>"
    "LoanSystem · AI Credit Approval · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)

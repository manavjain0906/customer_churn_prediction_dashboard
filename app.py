import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME — "SYNAPTIX" INSPIRED DARK PURPLE UI (COMPACT)
# ============================================================
BG_DEEP      = "#0b0714"
BG_PANEL     = "#150f24"
CARD_BG      = "rgba(255,255,255,0.04)"
CARD_BORDER  = "rgba(168,124,255,0.18)"
ACCENT_1     = "#8b5cf6"
ACCENT_2     = "#c084fc"
ACCENT_GLOW  = "#a855f7"
TEXT_MAIN    = "#f3f0fa"
TEXT_MUTED   = "#a394c2"
GREEN        = "#34d399"
RED          = "#f87171"
AMBER        = "#fbbf24"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ---------- Reduce overall page padding for a tighter layout ---------- */
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1400px;
}}

/* ---------- App background ---------- */
.stApp {{
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(139,92,246,0.35), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(192,132,252,0.18), transparent 55%),
        {BG_DEEP};
    color: {TEXT_MAIN};
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_DEEP} 100%);
    border-right: 1px solid {CARD_BORDER};
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1rem !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {{
    color: {TEXT_MUTED};
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {TEXT_MAIN};
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    gap: 0.35rem;
}}

div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {{
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_MAIN} !important;
    min-height: 2.1rem !important;
}}
div[data-baseweb="select"] * {{
    color: {TEXT_MAIN} !important;
}}

/* ---------- Headings (smaller, tighter, dashboard-style) ---------- */
h1 {{
    background: linear-gradient(90deg, {ACCENT_2}, {ACCENT_1});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    font-size: 1.7rem !important;
    margin-bottom: 0.2rem !important;
}}
h2 {{
    color: {TEXT_MAIN} !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}}
h3 {{
    color: {TEXT_MAIN} !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}}
p, span, label, .stMarkdown {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
}}

/* ---------- Dividers (thinner, less vertical space) ---------- */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, {CARD_BORDER}, transparent);
    margin: 0.9rem 0 !important;
}}

/* ---------- Buttons (compact) ---------- */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg, {ACCENT_1}, {ACCENT_GLOW});
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.4rem 1.1rem;
    font-weight: 600;
    font-size: 0.85rem;
    box-shadow: 0 4px 16px rgba(139,92,246,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(168,124,255,0.5);
    color: white;
}}

/* ---------- Metric cards (compact glass tiles) ---------- */
div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
div[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: {ACCENT_1};
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-weight: 600 !important;
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT_MAIN} !important;
    font-weight: 800 !important;
    font-size: 1.25rem !important;
}}
div[data-testid="stMetricDelta"] {{
    font-weight: 600 !important;
    font-size: 0.75rem !important;
}}

/* ---------- Alerts (compact glass panels) ---------- */
div[data-testid="stAlert"] {{
    border-radius: 10px !important;
    border: 1px solid {CARD_BORDER} !important;
    backdrop-filter: blur(8px);
    padding: 0.6rem 0.9rem !important;
    font-size: 0.85rem !important;
}}

/* ---------- DataFrames / tables ---------- */
div[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {CARD_BORDER};
}}

/* ---------- Progress bar ---------- */
div[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2}) !important;
}}
div[data-testid="stProgress"] {{
    margin-bottom: 0.3rem;
}}

/* ---------- Section "glass panel" wrapper class ---------- */
.glass-panel {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 6px 22px rgba(0,0,0,0.3);
}}
.glass-panel p {{
    font-size: 0.85rem;
    margin: 0.25rem 0;
}}
.glass-panel h3 {{
    font-size: 0.95rem !important;
}}

/* ---------- Reduce default block vertical gaps ---------- */
div[data-testid="stVerticalBlock"] > div {{
    gap: 0.5rem;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {BG_DEEP}; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT_1}; border-radius: 10px; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Matplotlib dark theme, tuned for small compact tiles
mpl.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": TEXT_MUTED,
    "axes.labelcolor": TEXT_MAIN,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "text.color": TEXT_MAIN,
    "axes.titlecolor": TEXT_MAIN,
    "grid.color": "#3a2f55",
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
PURPLE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("synaptix", ["#2a1a4a", ACCENT_1, ACCENT_2])

# Small helper so every chart is compact + tightly cropped
def compact_fig(figsize=(4, 2.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    return fig, ax

def show_compact(fig):
    fig.tight_layout(pad=1.0)
    st.pyplot(fig, transparent=True, use_container_width=True)
    plt.close(fig)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/churn_model.pkl")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/telo_churn.csv")

# ============================================================
# DATA PREPROCESSING
# ============================================================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ============================================================
# TITLE / HERO
# ============================================================

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.1rem;">
        <span style="font-size:1.5rem;">🪐</span>
        <h1 style="margin:0;">Customer Churn Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='margin-top:0;'>Predict customer churn using Machine Learning and analyze customer behavior through an interactive dashboard.</p>",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.6rem;">
        <span style="font-size:1.2rem;">👤</span>
        <span style="font-size:1rem; font-weight:700; color:#f3f0fa;">Customer Information</span>
    </div>
    """,
    unsafe_allow_html=True
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

# ============================================================
# CUSTOMER DATA
# ============================================================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# ============================================================
# CUSTOMER CHURN PREDICTION
# ============================================================

st.header("🔮 Customer Churn Prediction")

if st.button("🔮 Predict Churn"):

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]
    probability_percentage = probability * 100

    pred_col1, pred_col2 = st.columns([2, 1])

    with pred_col1:
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK — {probability_percentage:.2f}%")
            st.caption("This customer is predicted to churn.")
        else:
            st.success(f"✅ LOW CHURN RISK — {probability_percentage:.2f}%")
            st.caption("This customer is predicted to stay.")

        st.progress(min(int(probability_percentage), 100))
        st.caption(f"Probability of Churn: **{probability_percentage:.2f}%**")

# ============================================================
# MACHINE LEARNING DATA
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()
st.header("📈 Model Performance")
st.caption("Evaluation of the Machine Learning model on the unseen test dataset.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
with col2:
    st.metric("Precision", f"{precision * 100:.2f}%")
with col3:
    st.metric("Recall", f"{recall * 100:.2f}%")
with col4:
    st.metric("F1 Score", f"{f1 * 100:.2f}%")
with col5:
    st.metric("ROC-AUC", f"{roc_auc:.3f}")

# ============================================================
# CONFUSION MATRIX + ROC CURVE (side by side, compact)
# ============================================================

st.divider()
st.header("📊 Model Evaluation")

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = compact_fig((4.2, 3.6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax_cm, colorbar=False, cmap=PURPLE_CMAP)
    for text in ax_cm.texts:
        text.set_color("white")
        text.set_fontsize(9)
    ax_cm.set_title("Customer Churn Confusion Matrix")
    show_compact(fig_cm)

with eval_col2:
    st.subheader("📈 ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_probability)
    fig_roc, ax_roc = compact_fig((4.2, 3.6))
    ax_roc.plot(fpr, tpr, linewidth=2, color=ACCENT_2, label=f"Model (AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=TEXT_MUTED, label="Random")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.05)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    legend = ax_roc.legend(loc="lower right")
    legend.get_frame().set_alpha(0)
    for text in legend.get_texts():
        text.set_color(TEXT_MAIN)
    ax_roc.grid(True, alpha=0.3)
    show_compact(fig_roc)

# ============================================================
# CUSTOMER CHURN ANALYSIS
# ============================================================

st.divider()
st.header("📊 Customer Churn Analysis")

churn_rate = df["Churn"].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Churned Customers", int(df["Churn"].sum()))
with col3:
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    st.subheader("📄 Churn Rate by Contract")
    contract_churn = df.groupby("Contract")["Churn"].mean().mul(100)
    st.bar_chart(contract_churn, color=ACCENT_1, height=220)

with bar_col2:
    st.subheader("🌐 Churn Rate by Internet Service")
    internet_churn = df.groupby("InternetService")["Churn"].mean().mul(100)
    st.bar_chart(internet_churn, color=ACCENT_1, height=220)

bar_col3, bar_col4 = st.columns(2)

with bar_col3:
    st.subheader("📅 Churn Rate by Tenure")
    tenure_churn = df.groupby("tenure")["Churn"].mean().mul(100)
    st.line_chart(tenure_churn, color=ACCENT_2, height=220)

with bar_col4:
    st.subheader("💳 Churn Rate by Payment Method")
    payment_churn = df.groupby("PaymentMethod")["Churn"].mean().mul(100)
    st.bar_chart(payment_churn, color=ACCENT_1, height=220)

st.subheader("💡 Business Insights")

highest_contract = contract_churn.idxmax()
highest_contract_rate = contract_churn.max()
highest_internet = internet_churn.idxmax()
highest_internet_rate = internet_churn.max()
highest_payment = payment_churn.idxmax()
highest_payment_rate = payment_churn.max()

st.markdown(
    f"""
    <div class="glass-panel">
    <h3 style="margin-top:0;">🔍 Key Findings</h3>
    <p><b style="color:#f3f0fa;">Overall Churn Rate:</b> {churn_rate:.2f}%</p>
    <p><b style="color:#f3f0fa;">Highest Churn Contract:</b> {highest_contract} ({highest_contract_rate:.2f}%)</p>
    <p><b style="color:#f3f0fa;">Highest Churn Internet Service:</b> {highest_internet} ({highest_internet_rate:.2f}%)</p>
    <p><b style="color:#f3f0fa;">Highest Churn Payment Method:</b> {highest_payment} ({highest_payment_rate:.2f}%)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# INTERACTIVE ANALYSIS
# ============================================================

st.divider()
st.header("🎛️ Interactive Churn Analysis")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_contract = st.selectbox(
        "📄 Contract",
        ["All"] + sorted(df["Contract"].dropna().unique().tolist()),
        key="contract_filter"
    )
with filter_col2:
    selected_internet = st.selectbox(
        "🌐 Internet Service",
        ["All"] + sorted(df["InternetService"].dropna().unique().tolist()),
        key="internet_filter"
    )
with filter_col3:
    selected_gender = st.selectbox(
        "👤 Gender",
        ["All"] + sorted(df["gender"].dropna().unique().tolist()),
        key="gender_filter"
    )

filtered_df = df.copy()
if selected_contract != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
if selected_internet != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]

st.subheader("👥 Filtered Customer Summary")

total_customers = len(filtered_df)
churned_customers = int(filtered_df["Churn"].sum())
stayed_customers = total_customers - churned_customers
filtered_churn_rate = (filtered_df["Churn"].mean() * 100) if total_customers > 0 else 0

summary1, summary2, summary3, summary4 = st.columns(4)
with summary1:
    st.metric("👥 Customers", total_customers)
with summary2:
    st.metric("⚠️ Churned", churned_customers)
with summary3:
    st.metric("✅ Stayed", stayed_customers)
with summary4:
    st.metric("📊 Churn Rate", f"{filtered_churn_rate:.2f}%")

dist_col, table_col = st.columns([1, 2])

with dist_col:
    st.subheader("📊 Churn Distribution")
    if total_customers > 0:
        chart_data = pd.DataFrame({
            "Customer Status": ["Stayed", "Churned"],
            "Number of Customers": [stayed_customers, churned_customers]
        })
        st.bar_chart(chart_data.set_index("Customer Status"), color=ACCENT_1, height=220)
    else:
        st.warning("⚠️ No customers match the selected filters.")

with table_col:
    st.subheader("📋 Filtered Customer Data")
    if total_customers > 0:
        st.dataframe(filtered_df.head(100), use_container_width=True, height=260)

# ============================================================
# CUSTOMER RISK SEGMENTATION
# ============================================================

st.divider()
st.header("🎯 Customer Risk Segmentation")
st.caption("Customers are classified according to their predicted probability of churn.")

all_customer_probabilities = model.predict_proba(X)[:, 1]

def classify_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

risk_categories = [classify_risk(p) for p in all_customer_probabilities]

risk_df = df.copy()
risk_df["Churn Probability"] = all_customer_probabilities * 100
risk_df["Risk Category"] = risk_categories

high_risk = (risk_df["Risk Category"] == "High Risk").sum()
medium_risk = (risk_df["Risk Category"] == "Medium Risk").sum()
low_risk = (risk_df["Risk Category"] == "Low Risk").sum()

risk_col1, risk_col2, risk_col3 = st.columns(3)
with risk_col1:
    st.metric("🔴 High Risk", int(high_risk))
with risk_col2:
    st.metric("🟠 Medium Risk", int(medium_risk))
with risk_col3:
    st.metric("🟢 Low Risk", int(low_risk))

total_risk_customers = len(risk_df)
if total_risk_customers > 0:
    high_percentage = high_risk / total_risk_customers * 100
    medium_percentage = medium_risk / total_risk_customers * 100
    low_percentage = low_risk / total_risk_customers * 100
else:
    high_percentage = medium_percentage = low_percentage = 0

risk_chart_col, risk_pct_col = st.columns([1, 1])

with risk_chart_col:
    st.subheader("📊 Risk Distribution")
    risk_chart = pd.DataFrame({
        "Risk Category": ["High Risk", "Medium Risk", "Low Risk"],
        "Customers": [int(high_risk), int(medium_risk), int(low_risk)]
    })
    st.bar_chart(risk_chart.set_index("Risk Category"), color=ACCENT_1, height=200)

with risk_pct_col:
    st.subheader("📈 Risk Percentage")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("🔴 High %", f"{high_percentage:.1f}%")
    with p2:
        st.metric("🟠 Medium %", f"{medium_percentage:.1f}%")
    with p3:
        st.metric("🟢 Low %", f"{low_percentage:.1f}%")

st.subheader("🚨 High-Risk Customers")

high_risk_customers = risk_df[risk_df["Risk Category"] == "High Risk"].copy()
high_risk_customers = high_risk_customers.sort_values(by="Churn Probability", ascending=False)

if len(high_risk_customers) > 0:
    risk_columns = ["gender", "tenure", "Contract", "InternetService", "MonthlyCharges", "Churn Probability", "Risk Category"]
    available_risk_columns = [c for c in risk_columns if c in high_risk_customers.columns]
    st.dataframe(high_risk_customers[available_risk_columns].head(50), use_container_width=True, height=220)
else:
    st.success("✅ No high-risk customers were identified.")

# ============================================================
# CUSTOMER RETENTION ACTION CENTER
# ============================================================

st.divider()
st.header("🎯 Customer Retention Action Center")
st.caption("Use predicted churn risk to identify customers who need retention action.")

priority_col1, priority_col2, priority_col3 = st.columns(3)
with priority_col1:
    st.metric("🔴 Immediate Attention", int(high_risk))
with priority_col2:
    st.metric("🟠 Monitor", int(medium_risk))
with priority_col3:
    st.metric("🟢 Maintain", int(low_risk))

select_col, action_col = st.columns([1, 2])

with select_col:
    st.subheader("🔍 Select by Risk")
    selected_risk = st.selectbox(
        "Choose Risk Category",
        ["High Risk", "Medium Risk", "Low Risk", "All Customers"],
        key="retention_risk_filter"
    )

if selected_risk == "All Customers":
    retention_df = risk_df.copy()
else:
    retention_df = risk_df[risk_df["Risk Category"] == selected_risk].copy()

retention_df = retention_df.sort_values(by="Churn Probability", ascending=False)

with action_col:
    st.subheader("💡 Recommended Action")
    if selected_risk == "High Risk":
        st.error("🔴 Contact immediately • Personalized offers • Investigate service issues • Loyalty benefits • Encourage longer contracts")
    elif selected_risk == "Medium Risk":
        st.warning("🟠 Monitor behavior • Send offers • Improve engagement • Promote long-term contracts")
    elif selected_risk == "Low Risk":
        st.success("🟢 Maintain service quality • Loyalty rewards • Continue engagement")
    else:
        st.info("📊 Select a risk category to see recommended actions.")

selected_customer_count = len(retention_df)
average_probability = retention_df["Churn Probability"].mean() if selected_customer_count > 0 else 0

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.metric("Selected Customers", selected_customer_count)
with summary_col2:
    st.metric("Average Churn Probability", f"{average_probability:.2f}%")

st.subheader("📋 Customers Requiring Action")

if selected_customer_count > 0:
    retention_columns = ["gender", "tenure", "Contract", "InternetService", "MonthlyCharges", "TotalCharges", "Churn Probability", "Risk Category"]
    available_columns = [c for c in retention_columns if c in retention_df.columns]
    display_retention_df = retention_df[available_columns].copy()
    st.dataframe(display_retention_df, use_container_width=True, height=240)

    csv_data = display_retention_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Customer List as CSV",
        data=csv_data,
        file_name="customer_retention_list.csv",
        mime="text/csv"
    )
else:
    st.info("No customers found for the selected category.")

# ============================================================
# CHURN TREND & CUSTOMER VALUE ANALYSIS
# ============================================================

st.divider()
st.header("📊 Churn Trend & Customer Value Analysis")
st.caption("Visual analysis of customer churn based on tenure and monthly charges.")

st.subheader("💰 Customer Value Overview")

average_monthly_charges = df["MonthlyCharges"].mean()
average_total_charges = df["TotalCharges"].mean()
average_tenure = df["tenure"].mean()

value_col1, value_col2, value_col3 = st.columns(3)
with value_col1:
    st.metric("Avg Monthly Charges", f"${average_monthly_charges:.2f}")
with value_col2:
    st.metric("Avg Total Charges", f"${average_total_charges:.2f}")
with value_col3:
    st.metric("Avg Tenure", f"{average_tenure:.1f} mo")

def tenure_group(t):
    if t <= 12:
        return "0–12 Mo"
    elif t <= 24:
        return "13–24 Mo"
    elif t <= 48:
        return "25–48 Mo"
    else:
        return "49+ Mo"

df_analysis = df.copy()
df_analysis["Tenure Group"] = df_analysis["tenure"].apply(tenure_group)
tenure_order = ["0–12 Mo", "13–24 Mo", "25–48 Mo", "49+ Mo"]

def charge_group(c):
    if c < 40:
        return "Low (<$40)"
    elif c < 70:
        return "Med ($40-70)"
    elif c < 100:
        return "High ($70-100)"
    else:
        return "V.High (>$100)"

df_analysis["Charge Group"] = df_analysis["MonthlyCharges"].apply(charge_group)
charge_order = ["Low (<$40)", "Med ($40-70)", "High ($70-100)", "V.High (>$100)"]

# --- 2x2 compact chart grid ---
grid_col1, grid_col2 = st.columns(2)

with grid_col1:
    st.subheader("📅 Churn Rate by Tenure Group")
    tenure_churn = df_analysis.groupby("Tenure Group")["Churn"].mean().mul(100).reindex(tenure_order)
    fig1, ax1 = compact_fig((4.4, 2.8))
    ax1.bar(tenure_churn.index, tenure_churn.values, color=ACCENT_1)
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_ylim(0, max(tenure_churn.values) + 10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(tenure_churn.values):
        ax1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=TEXT_MAIN)
    show_compact(fig1)

with grid_col2:
    st.subheader("💳 Churn Rate by Charge Group")
    charge_churn = df_analysis.groupby("Charge Group")["Churn"].mean().mul(100).reindex(charge_order)
    fig2, ax2 = compact_fig((4.4, 2.8))
    ax2.bar(charge_churn.index, charge_churn.values, color=ACCENT_2)
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_ylim(0, max(charge_churn.values) + 10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(charge_churn.values):
        ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=TEXT_MAIN)
    show_compact(fig2)

grid_col3, grid_col4 = st.columns(2)

with grid_col3:
    st.subheader("📈 Churn Trend by Exact Tenure")
    exact_tenure_churn = df_analysis.groupby("tenure")["Churn"].mean().mul(100)
    fig3, ax3 = compact_fig((4.4, 2.8))
    ax3.plot(exact_tenure_churn.index, exact_tenure_churn.values, marker="o", markersize=3,
             linewidth=1.5, color=ACCENT_2, markerfacecolor=ACCENT_1)
    ax3.set_xlabel("Tenure (Months)")
    ax3.set_ylabel("Churn Rate (%)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    show_compact(fig3)

with grid_col4:
    st.subheader("📊 Monthly Charges vs Churn")
    monthly_charge_churn = df_analysis.groupby("MonthlyCharges")["Churn"].mean().mul(100)
    fig4, ax4 = compact_fig((4.4, 2.8))
    ax4.scatter(monthly_charge_churn.index, monthly_charge_churn.values, alpha=0.7, s=14,
                color=ACCENT_1, edgecolors=ACCENT_2, linewidths=0.4)
    ax4.set_xlabel("Monthly Charges ($)")
    ax4.set_ylabel("Churn Rate (%)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    show_compact(fig4)

st.subheader("💰 High-Value Customers at Risk")

high_value_threshold = df["MonthlyCharges"].quantile(0.75)
high_value_customers = risk_df[
    (risk_df["MonthlyCharges"] >= high_value_threshold) & (risk_df["Churn Probability"] >= 50)
].copy()
high_value_customers = high_value_customers.sort_values(by="Churn Probability", ascending=False)

if len(high_value_customers) > 0:
    high_value_columns = ["gender", "tenure", "Contract", "InternetService", "MonthlyCharges", "TotalCharges", "Churn Probability", "Risk Category"]
    available_columns = [c for c in high_value_columns if c in high_value_customers.columns]
    st.dataframe(high_value_customers[available_columns].head(50), use_container_width=True, height=220)
else:
    st.success("✅ No high-value customers with high churn probability were identified.")

st.subheader("💡 Business Insights")

highest_tenure_group = tenure_churn.idxmax()
highest_tenure_rate = tenure_churn.max()
highest_charge_group = charge_churn.idxmax()
highest_charge_rate = charge_churn.max()

st.markdown(
    f"""
    <div class="glass-panel">
    <h3 style="margin-top:0;">🔍 Key Findings</h3>
    <p>📅 <b style="color:#f3f0fa;">Highest Churn Tenure Group:</b> {highest_tenure_group} — {highest_tenure_rate:.2f}%</p>
    <p>💳 <b style="color:#f3f0fa;">Highest Churn Charge Group:</b> {highest_charge_group} — {highest_charge_rate:.2f}%</p>
    <p>💰 <b style="color:#f3f0fa;">Average Monthly Charges:</b> ${average_monthly_charges:.2f}</p>
    <p>💰 <b style="color:#f3f0fa;">Average Total Charges:</b> ${average_total_charges:.2f}</p>
    <p>📅 <b style="color:#f3f0fa;">Average Customer Tenure:</b> {average_tenure:.1f} months</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("🎯 Retention Recommendations")

st.markdown(
    """
    <div class="glass-panel">
    <p><b style="color:#f3f0fa;">1️⃣ Focus on new customers</b> — stronger onboarding and engagement in year one.</p>
    <p><b style="color:#f3f0fa;">2️⃣ Monitor high-paying customers</b> — priority retention offers for high-value, high-risk accounts.</p>
    <p><b style="color:#f3f0fa;">3️⃣ Encourage long-term contracts</b> — move short-term customers to 1-2 year plans.</p>
    <p><b style="color:#f3f0fa;">4️⃣ Personalize retention offers</b> — use churn probability to target offers.</p>
    <p><b style="color:#f3f0fa;">5️⃣ Protect recurring revenue</b> — prioritize high-value customers.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CUSTOMER PREDICTION HISTORY
# ============================================================

st.divider()
st.header("📝 Customer Prediction History")
st.caption("Save customer predictions, review previous predictions, and download the history as CSV.")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("💾 Save Current Customer Prediction", key="save_prediction"):

    current_prediction = model.predict(customer_data)[0]
    current_probability = model.predict_proba(customer_data)[0][1]
    current_probability_percentage = current_probability * 100

    if current_prediction == 1:
        prediction_result = "Churn"
        risk_level = "High Risk"
    else:
        prediction_result = "Stay"
        risk_level = "Low Risk"

    prediction_record = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "Contract": contract,
        "Internet Service": internet_service,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "Churn Probability (%)": round(current_probability_percentage, 2),
        "Prediction": prediction_result,
        "Risk Level": risk_level
    }

    st.session_state.prediction_history.append(prediction_record)
    st.success("✅ Customer prediction saved successfully!")

if len(st.session_state.prediction_history) > 0:

    st.subheader("📋 Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, height=220)

    st.subheader("📊 Prediction History Summary")

    total_predictions = len(history_df)
    total_predicted_churn = (history_df["Prediction"] == "Churn").sum()
    total_predicted_stay = (history_df["Prediction"] == "Stay").sum()
    average_prediction_probability = history_df["Churn Probability (%)"].mean()

    history_col1, history_col2, history_col3, history_col4 = st.columns(4)
    with history_col1:
        st.metric("Total Predictions", total_predictions)
    with history_col2:
        st.metric("Predicted Churn", int(total_predicted_churn))
    with history_col3:
        st.metric("Predicted Stay", int(total_predicted_stay))
    with history_col4:
        st.metric("Avg Churn Probability", f"{average_prediction_probability:.2f}%")

    hist_chart_col, hist_actions_col = st.columns([1, 1])

    with hist_chart_col:
        st.subheader("📈 Prediction Distribution")
        history_chart = pd.DataFrame({
            "Prediction": ["Predicted Churn", "Predicted Stay"],
            "Customers": [int(total_predicted_churn), int(total_predicted_stay)]
        })
        fig_history, ax_history = compact_fig((4.4, 2.8))
        ax_history.bar(history_chart["Prediction"], history_chart["Customers"], color=[RED, GREEN])
        ax_history.set_ylabel("Customers")
        ax_history.grid(axis="y", linestyle="--", alpha=0.4)
        for i, v in enumerate(history_chart["Customers"]):
            ax_history.text(i, v + 0.1, str(v), ha="center", fontsize=8, color=TEXT_MAIN)
        show_compact(fig_history)

    with hist_actions_col:
        st.subheader("📥 Manage History")
        history_csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction History as CSV",
            data=history_csv,
            file_name="customer_prediction_history.csv",
            mime="text/csv",
            key="download_history"
        )
        if st.button("🗑️ Clear Prediction History", key="clear_history"):
            st.session_state.prediction_history = []
            st.success("✅ Prediction history cleared.")
            st.rerun()

else:
    st.info("ℹ️ No prediction history available yet.")
    st.caption("Enter customer information in the sidebar, click **Predict Churn**, then click **Save Current Customer Prediction**.")

# ============================================================
# PROJECT COMPLETION
# ============================================================

st.divider()
st.success("✅ Customer Churn Prediction Dashboard Completed")
st.caption(
    "Machine Learning • Data Analysis • Interactive Dashboard "
    "• Risk Segmentation • Customer Retention • "
    "Customer Value Analysis • Prediction History"
)

import streamlit as st
import pandas as pd
import plotly.io as pio
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mkopo Insights | BNPL Risk Management",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. PROJECT PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" /
    "recommended_lipa_mdogo_mdogo_model.joblib"
)

DATA_PATH = (
    BASE_DIR / "data" /
    "BNPL_Financial_Default_Risk_Dataset.csv"
)

USD_TO_KES = 129.89

FEATURES = [
    "Age",
    "Monthly_Income_KES",
    "Credit_Score",
    "Total_BNPL_Active_Loans",
    "Total_BNPL_Debt_KES",
    "Average_Transaction_Value_KES",
    "Employment_Status",
    "Late_Payment_History",
    "Shopping_Category_Most_Frequent"
]

RISK_COLORS = {
    "Low": "#16A085",
    "Medium": "#F39C12",
    "High": "#E74C3C"
}

# ==================================================
# 2B. PREMIUM DARK PLOTLY CHART THEME
# ==================================================

# ==================================================
# 2B. PREMIUM LIGHT PLOTLY CHART THEME
# ==================================================

pio.templates["mkopo_light"] = pio.templates["plotly_white"]

pio.templates["mkopo_light"].layout.update(
    {
        "paper_bgcolor": "#F4F6FA",
        "plot_bgcolor": "#FFFFFF",

        "font": {
            "family": "Arial",
            "size": 13,
            "color": "#263445"
        },

        "title": {
            "font": {
                "size": 19,
                "color": "#263445"
            },
            "x": 0.02
        },

        "xaxis": {
            "gridcolor": "#E5E9F0",
            "linecolor": "#CBD5E1",
            "zerolinecolor": "#CBD5E1"
        },

        "yaxis": {
            "gridcolor": "#E5E9F0",
            "linecolor": "#CBD5E1",
            "zerolinecolor": "#CBD5E1"
        },

        "legend": {
            "font": {"color": "#263445"},
            "bgcolor": "#FFFFFF"
        },

        "colorway": [
            "#D6AD60",
            "#159D82",
            "#4776A8",
            "#E57373",
            "#8B78C6"
        ]
    }
)

px.defaults.template = "mkopo_light"

# ==================================================
# 3. DASHBOARD STYLING - PREMIUM DARK FINANCE
# ==================================================

# Theme colours
DARK_BG = "#202B3C"
SIDEBAR_BG = "#101827"
CARD_BG = "#344258"
GOLD = "#D6AD60"
MAIN_TEXT = "#F5F7FA"
SECONDARY_TEXT = "#B8C2D1"
SUCCESS = "#35B58A"

st.markdown("""

<style>
/* =========================================
   FINAL VISUAL POLISH - MKOPO INSIGHTS
========================================= */

/* 1. SIDEBAR BRAND NAME */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong {
    color: #F5F7FA !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #D6AD60 !important;
}

/* 2. CUSTOMER INFORMATION EXPANDER */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6AD60 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary p {
    color: #17243A !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] summary svg {
    color: #B58A42 !important;
    fill: none !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #263445;
}

/* 3. PREMIUM PORTFOLIO FILTERS */

[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #243B5A !important;
    border: 1px solid #D6AD60 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    min-height: 54px;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #415979 !important;
    border-radius: 6px !important;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

[data-testid="stMultiSelect"] svg {
    color: #D6AD60 !important;
}

/* 4. MAKE SELECTBOX TEXT READABLE */

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #243B5A !important;
    border: 1px solid #D6AD60 !important;
    border-radius: 10px !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* 5. IMPROVE FILTER LABELS */

[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label {
    color: #17243A !important;
    font-weight: 600 !important;
}

/* 6. KEEP SIDEBAR LABELS WHITE */

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #F5F7FA !important;
}

/* 7. ENSURE EXPANDER CONTENT IS LEGIBLE */

[data-testid="stExpander"] [data-testid="stWidgetLabel"] p {
    color: #17243A !important;
}

[data-testid="stExpander"] [data-testid="stMetricValue"] {
    color: #17243A !important;
}
/* Reduce top spacing in sidebar */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
}

/* Compact brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 0 5px 0;
    color: #D6AD60;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0;
    white-space: nowrap;
}

/* Team information */
.project-team {
    color: #F4F6FA;
    font-size: 14px;
    line-height: 1.35;
}

.project-team div {
    margin-bottom: 3px;
}

/* =========================================
   MAIN APPLICATION BACKGROUND
========================================= */

.stApp {
    background-color: #F4F6FA;
    color: #263445;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F6FA;
}

[data-testid="stHeader"] {
    background-color: #F4F6FA;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* =========================================
   SIDEBAR
========================================= */

[data-testid="stSidebar"] {
    background-color: #101827;
    border-right: 1px solid #344258;
}

[data-testid="stSidebar"] * {
    color: #F5F7FA;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B8C2D1;
}

/* =========================================
   HEADINGS AND TEXT
========================================= */

h1, h2, h3, h4, h5, h6 {
    color: #17243A !important;
}

p, label, li {
    color: #344258;
}

small {
    color: #B8C2D1;
}

/* =========================================
   KPI METRIC CARDS
========================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-color: #E2E8F0;
    border-radius: 14px;
}

div[data-testid="stMetricDelta"] {
    color: #35B58A !important;
}

/* =========================================
   GENERAL CARDS AND CONTAINERS
========================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #344258;
    border-color: #526176;
    border-radius: 14px;
}

/* =========================================
   INPUTS AND FILTERS
========================================= */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea {
    background-color: #344258 !important;
    color: #F5F7FA !important;
    border: 1px solid #64748B !important;
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #344258;
    color: #F5F7FA;
    border-color: #64748B;
}

[data-baseweb="popover"] {
    background-color: #344258;
}

/* =========================================
   BUTTONS
========================================= */

.stButton button,
.stDownloadButton button {
    background-color: #D6AD60;
    color: #101827 !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #E7C47F;
    color: #101827 !important;
    border: 1px solid #D6AD60;
}

/* =========================================
   ALERTS AND NOTIFICATIONS
========================================= */

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #526176;
}

/* =========================================
   DATA TABLES
========================================= */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #344258;
    border-radius: 10px;
}

/* =========================================
   EXPANDERS
========================================= */

[data-testid="stExpander"] {
    background-color: #344258;
    border: 1px solid #526176;
    border-radius: 10px;
}

/* =========================================
   DIVIDERS
========================================= */

hr {
    border-color: #526176;
}

/* =========================================
   DASHBOARD INFORMATION NOTE
========================================= */

.dashboard-note {
    background-color: #344258;
    padding: 16px;
    border-radius: 12px;
    color: #F5F7FA;
    border-left: 4px solid #D6AD60;
}

/* =========================================
   RADIO BUTTONS AND CHECKBOXES
========================================= */

.stRadio label,
.stCheckbox label {
    color: #F5F7FA !important;
}

/* =========================================
   CAPTIONS
========================================= */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #B8C2D1 !important;
}

/* =========================================
   LINKS
========================================= */

a {
    color: #D6AD60 !important;
}

/* =========================================
   FIX 1: DROPDOWN MENUS AND OPTIONS
========================================= */

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #344258 !important;
    border: 1px solid #D6AD60 !important;
}

li[role="option"],
div[role="option"] {
    background-color: #344258 !important;
    color: #F5F7FA !important;
}

li[role="option"]:hover,
div[role="option"]:hover {
    background-color: #526176 !important;
    color: #FFFFFF !important;
}

li[aria-selected="true"],
div[aria-selected="true"] {
    background-color: #526176 !important;
    color: #D6AD60 !important;
}

/* Selected values inside dropdowns */

[data-baseweb="tag"] {
    background-color: #526176 !important;
    color: #FFFFFF !important;
}

[data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

/* =========================================
   FIX 2: DROPDOWN TEXT AND ICONS
========================================= */

div[data-baseweb="select"] * {
    color: #F5F7FA !important;
}

div[data-baseweb="select"] svg {
    fill: #D6AD60 !important;
    color: #D6AD60 !important;
}

button svg {
    color: #D6AD60 !important;
    fill: #D6AD60 !important;
}

/* =========================================
   FIX 3: SIDEBAR ICONS AND NAVIGATION
========================================= */

[data-testid="stSidebar"] svg {
    color: #D6AD60 !important;
    fill: none;
}

[data-testid="stSidebar"] label {
    color: #F5F7FA !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #F5F7FA !important;
}

/* =========================================
   FIX 4: VISIBLE SCROLLBAR
========================================= */

html {
    scrollbar-color: #D6AD60 #101827;
    scrollbar-width: auto;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    scrollbar-color: #D6AD60 #101827;
}

::-webkit-scrollbar {
    width: 13px;
    height: 13px;
}

::-webkit-scrollbar-track {
    background: #101827;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #D6AD60;
    border-radius: 10px;
    border: 3px solid #101827;
}

::-webkit-scrollbar-thumb:hover {
    background: #E7C47F;
}

/* =========================================
   FIX 5: IMPROVE INPUT LABEL VISIBILITY
========================================= */

.stSelectbox label,
.stMultiSelect label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
.stRadio label,
.stCheckbox label {
    color: #F5F7FA !important;
    font-weight: 500;
}

/* Placeholder text */

input::placeholder {
    color: #B8C2D1 !important;
}

/* =========================================
   MKOPO INSIGHTS BRANDING
========================================= */

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 0 18px 0;
    color: #D6AD60;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.sidebar-brand .brand-icon {
    font-size: 27px;
    line-height: 1;
}
/* FINAL SIDEBAR ADJUSTMENTS */

/* Reduce top space and brand spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.15rem !important;
}

/* Compact gold brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 0 5px 0 !important;
    margin: 0 !important;
    color: #D6AD60 !important;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 0.2px;
    white-space: nowrap;
}

.sidebar-brand span {
    color: #D6AD60 !important;
}

/* Reduce divider spacing */
section[data-testid="stSidebar"] hr {
    margin-top: 0.35rem !important;
    margin-bottom: 0.35rem !important;
}

/* Indent team names */
.project-team {
    padding-left: 12px !important;
    line-height: 1.35 !important;
}

.project-team div {
    margin-bottom: 3px !important;
}
/* Reduce top spacing in sidebar */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)
# ==================================================
# 4. CURRENCY FORMATTER
# ==================================================

def format_kes(amount):
    if pd.isna(amount):
        return "N/A"

    return f"KSh {amount:,.2f}"

# ==================================================
# CUSTOMER RISK MANAGEMENT RECOMMENDATIONS
# ==================================================

RISK_LABELS = {
    0: "High",
    1: "Low",
    2: "Medium"
}


def get_management_recommendation(risk_level):

    if risk_level == "High":

        return {
            "action": "Enhanced Due Diligence",
            "priority": "High",
            "recommendation": [
                "Review the customer's repayment history.",
                "Verify income and employment information.",
                "Review existing BNPL obligations and outstanding debt.",
                "Consider reducing the credit limit or temporarily "
                "restricting additional exposure.",
                "Escalate the case for manual credit review before "
                "approving further credit."
            ]
        }

    elif risk_level == "Medium":

        return {
            "action": "Controlled Credit Monitoring",
            "priority": "Moderate",
            "recommendation": [
                "Maintain the current credit limit pending review.",
                "Consider a cautious limit reduction if affordability "
                "or repayment concerns are identified.",
                "Monitor repayment behaviour more frequently.",
                "Verify income and existing credit obligations.",
                "Avoid automatic credit-limit increases."
            ]
        }

    else:

        return {
            "action": "Standard Credit Monitoring",
            "priority": "Routine",
            "recommendation": [
                "Consider maintaining the existing credit limit.",
                "Continue routine repayment monitoring.",
                "Review affordability before approving additional credit.",
                "Consider a gradual limit increase only where "
                "repayment capacity supports it."
            ]
        }

# ==================================================
# 5. LOAD MACHINE-LEARNING MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ==================================================
# 6. LOAD AND PREPARE CUSTOMER DATA
# ==================================================

@st.cache_data
def load_customer_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Convert financial variables to Kenyan shillings
    df["Monthly_Income_KES"] = (
        df["Income_USD"] * USD_TO_KES
    )

    df["Total_BNPL_Debt_KES"] = (
        df["Total_BNPL_Debt_USD"] * USD_TO_KES
    )

    df["Average_Transaction_Value_KES"] = (
        df["Average_Transaction_Value_USD"]
        * USD_TO_KES
    )

    # Standardise risk labels for dashboard display
    df["Default_Risk"] = (
        df["Default_Risk"].astype(str).str.strip().str.title()
    )

    return df


# ==================================================
# 7. LOAD APPLICATION DATA
# ==================================================

try:

    model = load_model()
    df = load_customer_data()

except Exception as error:

    st.error(f"Application loading error: {error}")
    st.stop()


# ==================================================
# 8. SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-icon">💳</span>
            <span>MKOPO INSIGHTS</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("BNPL Credit Risk Intelligence")

    st.divider()

    # MAIN MENU
    page = st.radio(
        "MAIN MENU",
        [
            "Executive Overview",
            "Customer Risk Assessment",
            "Credit Management",
            "Portfolio Analytics",
            "Model Performance"
        ]
    )

    st.divider()

    # PROJECT INFORMATION
    st.markdown("### PROJECT INFORMATION")

    st.markdown(
        """
        <div class="project-team">
            <div><strong>Prepared by:</strong></div>
            <div>Francis Kamau</div>
            <div>Kelvin Namutenda Omuncheyi</div>
            <div>Benedict Onyango Aloo</div>
            <div>George Okumu</div>
            <div>Faith Ngina Muiruri</div>
            <div>Barbara Kinoti</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Programme: Moringa School — Data Science")
    st.caption("Project: Data Science Capstone Project")
    st.caption("Technology: Python | Machine Learning | Streamlit | Plotly")

    st.caption("KENYA | KSH CURRENCY")
    st.caption("Management Decision Support")

# ==================================================
# 9. DASHBOARD HEADER
# ==================================================

st.markdown(
    "<h1 style='color:#17243A;'>Credit Risk Intelligence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "### Buy Now, Pay Later | Management Decision Support"
)

st.markdown(
    '<div class="dashboard-note">'
    'A data-driven decision-support platform for responsible '
    'BNPL lending and portfolio risk management.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# 10. EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.subheader("Executive Portfolio Overview")

    st.caption(
        "Portfolio information is calculated from the supplied "
        "10,000-record demonstration dataset."
    )

    total_customers = len(df)

    total_debt = df["Total_BNPL_Debt_KES"].sum()

    average_debt = df["Total_BNPL_Debt_KES"].mean()

    average_income = df["Monthly_Income_KES"].mean()

    risk_counts = (
        df["Default_Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    high_risk = int(risk_counts["High"])
    medium_risk = int(risk_counts["Medium"])
    low_risk = int(risk_counts["Low"])

    high_risk_share = (
        high_risk / total_customers * 100
        if total_customers else 0
    )

    # Portfolio KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total BNPL Debt",
        format_kes(total_debt)
    )

    col3.metric(
        "Average BNPL Debt",
        format_kes(average_debt)
    )

    col4.metric(
        "High-Risk Share",
        f"{high_risk_share:.2f}%"
    )

    st.write("")

    # Risk category summary
    st.markdown("### Customer Risk Distribution")

    c1, c2, c3 = st.columns(3)

    c1.metric("Low Risk", f"{low_risk:,}")
    c2.metric("Medium Risk", f"{medium_risk:,}")
    c3.metric("High Risk", f"{high_risk:,}")

    # Risk distribution chart
    left, right = st.columns(2)

    with left:

        risk_chart = pd.DataFrame({
            "Risk Category": risk_counts.index,
            "Customers": risk_counts.values
        })

        fig_risk = px.pie(
            risk_chart,
            names="Risk Category",
            values="Customers",
            hole=0.58,
            color="Risk Category",
            color_discrete_map=RISK_COLORS,
            title="Portfolio Risk Composition"
        )

        fig_risk.update_layout(
            legend_title="Risk Category",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    with right:

        employment_risk = pd.crosstab(
            df["Employment_Status"],
            df["Default_Risk"]
        ).reindex(
            columns=["Low", "Medium", "High"],
            fill_value=0
        ).reset_index()

        employment_long = employment_risk.melt(
            id_vars="Employment_Status",
            var_name="Risk Category",
            value_name="Customers"
        )

        fig_employment = px.bar(
            employment_long,
            x="Employment_Status",
            y="Customers",
            color="Risk Category",
            barmode="stack",
            color_discrete_map=RISK_COLORS,
            title="Risk Distribution by Employment"
        )

        fig_employment.update_layout(
            xaxis_title="Employment Status",
            yaxis_title="Number of Customers",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(
            fig_employment,
            use_container_width=True
        )

    # Shopping category analysis
    st.markdown("### Risk by Shopping Category")

    shopping_risk = pd.crosstab(
        df["Shopping_Category_Most_Frequent"],
        df["Default_Risk"]
    ).reindex(
        columns=["Low", "Medium", "High"],
        fill_value=0
    ).reset_index()

    shopping_long = shopping_risk.melt(
        id_vars="Shopping_Category_Most_Frequent",
        var_name="Risk Category",
        value_name="Customers"
    )

    fig_shopping = px.bar(
        shopping_long,
        x="Shopping_Category_Most_Frequent",
        y="Customers",
        color="Risk Category",
        barmode="group",
        color_discrete_map=RISK_COLORS,
        title="Customer Risk across Shopping Categories"
    )

    fig_shopping.update_layout(
        xaxis_title="Shopping Category",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(
        fig_shopping,
        use_container_width=True
    )

    # Portfolio data quality
    st.markdown("### Data Quality Snapshot")

    missing_income = int(df["Monthly_Income_KES"].isna().sum())

    missing_credit_score = int(
        df["Credit_Score"].isna().sum()
    )

    q1, q2, q3 = st.columns(3)

    q1.metric("Missing Income Records", f"{missing_income:,}")

    q2.metric(
        "Missing Credit Scores",
        f"{missing_credit_score:,}"
    )

    q3.metric(
        "Average Monthly Income",
        format_kes(average_income)
    )

    st.warning(
        "Dataset limitation: the source records are USD-denominated "
        "and converted using KSh 129.89 per USD. This is a "
        "demonstration dataset, not verified Kenyan lender data. "
        "High-risk share is not the same as an actual default rate."
    )


# ==================================================
# 11. CUSTOMER RISK ASSESSMENT
# ==================================================

elif page == "Customer Risk Assessment":

    st.subheader("Customer Risk Assessment")

    st.caption(
        "Assess an existing customer from the demonstration dataset "
        "or enter details for a new customer."
    )

    # Choose customer source
    assessment_source = st.radio(
        "Select assessment method",
        [
            "Use an existing test customer",
            "Enter a new customer"
        ],
        horizontal=True
    )

    customer_data = None
    customer_reference = "New customer"

    # --------------------------------------------------
    # EXISTING TEST CUSTOMER
    # --------------------------------------------------

    if assessment_source == "Use an existing test customer":

        customer_ids = df["Customer_ID"].astype(str).tolist()

        selected_id = st.selectbox(
            "Search or select Customer ID",
            customer_ids
        )

        selected_customer = df[
            df["Customer_ID"].astype(str) == selected_id
        ].iloc[0]

        customer_reference = selected_id

        customer_data = {
            feature: selected_customer[feature]
            for feature in FEATURES
        }

        st.info(
            "The selected customer's original dataset details "
            "will be used for the assessment."
        )

        with st.expander("View selected customer information"):

            display_data = pd.DataFrame({
                "Customer Feature": FEATURES,
                "Customer Value": [
                    customer_data[feature]
                    for feature in FEATURES
                ]
            })

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------
    # NEW CUSTOMER INPUT FORM
    # --------------------------------------------------

    else:

        st.markdown("### Enter Customer Information")

        employment_options = sorted(
            df["Employment_Status"].dropna().unique().tolist()
        )

        late_payment_options = sorted(
            df["Late_Payment_History"].dropna().unique().tolist()
        )

        shopping_options = sorted(
            df["Shopping_Category_Most_Frequent"]
            .dropna()
            .unique()
            .tolist()
        )

        with st.form("customer_assessment_form"):

            col1, col2 = st.columns(2)

            with col1:

                age = st.number_input(
                    "Customer Age (Years)",
                    min_value=18,
                    max_value=100,
                    value=30,
                    step=1
                )

                monthly_income = st.number_input(
                    "Monthly Income (KSh)",
                    min_value=0.0,
                    value=50000.0,
                    step=5000.0,
                    format="%.2f"
                )

                credit_score = st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=650,
                    step=1
                )

                active_loans = st.number_input(
                    "Active BNPL Loans",
                    min_value=0,
                    max_value=10,
                    value=1,
                    step=1
                )

            with col2:

                bnpl_debt = st.number_input(
                    "Total Outstanding BNPL Debt (KSh)",
                    min_value=0.0,
                    value=10000.0,
                    step=1000.0,
                    format="%.2f"
                )

                average_transaction = st.number_input(
                    "Average Transaction Value (KSh)",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                    format="%.2f"
                )

                employment = st.selectbox(
                    "Employment Status",
                    employment_options
                )

                late_payment = st.selectbox(
                    "Late Payment History",
                    late_payment_options
                )

            shopping_category = st.selectbox(
                "Most Frequent Shopping Category",
                shopping_options
            )

            submitted = st.form_submit_button(
                "Assess Customer Risk",
                type="primary",
                use_container_width=True
            )

        if submitted:

            customer_data = {
                "Age": age,
                "Monthly_Income_KES": monthly_income,
                "Credit_Score": credit_score,
                "Total_BNPL_Active_Loans": active_loans,
                "Total_BNPL_Debt_KES": bnpl_debt,
                "Average_Transaction_Value_KES": average_transaction,
                "Employment_Status": employment,
                "Late_Payment_History": late_payment,
                "Shopping_Category_Most_Frequent": shopping_category
            }

            st.session_state["assessed_customer"] = customer_data

            st.session_state["customer_reference"] = "New customer"

    # --------------------------------------------------
    # RUN MODEL PREDICTION
    # --------------------------------------------------

    if assessment_source == "Enter a new customer":

        if "assessed_customer" in st.session_state:

            customer_data = st.session_state["assessed_customer"]
            customer_reference = st.session_state["customer_reference"]

    if customer_data is not None:

        if st.button(
            "Run Risk Assessment",
            type="primary",
            key="run_customer_assessment"
        ):

            try:

                input_df = pd.DataFrame(
                    [customer_data],
                    columns=FEATURES
                )

                predicted_code = int(model.predict(input_df)[0])

                risk_level = RISK_LABELS[predicted_code]

                probabilities = model.predict_proba(input_df)[0]

                probability_results = {
                    RISK_LABELS[int(class_code)]: float(probability)
                    for class_code, probability
                    in zip(model.classes_, probabilities)
                }

                st.session_state["latest_assessment"] = {
                    "reference": customer_reference,
                    "risk_level": risk_level,
                    "probabilities": probability_results,
                    "customer_data": customer_data
                }

            except Exception as error:

                st.error(
                    f"Unable to assess customer: {error}"
                )

        # --------------------------------------------------
        # DISPLAY ASSESSMENT RESULTS
        # --------------------------------------------------

        if "latest_assessment" in st.session_state:

            result = st.session_state["latest_assessment"]

            if result["reference"] == customer_reference:

                risk_level = result["risk_level"]

                probabilities = result["probabilities"]

                st.divider()

                st.markdown("## Customer Assessment Results")

                st.write(
                    f"**Customer Reference:** {result['reference']}"
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "Predicted Risk Category",
                        risk_level
                    )

                with result_col2:

                    st.metric(
                        "Estimated High-Risk Class Probability",
                        f"{probabilities['High'] * 100:.2f}%"
                    )

                if risk_level == "High":

                    st.error(
                        "HIGH RISK — Further credit exposure requires "
                        "careful review."
                    )

                elif risk_level == "Medium":

                    st.warning(
                        "MEDIUM RISK — Controlled credit monitoring "
                        "is recommended."
                    )

                else:

                    st.success(
                        "LOW RISK — Standard credit monitoring "
                        "is recommended."
                    )

                # Probability chart
                st.markdown("### Model-Estimated Risk Probabilities")

                probability_df = pd.DataFrame({
                    "Risk Category": ["Low", "Medium", "High"],
                    "Estimated Probability": [
                        probabilities["Low"] * 100,
                        probabilities["Medium"] * 100,
                        probabilities["High"] * 100
                    ]
                })

                fig_probability = px.bar(
                    probability_df,
                    x="Risk Category",
                    y="Estimated Probability",
                    color="Risk Category",
                    color_discrete_map=RISK_COLORS,
                    text="Estimated Probability",
                    range_y=[0, 100],
                    title="Estimated Probability of Each Risk Class"
                )

                fig_probability.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside"
                )

                fig_probability.update_layout(
                    yaxis_title="Model-estimated probability (%)",
                    xaxis_title="Risk Category",
                    showlegend=False
                )

                st.plotly_chart(
                    fig_probability,
                    use_container_width=True
                )

                # Management recommendations
                recommendation = get_management_recommendation(
                    risk_level
                )

                st.markdown("### Management Recommendation")

                st.markdown(
                    f"**Recommended Action:** "
                    f"{recommendation['action']}"
                )

                st.markdown(
                    f"**Priority:** {recommendation['priority']}"
                )

                for action in recommendation["recommendation"]:

                    st.markdown(f"- {action}")

                st.info(
                    "Important: Model-estimated class probabilities "
                    "are not verified default rates. This assessment "
                    "is decision support and must be reviewed by an "
                    "authorised credit officer."
                )

# ==================================================
# 12. CREDIT MANAGEMENT
# ==================================================


elif page == "Credit Management":

    st.subheader("Credit Management Centre")

    st.caption(
        "Portfolio monitoring, credit exposure controls and "
        "management action recommendations."
    )

    # --------------------------------------------------
    # 1. PORTFOLIO RISK POSITION
    # --------------------------------------------------

    risk_counts = (
        df["Default_Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    total_customers = len(df)

    high_risk_count = int(risk_counts["High"])
    medium_risk_count = int(risk_counts["Medium"])
    low_risk_count = int(risk_counts["Low"])

    high_risk_share = (
        high_risk_count / total_customers * 100
        if total_customers else 0
    )

    high_risk_debt = df.loc[
        df["Default_Risk"] == "High",
        "Total_BNPL_Debt_KES"
    ].sum()

    total_portfolio_debt = df["Total_BNPL_Debt_KES"].sum()

    high_risk_debt_share = (
        high_risk_debt / total_portfolio_debt * 100
        if total_portfolio_debt else 0
    )

    st.markdown("### Portfolio Risk Position")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("High-Risk Customers", f"{high_risk_count:,}")

    c2.metric("High-Risk Share", f"{high_risk_share:.2f}%")

    c3.metric("High-Risk Debt", format_kes(high_risk_debt))

    c4.metric("High-Risk Debt Exposure", f"{high_risk_debt_share:.2f}%")

    st.divider()

    # --------------------------------------------------
    # 2. MANAGEMENT POLICY SETTINGS
    # --------------------------------------------------

    st.markdown("### Configure Management Risk Thresholds")

    st.caption(
        "These are configurable demonstration policy settings. "
        "Management should approve thresholds before operational use."
    )

    col1, col2 = st.columns(2)

    with col1:

        high_risk_trigger = st.slider(
            "High-risk customer concentration alert (%)",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

        high_limit_reduction = st.slider(
            "Suggested High-risk credit-limit reduction (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5
        )

    with col2:

        medium_risk_trigger = st.slider(
            "Medium-risk customer concentration alert (%)",
            min_value=1,
            max_value=50,
            value=15,
            step=1
        )

        medium_limit_reduction = st.slider(
            "Suggested Medium-risk credit-limit reduction (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=5
        )

    st.divider()

    # --------------------------------------------------
    # 3. PORTFOLIO ALERTS
    # --------------------------------------------------

    st.markdown("### Management Alerts")

    if high_risk_share >= high_risk_trigger:

        st.error(
            f"HIGH PRIORITY: High-risk customer concentration is "
            f"{high_risk_share:.2f}%, exceeding the configured "
            f"alert threshold of {high_risk_trigger}%."
        )

        st.markdown("**Recommended management actions:**")

        st.markdown(
            "- Review the high-risk customer portfolio.\n"
            "- Tighten credit-limit increase approvals.\n"
            "- Prioritise repayment monitoring and collection follow-up.\n"
            "- Review affordability and outstanding customer obligations.\n"
            "- Investigate whether risk is concentrated in particular "
            "employment or shopping segments."
        )

    else:

        st.success(
            f"High-risk customer concentration is "
            f"{high_risk_share:.2f}%, within the configured "
            f"{high_risk_trigger}% alert threshold."
        )

    if (medium_risk_count / total_customers * 100) >= medium_risk_trigger:

        st.warning(
            "Medium-risk customer concentration has reached or "
            "exceeded the configured management threshold. "
            "Consider closer repayment monitoring."
        )

    # --------------------------------------------------
    # 4. VERIFIED DEFAULT RATE MONITORING
    # --------------------------------------------------

    st.divider()

    st.markdown("### Actual Default Rate Monitoring")

    st.info(
        "The supplied demonstration dataset contains risk labels, "
        "not verified repayment outcomes. Enter an observed default "
        "rate only when it comes from actual portfolio repayment records."
    )

    has_verified_default_rate = st.checkbox(
        "I have a verified observed portfolio default rate"
    )

    if has_verified_default_rate:

        observed_default_rate = st.number_input(
            "Observed portfolio default rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

        default_rate_trigger = st.number_input(
            "Management escalation threshold (%)",
            min_value=1.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        if observed_default_rate >= default_rate_trigger:

            st.error(
                f"CRITICAL PORTFOLIO ALERT: Observed default rate "
                f"is {observed_default_rate:.1f}%, meeting or "
                f"exceeding the {default_rate_trigger:.1f}% "
                f"escalation threshold."
            )

            st.markdown("**Suggested urgent management response:**")

            st.markdown(
                "- Escalate the matter to senior credit management.\n"
                "- Pause automatic credit-limit increases in the "
                "affected portfolio pending investigation.\n"
                "- Conduct enhanced due diligence on affected "
                "customer segments.\n"
                "- Review collection effectiveness and repayment "
                "arrangements.\n"
                "- Investigate underwriting, fraud, affordability "
                "and economic factors.\n"
                "- Develop a corrective action plan with clear "
                "owners and review dates."
            )

        else:

            st.success(
                f"Observed default rate is {observed_default_rate:.1f}%, "
                f"below the configured escalation threshold."
            )

    else:

        st.caption(
            "No verified default rate has been supplied. "
            "The dashboard will not treat model risk classifications "
            "as actual defaults."
        )

    # --------------------------------------------------
    # 5. CREDIT LIMIT MANAGEMENT SIMULATOR
    # --------------------------------------------------

    st.divider()

    st.markdown("### Credit Limit Management Simulator")

    st.caption(
        "Illustrative policy calculator. The suggested limit is "
        "not an affordability assessment or an automatic credit decision."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_risk = st.selectbox(
            "Customer risk category",
            ["Low", "Medium", "High"]
        )

    with col2:

        current_limit = st.number_input(
            "Current customer credit limit (KSh)",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

    if selected_risk == "High":

        reduction_rate = high_limit_reduction

        proposed_action = (
            "Consider reducing the limit and conducting "
            "enhanced due diligence."
        )

    elif selected_risk == "Medium":

        reduction_rate = medium_limit_reduction

        proposed_action = (
            "Apply controlled monitoring and review affordability "
            "before increasing exposure."
        )

    else:

        reduction_rate = 0

        proposed_action = (
            "Maintain the current limit subject to affordability "
            "and repayment checks."
        )

    proposed_limit = current_limit * (1 - reduction_rate / 100)

    if st.button(
        "Calculate Suggested Credit Limit",
        type="primary"
    ):

        st.markdown("#### Suggested Management Position")

        a, b, c = st.columns(3)

        a.metric(
            "Current Credit Limit",
            format_kes(current_limit)
        )

        b.metric(
            "Illustrative Reduction",
            f"{reduction_rate}%"
        )

        c.metric(
            "Proposed Credit Limit",
            format_kes(proposed_limit)
        )

        st.info(proposed_action)

        st.caption(
            "Final credit limits should be based on verified income, "
            "affordability, repayment capacity, existing obligations "
            "and approved lending policy."
        )

    # --------------------------------------------------
    # 6. CUSTOMER SEGMENT MANAGEMENT
    # --------------------------------------------------

    st.divider()

    st.markdown("### Customer Segment Management")

    segment = st.selectbox(
        "Select a customer segment",
        [
            "All customers",
            "High risk",
            "Medium risk",
            "Low risk"
        ]
    )

    if segment == "All customers":

        segment_df = df.copy()

    else:

        segment_df = df[
            df["Default_Risk"] == segment.replace(" risk", "").title()
        ].copy()

    st.metric(
        "Customers in Selected Segment",
        f"{len(segment_df):,}"
    )

    segment_display = segment_df[
        [
            "Customer_ID",
            "Employment_Status",
            "Default_Risk",
            "Monthly_Income_KES",
            "Total_BNPL_Debt_KES",
            "Total_BNPL_Active_Loans"
        ]
    ].copy()

    segment_display["Monthly_Income_KES"] = (
        segment_display["Monthly_Income_KES"].map(format_kes)
    )

    segment_display["Total_BNPL_Debt_KES"] = (
        segment_display["Total_BNPL_Debt_KES"].map(format_kes)
    )

    st.dataframe(
        segment_display.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing up to 100 customers. This is demonstration data; "
        "customer-level actions require verification and authorised review."
    )


# ==================================================
# 13. PORTFOLIO ANALYTICS
# ==================================================


elif page == "Portfolio Analytics":
    st.subheader("Portfolio Analytics")
    st.caption(
        "Interactive analysis of customer risk, BNPL exposure, "
        "repayment-history indicators, and customer segments."
    )

    # -----------------------------------------
    # 1. Portfolio filters
    # -----------------------------------------
    st.markdown("### Portfolio Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_risks = st.multiselect(
            "Risk categories",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="analytics_risk_filter"
        )

    with filter_col2:
        selected_employment = st.multiselect(
            "Employment status",
            options=sorted(df["Employment_Status"].dropna().unique().tolist()),
            default=sorted(df["Employment_Status"].dropna().unique().tolist()),
            key="analytics_employment_filter"
        )

    with filter_col3:
        selected_late_payments = st.multiselect(
            "Late-payment history",
            options=sorted(
                df["Late_Payment_History"].dropna().unique().tolist()
            ),
            default=sorted(
                df["Late_Payment_History"].dropna().unique().tolist()
            ),
            key="analytics_late_payment_filter"
        )

    filtered_df = df[
        df["Default_Risk"].isin(selected_risks)
        & df["Employment_Status"].isin(selected_employment)
        & df["Late_Payment_History"].isin(selected_late_payments)
    ].copy()

    if filtered_df.empty:
        st.warning(
            "No customers match these filters. "
            "Please change your filter selections."
        )
        st.stop()

    # -----------------------------------------
    # 2. Portfolio summary KPIs
    # -----------------------------------------
    st.markdown("### Portfolio Performance Snapshot")

    customer_count = len(filtered_df)
    total_debt = filtered_df["Total_BNPL_Debt_KES"].sum()
    average_debt = filtered_df["Total_BNPL_Debt_KES"].mean()

    high_risk_count = (
        filtered_df["Default_Risk"] == "High"
    ).sum()

    high_risk_percentage = (
        high_risk_count / customer_count * 100
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Customers", f"{customer_count:,}")
    kpi2.metric("Total BNPL Debt", format_kes(total_debt))
    kpi3.metric("Average Debt", format_kes(average_debt))
    kpi4.metric(
        "High-Risk Share",
        f"{high_risk_percentage:.2f}%"
    )

    st.divider()

    # -----------------------------------------
    # 3. Risk distribution and debt exposure
    # -----------------------------------------
    st.markdown("### Risk Distribution and Debt Exposure")

    chart_col1, chart_col2 = st.columns(2)

    risk_counts = (
        filtered_df["Default_Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk", "Customers"]

    with chart_col1:
        fig_risk = px.bar(
            risk_counts,
            x="Risk",
            y="Customers",
            color="Risk",
            color_discrete_map=RISK_COLORS,
            title="Customers by Risk Category",
            text="Customers"
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    debt_by_risk = (
        filtered_df.groupby("Default_Risk", as_index=False)[
            "Total_BNPL_Debt_KES"
        ]
        .sum()
    )
    debt_by_risk.columns = ["Risk", "Total Debt (KES)"]

    with chart_col2:
        fig_debt = px.bar(
            debt_by_risk,
            x="Risk",
            y="Total Debt (KES)",
            color="Risk",
            color_discrete_map=RISK_COLORS,
            title="Outstanding BNPL Debt by Risk Category"
        )
        st.plotly_chart(
            fig_debt,
            use_container_width=True
        )

    # -----------------------------------------
    # 4. Employment and late-payment patterns
    # -----------------------------------------
    st.markdown("### Customer Behaviour and Risk")

    behaviour_col1, behaviour_col2 = st.columns(2)

    employment_risk = (
        filtered_df.groupby(
            ["Employment_Status", "Default_Risk"]
        )
        .size()
        .reset_index(name="Customers")
    )

    with behaviour_col1:
        fig_employment = px.bar(
            employment_risk,
            x="Employment_Status",
            y="Customers",
            color="Default_Risk",
            color_discrete_map=RISK_COLORS,
            barmode="stack",
            title="Risk Profile by Employment Status"
        )
        st.plotly_chart(
            fig_employment,
            use_container_width=True
        )

    late_payment_risk = (
        filtered_df.groupby(
            ["Late_Payment_History", "Default_Risk"]
        )
        .size()
        .reset_index(name="Customers")
    )

    with behaviour_col2:
        fig_late = px.bar(
            late_payment_risk,
            x="Late_Payment_History",
            y="Customers",
            color="Default_Risk",
            color_discrete_map=RISK_COLORS,
            barmode="group",
            title="Risk Profile by Late-Payment History"
        )
        st.plotly_chart(
            fig_late,
            use_container_width=True
        )

    # -----------------------------------------
    # 5. Shopping category analysis
    # -----------------------------------------
    st.markdown("### Shopping Category Exposure")

    category_analysis = (
        filtered_df.groupby(
            "Shopping_Category_Most_Frequent",
            as_index=False
        )
        .agg(
            Customers=("Customer_ID", "count"),
            Total_Debt_KES=("Total_BNPL_Debt_KES", "sum")
        )
        .sort_values("Total_Debt_KES", ascending=False)
    )

    fig_category = px.bar(
        category_analysis,
        x="Shopping_Category_Most_Frequent",
        y="Total_Debt_KES",
        color="Customers",
        title="BNPL Debt Exposure by Shopping Category",
        labels={
            "Shopping_Category_Most_Frequent": "Shopping Category",
            "Total_Debt_KES": "Total Debt (KES)"
        }
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    # -----------------------------------------
    # 6. Management interpretation
    # -----------------------------------------
    st.markdown("### Management Insights")

    highest_debt_risk = (
        debt_by_risk.sort_values(
            "Total Debt (KES)", ascending=False
        ).iloc[0]["Risk"]
    )

    highest_category = (
        category_analysis.iloc[0][
            "Shopping_Category_Most_Frequent"
        ]
    )

    st.info(
        f"The selected portfolio contains {customer_count:,} customers, "
        f"with total outstanding BNPL debt of {format_kes(total_debt)}. "
        f"High-risk customers represent {high_risk_percentage:.2f}% "
        f"of the filtered portfolio."
    )

    st.write(
        f"**Largest debt exposure by risk:** {highest_debt_risk}."
    )
    st.write(
        f"**Largest shopping-category debt exposure:** "
        f"{highest_category}."
    )

    if high_risk_percentage >= 10:
        st.warning(
            "Management attention: high-risk customer concentration "
            "is at least 10% of the selected portfolio. "
            "Review affordability, repayment behaviour, and "
            "existing credit exposure."
        )
    else:
        st.success(
            "High-risk customer concentration is below the "
            "illustrative 10% monitoring threshold. "
            "Continue routine portfolio monitoring."
        )

    st.caption(
        "The 10% threshold is an illustrative management rule, "
        "not a validated regulatory or statistical threshold."
    )

    # -----------------------------------------
    # 7. Download filtered portfolio
    # -----------------------------------------
    st.markdown("### Download Portfolio Report")

    export_columns = [
        "Customer_ID",
        "Age",
        "Employment_Status",
        "Monthly_Income_KES",
        "Credit_Score",
        "Total_BNPL_Active_Loans",
        "Total_BNPL_Debt_KES",
        "Late_Payment_History",
        "Shopping_Category_Most_Frequent",
        "Default_Risk"
    ]

    export_df = filtered_df[export_columns].copy()

    st.download_button(
        label="Download Filtered Portfolio (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="mkopo_insights_portfolio_report.csv",
        mime="text/csv",
        key="download_portfolio_csv"
    )

    st.caption(
        "Demo dataset: monetary fields were converted from USD "
        "to KES using an assumed rate of 129.89. "
        "Risk categories are dataset labels, not verified "
        "Kenyan customer defaults or observed default rates. "
        "Use for demonstration and analysis only."
    )

# ==================================================
# 14. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.subheader("Model Performance")
    st.caption(
        "Evaluation of machine-learning models using cross-validation, "
        "held-out test results and permutation importance."
    )

    # -----------------------------------------
    # 1. Load actual evaluation reports
    # -----------------------------------------
    reports_dir = Path("reports")
    figures_dir = Path("figures")

    try:
        test_results = pd.read_csv(
            reports_dir / "test_set_results.csv"
        )

        cv_results = pd.read_csv(
            reports_dir / "cross_validation_results.csv"
        )

        importance_results = pd.read_csv(
            reports_dir / "permutation_importance.csv"
        )

    except Exception as e:
        st.error(f"Unable to load evaluation reports: {e}")
        st.stop()

    # -----------------------------------------
    # 2. Identify selected model
    # -----------------------------------------
    selected_model = (
        cv_results.sort_values(
            "Best CV Macro F1",
            ascending=False
        ).iloc[0]["Model"]
    )

    selected_test = test_results[
        test_results["Model"] == selected_model
    ]

    if selected_test.empty:
        st.error(
            "The selected model was not found in the test results."
        )
        st.stop()

    selected_test = selected_test.iloc[0]

    st.markdown("### Selected Model")

    st.success(
        f"{selected_model} was selected based on the highest "
        "cross-validation Macro F1."
    )

    # -----------------------------------------
    # 3. Selected model KPI cards
    # -----------------------------------------
    st.markdown("### Held-Out Test Performance")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        "Accuracy",
        f"{selected_test['Accuracy'] * 100:.2f}%"
    )

    kpi2.metric(
        "Macro Precision",
        f"{selected_test['Precision (Macro)'] * 100:.2f}%"
    )

    kpi3.metric(
        "Macro Recall",
        f"{selected_test['Recall (Macro)'] * 100:.2f}%"
    )

    kpi4.metric(
        "Macro F1-Score",
        f"{selected_test['Macro F1-Score'] * 100:.2f}%"
    )

    st.caption(
        "These metrics describe performance on the held-out test set. "
        "They are not actual future repayment or default rates."
    )

    st.divider()

    # -----------------------------------------
    # 4. Compare test-set performance
    # -----------------------------------------
    st.markdown("### Model Comparison — Test Set")

    metric_choice = st.selectbox(
        "Choose performance metric",
        [
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "Macro F1-Score"
        ]
    )

    comparison = test_results.sort_values(
        metric_choice,
        ascending=False
    )

    fig_comparison = px.bar(
        comparison,
        x="Model",
        y=metric_choice,
        color="Model",
        text=comparison[metric_choice].apply(
            lambda x: f"{x:.2%}"
        ),
        title=f"{metric_choice} by Model",
        range_y=[0, 1]
    )

    fig_comparison.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

    st.dataframe(
        test_results.style.format({
            "Accuracy": "{:.2%}",
            "Precision (Macro)": "{:.2%}",
            "Recall (Macro)": "{:.2%}",
            "Macro F1-Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------
    # 5. Cross-validation performance
    # -----------------------------------------
    st.markdown("### Cross-Validation Results")

    cv_sorted = cv_results.sort_values(
        "Best CV Macro F1",
        ascending=False
    )

    fig_cv = px.bar(
        cv_sorted,
        x="Model",
        y="Best CV Macro F1",
        color="Model",
        text=cv_sorted["Best CV Macro F1"].apply(
            lambda x: f"{x:.2%}"
        ),
        title="Best Cross-Validation Macro F1",
        range_y=[0, 1]
    )

    fig_cv.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_cv,
        use_container_width=True
    )

    st.info(
        f"{selected_model} achieved the highest cross-validation "
        f"Macro F1 of "
        f"{cv_results['Best CV Macro F1'].max():.2%}."
    )

    # -----------------------------------------
    # 6. Confusion matrix images
    # -----------------------------------------
    st.markdown("### Confusion Matrix Analysis")

    matrix_model = st.selectbox(
        "Select model to inspect",
        [
            "XGBoost",
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression"
        ],
        index=0
    )

    matrix_files = {
        "XGBoost": "xgboost_confusion_matrix.png",
        "Random Forest": "random_forest_confusion_matrix.png",
        "Gradient Boosting": "gradient_boosting_confusion_matrix.png",
        "Logistic Regression": "logistic_regression_confusion_matrix.png"
    }

    matrix_path = figures_dir / matrix_files[matrix_model]

    if matrix_path.exists():
        st.image(
            str(matrix_path),
            caption=f"{matrix_model} — Confusion Matrix",
            use_container_width=True
        )
    else:
        st.warning(
            f"Confusion matrix image not found: {matrix_path}"
        )

    st.caption(
        "The confusion matrix compares actual risk categories "
        "against model-predicted categories. Pay particular "
        "attention to High-risk customers predicted as Low risk."
    )

    # -----------------------------------------
    # 7. Feature importance
    # -----------------------------------------
    st.markdown("### Feature Importance")

    importance_sorted = importance_results.sort_values(
        "importance_mean",
        ascending=True
    )

    fig_importance = px.bar(
        importance_sorted,
        x="importance_mean",
        y="feature",
        error_x="importance_std",
        orientation="h",
        title="Permutation Importance — Selected Model",
        labels={
            "importance_mean": "Mean change in Macro F1",
            "feature": "Customer Feature"
        }
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.caption(
        "Permutation importance measures how much model performance "
        "changes when a feature is shuffled. Negative values indicate "
        "that shuffling slightly improved the score in this evaluation. "
        "Importance does not establish causation."
    )

    # -----------------------------------------
    # 8. Management interpretation
    # -----------------------------------------
    st.markdown("### Management Interpretation")

    best_test_model = test_results.loc[
        test_results["Macro F1-Score"].idxmax()
    ]

    st.write(
        f"**Cross-validation selection:** {selected_model}."
    )

    st.write(
        f"**Highest test Macro F1:** "
        f"{best_test_model['Model']} "
        f"({best_test_model['Macro F1-Score']:.2%})."
    )

    st.write(
        "**Key predictive features:** Monthly income and "
        "total BNPL debt have the highest permutation importance "
        "in this evaluation."
    )

    st.warning(
        "The difference between cross-validation and test-set "
        "rankings should be reviewed before production deployment. "
        "Further validation, including class-level performance, "
        "fairness, data leakage and external validation, is required."
    )

    # -----------------------------------------
    # 9. Download evaluation reports
    # -----------------------------------------
    st.markdown("### Download Model Evaluation Reports")

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:
        st.download_button(
            "Test Results CSV",
            data=test_results.to_csv(index=False).encode("utf-8"),
            file_name="test_set_results.csv",
            mime="text/csv"
        )

    with download_col2:
        st.download_button(
            "Cross-Validation CSV",
            data=cv_results.to_csv(index=False).encode("utf-8"),
            file_name="cross_validation_results.csv",
            mime="text/csv"
        )

    with download_col3:
        st.download_button(
            "Feature Importance CSV",
            data=importance_results.to_csv(index=False).encode("utf-8"),
            file_name="permutation_importance.csv",
            mime="text/csv"
        )

    st.divider()

    st.caption(
        "Important: This is a demonstration model trained using "
        "a dataset that is not a verified Kenyan lender portfolio. "
        "Its performance must not be interpreted as proof of "
        "future Kenyan BNPL repayment outcomes."
    )

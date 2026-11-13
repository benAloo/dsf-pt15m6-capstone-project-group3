from __future__ import annotations

import io
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth
import core
import theme

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Mkopo Insights",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_card_ids = itertools.count()


def card():
    """A bordered, styled container (the key gives the CSS a stable hook)."""
    return st.container(border=True, key=f"card{next(_card_ids)}")


INK, MUTED, LINE, LEAF = theme.INK, theme.MUTED, theme.LINE, theme.EMERALD
GREEN, AMBER, RED = theme.GREEN, theme.AMBER, theme.RED
TIERS = core.CLASSES
BANDS = [0, 18_047, 30_000, 50_000, 100_000, np.inf]
BAND_LABELS = ["Under 18K", "18K to 30K", "30K to 50K", "50K to 100K", "100K+"]
SESSION_HOURS = 8
MAX_ATTEMPTS, LOCK_SECONDS = 5, 60


DEMO_DEFAULT = True
DEMO_EMAIL = "group3@gmail.com"
DEMO_PASSWORD = "Group3@2026!!%$"


def demo_mode() -> bool:
    try:
        if "demo_mode" in st.secrets:
            return bool(st.secrets["demo_mode"])
    except Exception:  # noqa: BLE001  (no secrets file is the normal local case)
        pass
    return DEMO_DEFAULT


def sign_out() -> None:
    for key in ("auth_ok", "auth_time", "auth_user", "fails", "lock_until"):
        st.session_state.pop(key, None)


def render_login() -> None:
    demo = demo_mode()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            """
<div class='login-hero'>
  <div>
    <div class='brand'><div class='logo'>M</div><div><div class='brand-name'>Mkopo Insights</div>
    <div class='brand-sub'>BNPL credit risk for Kenyan lenders</div></div></div>
    <h2>Know the risk before you lend.</h2>
    <div class='lead'>Score customers, understand the reasons behind every score, and set approval cutoffs with
    calibrated probabilities built for the Kenyan market.</div>
    <ul>
      <li>Risk scores with customer-level reasons</li>
      <li>Batch scoring and a business-impact simulator</li>
      <li>Kenya data with every assumption documented</li>
    </ul>
  </div>
  <div class='foot'>Build by Group 3</div>
</div>""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
        with card():
            st.markdown(
                "<div class='login-title'>Admin sign in</div>"
                "<div class='login-sub'>Enter your admin credentials to open the dashboard.</div>",
                unsafe_allow_html=True,
            )
            now = time.time()
            remaining = int(st.session_state.get("lock_until", 0) - now)
            locked = remaining > 0
            with st.form("login", border=False):
                email = st.text_input("Email", value=DEMO_EMAIL if demo else "", placeholder="you@company.com", autocomplete="username")
                password = st.text_input("Password", type="password", value=DEMO_PASSWORD if demo else "", placeholder="Your password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch", disabled=locked)
            if demo:
                st.markdown("<div class='small'>Demo mode: the demo credentials are pre-filled. Press <b>Sign in</b> to continue.</div>", unsafe_allow_html=True)
            if locked:
                st.warning(f"Too many failed attempts. Try again in {remaining} seconds.")
            elif submitted:
                if auth.verify(email, password):
                    st.session_state.update(auth_ok=True, auth_time=time.time(), auth_user=email.strip().lower(), fails=0)
                    st.rerun()
                st.session_state["fails"] = st.session_state.get("fails", 0) + 1
                if st.session_state["fails"] >= MAX_ATTEMPTS:
                    st.session_state["lock_until"] = time.time() + LOCK_SECONDS
                    st.session_state["fails"] = 0
                    st.rerun()
                st.error("The email or password is incorrect.")
            st.markdown("<div class='small' style='margin:10px 0 6px 0'>Need access? Contact the project administrator.</div>", unsafe_allow_html=True)


if st.session_state.get("auth_ok") and time.time() - st.session_state.get("auth_time", 0) > SESSION_HOURS * 3600:
    sign_out()
if not st.session_state.get("auth_ok"):
    theme.inject(login=True)
    render_login()
    st.stop()
theme.inject()


@st.cache_resource(show_spinner=False)
def get_art() -> core.Artifacts:
    return core.load_artifacts(BASE)


@st.cache_data(show_spinner=False)
def get_portfolio() -> pd.DataFrame:
    art = get_art()
    df = pd.read_csv(BASE / "data" / "kenya_calibrated_dataset.csv")
    scored = core.score_customers(art, df)
    out = pd.concat([df, scored], axis=1)
    out["Debt_to_Income"] = out["Total_BNPL_Debt_KES"] / out["Monthly_Income_KES"].clip(lower=1)
    out["Income_Band"] = pd.cut(out["Monthly_Income_KES"], bins=BANDS, labels=BAND_LABELS, right=False)
    return out


@st.cache_data(show_spinner=False)
def get_reports() -> dict[str, pd.DataFrame]:
    names = ["model_comparison", "confusion_matrix", "per_class_report", "feature_importance", "reliability",
             "calibration_metrics", "fairness", "sensitivity", "test_predictions"]
    out = {}
    for n in names:
        p = BASE / "reports" / f"{n}.csv"
        if p.exists():
            out[n] = pd.read_csv(p, index_col=0) if n in ("confusion_matrix", "per_class_report") else pd.read_csv(p)
    return out


ART = get_art()
META = ART.meta
TH = ART.thresholds
PORT = get_portfolio()
REP = get_reports()
FX_DEFAULT = float(META["calibration"]["fx_usd_kes"])


def style(fig: go.Figure, h: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=6, r=6, t=34, b=6),
        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(family="Bricolage Grotesque, sans-serif", size=14), showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(font_family="Plus Jakarta Sans, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#EDF1EE", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EDF1EE", zeroline=False, linecolor=LINE)
    return fig


def page_header(title: str, sub: str) -> None:
    theme.hero(title, sub, tag=st.session_state.get("page", "Mkopo Insights"))


def kpi_strip(items: list[tuple]) -> None:
    theme.kpi_grid(items)


def callout(text: str, warn: bool = False) -> None:
    theme.callout(text, warn)


def badge(tier: str) -> str:
    return theme.badge(tier)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------
# Currency: every amount the user types can be entered in KSh or USD.
# The model always works in KSh, so USD entries are converted before scoring.
# ---------------------------------------------------------------------------
CURRENCIES = ["KSh", "USD"]
st.session_state.setdefault("cur", "KSh")
st.session_state.setdefault("fx", FX_DEFAULT)


def currency_controls(scope: str, fields: tuple[str, ...] = ()) -> tuple[str, float]:
    """Currency picker plus exchange rate. `fields` are the money inputs on the page, so a new
    exchange rate can be applied to the USD amounts the user has already typed."""

    def on_currency() -> None:
        st.session_state["cur"] = st.session_state[f"cur_{scope}"]

    def on_rate() -> None:
        new = float(st.session_state[f"fx_{scope}"])
        for f in fields:  # keep the typed USD amounts, update their shilling equivalent
            wkey = f"w_{f}_USD"
            if wkey in st.session_state:
                st.session_state[f"kes_{f}"] = float(st.session_state[wkey]) * new
        st.session_state["fx"] = new

    c1, c2 = st.columns([1, 1])
    cur = c1.radio("Enter amounts in", CURRENCIES, index=CURRENCIES.index(st.session_state["cur"]), horizontal=True,
                   key=f"cur_{scope}", on_change=on_currency)
    if cur == "USD":
        rate = c2.number_input("Rate (KSh per USD)", 50.0, 400.0, float(st.session_state["fx"]), step=0.5, key=f"fx_{scope}", on_change=on_rate)
    else:
        c2.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        rate = float(st.session_state["fx"])
    return cur, float(rate)


def money_input(label: str, field: str, cur: str, rate: float, kes_default: float, kes_max: float,
                kes_step: float, usd_step: float, kes_min: float = 0.0) -> float:
    """A money field in the chosen currency. Returns the amount in KSh."""
    canon = f"kes_{field}"
    st.session_state.setdefault(canon, float(kes_default))
    usd = cur == "USD"
    wkey = f"w_{field}_{cur}"

    def sync() -> None:
        v = float(st.session_state[wkey])
        st.session_state[canon] = v * float(st.session_state["fx"]) if usd else v

    kes = float(st.session_state[canon])
    val = st.number_input(
        f"{label} ({cur})",
        min_value=float(kes_min / FX_DEFAULT if usd else kes_min),
        max_value=float(kes_max / FX_DEFAULT if usd else kes_max),
        value=float(round(kes / rate, 2) if usd else round(kes)),
        step=float(usd_step if usd else kes_step),
        format="%.2f" if usd else "%.0f",
        key=wkey, on_change=sync,
    )
    kes_val = val * rate if usd else val
    if usd:
        st.caption(f"About KSh {round(kes_val, -1):,.0f}")
    return float(kes_val)


def money(kes: float, short: bool = False) -> str:
    """Format a KSh amount in the currency the user chose."""
    if st.session_state.get("cur") == "USD":
        v = kes / float(st.session_state["fx"])
        if short and abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        if short and abs(v) >= 1e3:
            return f"${v / 1e3:.1f}K"
        return f"${v:,.0f}"
    return core.kes_short(kes) if short else core.kes(kes)


def fmt_value(feature: str, row: pd.Series) -> str:
    v = row[feature]
    if feature in ("Monthly_Income_KES", "Total_BNPL_Debt_KES", "Average_Transaction_Value_KES"):
        return money(float(v))
    if feature == "Debt_to_Income":
        return f"{float(v):.2f}x monthly income"
    if feature in ("Age", "Credit_Score", "Total_BNPL_Active_Loans"):
        return f"{float(v):,.0f}"
    return str(v)


PAGES = [
    ("Overview", ":material/space_dashboard:"),
    ("Risk scorecard", ":material/speed:"),
    ("Batch scoring", ":material/upload_file:"),
    ("Business impact", ":material/payments:"),
    ("Portfolio explorer", ":material/travel_explore:"),
    ("Model and governance", ":material/verified_user:"),
]
st.session_state.setdefault("page", "Overview")


def set_page(name: str) -> None:
    st.session_state["page"] = name


with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='logo'>M</div><div><div class='brand-name'>Mkopo Insights</div>"
        "<div class='brand-sub'>BNPL credit risk for Kenyan lenders</div></div></div>"
        "<div class='side-label'>Workspace</div>",
        unsafe_allow_html=True,
    )
    for name, icon in PAGES:
        st.button(f"{icon} {name}", key=f"nav_{name}", type="primary" if st.session_state["page"] == name else "secondary",
                  on_click=set_page, args=(name,), width="stretch")
    tm = META["test_metrics"]
    st.markdown(
        f"<div class='side-card'><b>Demo build</b><br>Not observed repayments.<br><br>"
        f"Model: {META['model_name']}<br>Test macro F1: {tm['macro_f1']:.2f}<br>Customers: {len(PORT):,}</div>"
        f"<div class='side-label'>Signed in</div>"
        f"<div class='side-card'><b>Administrator</b><br>{st.session_state.get('auth_user', '')}</div>",
        unsafe_allow_html=True,
    )
    st.button(":material/logout: Sign out", key="signout", on_click=sign_out, width="stretch")
page = st.session_state["page"]


def page_overview() -> None:
    page_header(
        "Portfolio risk at a glance",
        "Where BNPL customers and shillings sit across risk tiers, and what separates the safe book from the stretched one.",
    )
    df = PORT
    exposure = df["Total_BNPL_Debt_KES"].sum()
    by_tier_n = df["Risk_Tier"].value_counts().reindex(TIERS).fillna(0)
    by_tier_x = df.groupby("Risk_Tier")["Total_BNPL_Debt_KES"].sum().reindex(TIERS).fillna(0)
    flagged = (df["Risk_Tier"] != "Low").mean()
    high_x = by_tier_x["High"] / exposure

    kpi_strip([
        ("Customers", f"{len(df):,}", "In the scored portfolio"),
        ("BNPL exposure", core.kes_short(exposure), f"Median {core.kes(df['Total_BNPL_Debt_KES'].median())} per customer"),
        ("Flagged Medium or High", pct(flagged), f"{int((df['Risk_Tier'] != 'Low').sum()):,} customers", AMBER),
        ("Exposure in High tier", pct(high_x), core.kes_short(by_tier_x["High"]), RED),
        ("Median monthly income", core.kes(df["Monthly_Income_KES"].median()), "All employment types"),
    ])

    fig = go.Figure()
    for tier in TIERS:
        fig.add_bar(
            y=["Customers", "Exposure (KSh)"], x=[by_tier_n[tier] / by_tier_n.sum(), by_tier_x[tier] / exposure],
            orientation="h", name=tier, marker_color=core.TIER_COLORS[tier],
            text=[pct(by_tier_n[tier] / by_tier_n.sum()), pct(by_tier_x[tier] / exposure)],
            textposition="inside", insidetextanchor="middle", textfont=dict(color="#fff", size=13),
            hovertemplate="%{y}: %{text}<extra>" + tier + "</extra>",
        )
    fig.update_layout(barmode="stack", xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(autorange="reversed"))
    with card():
        st.markdown("#### Risk ribbon")
        st.caption("Share of customers and share of outstanding BNPL shillings in each tier. A wider red band in the lower bar means the risk is concentrated in larger balances.")
        st.plotly_chart(style(fig, 170), width="stretch")

    c1, c2 = st.columns(2)
    with c1, card():
        st.markdown("#### Risk score distribution")
        fig = go.Figure()
        for tier in TIERS:
            d = df[df["Risk_Tier"] == tier]["Risk_Score"]
            fig.add_histogram(x=d, name=tier, marker_color=core.TIER_COLORS[tier], xbins=dict(start=0, end=100, size=4), opacity=0.9)
        fig.add_vline(x=100 * TH["medium"], line_dash="dash", line_color=INK,
                      annotation_text="Medium cutoff", annotation_position="top")
        fig.update_layout(barmode="stack", xaxis_title="Risk score (0 to 100)", yaxis_title="Customers")
        st.plotly_chart(style(fig, 330), width="stretch")
    with c2, card():
        st.markdown("#### Tier mix by employment status")
        mix = pd.crosstab(df["Employment_Status"], df["Risk_Tier"], normalize="index").reindex(columns=TIERS)
        mix = mix.loc[core.EMPLOYMENT_OPTIONS]
        fig = go.Figure()
        for tier in TIERS:
            fig.add_bar(y=mix.index, x=mix[tier], name=tier, orientation="h", marker_color=core.TIER_COLORS[tier],
                        text=[pct(v, 0) if v > 0.06 else "" for v in mix[tier]], textfont=dict(color="#fff"))
        fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 330), width="stretch")

    c3, c4 = st.columns([3, 2])
    with c3, card():
        st.markdown("#### Income against Kenyan benchmarks")
        st.caption("Monthly income of every customer, with the statutory minimum wage and KNBS formal-sector reference points.")
        cal = META["calibration"]["benchmarks"]
        fig = go.Figure()
        for tier in TIERS:
            d = df[df["Risk_Tier"] == tier]["Monthly_Income_KES"].clip(lower=500)
            fig.add_histogram(x=np.log10(d), name=tier, marker_color=core.TIER_COLORS[tier], opacity=0.9,
                              xbins=dict(start=2.6, end=5.6, size=0.06), hovertemplate="%{y} customers<extra>" + tier + "</extra>")
        ticks = [1000, 3000, 10000, 30000, 100000, 300000]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=["1K", "3K", "10K", "30K", "100K", "300K"], title="Monthly income (KSh, log scale)")
        for v, lab in [(cal["min_wage_city"], "City minimum wage"), (cal["est_formal_median_monthly"], "Formal median"), (cal["knbs_formal_avg_monthly"], "KNBS average")]:
            fig.add_vline(x=np.log10(v), line_dash="dot", line_color=INK, annotation_text=lab, annotation_position="top", annotation_font_size=10)
        fig.update_layout(barmode="stack", yaxis_title="Customers")
        st.plotly_chart(style(fig, 340), width="stretch")
    with c4, card():
        st.markdown("#### Risk by income band")
        band = df.groupby("Income_Band", observed=True).agg(customers=("Risk_Tier", "size"), flagged=("Risk_Tier", lambda s: (s != "Low").mean())).reindex(BAND_LABELS)
        fig = go.Figure(go.Bar(x=band["flagged"], y=band.index, orientation="h", marker_color=LEAF,
                               text=[f"{pct(f, 0)}  ({int(n):,})" for f, n in zip(band["flagged"], band["customers"])],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, max(0.1, band["flagged"].max() * 1.35)], title="Share flagged Medium or High"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 340, legend=False), width="stretch")

    low_inc = df[df["Monthly_Income_KES"] < 30_000]
    hi = df[df["Risk_Tier"] == "High"]
    share = (hi["Monthly_Income_KES"] < 30_000).mean() if len(hi) else 0
    callout(f"<b>What stands out.</b> {pct(share, 0)} of High-tier customers earn under KSh 30,000 a month, "
            f"and {pct((low_inc['Risk_Tier'] != 'Low').mean(), 0)} of everyone in that income range is flagged, against "
            f"{pct((df[df['Monthly_Income_KES'] >= 30_000]['Risk_Tier'] != 'Low').mean(), 0)} above it.")


def gauge(score: float, tier: str) -> go.Figure:
    cut = 100 * TH["medium"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=46, color=INK), valueformat=".0f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=MUTED),
            bar=dict(color=core.TIER_COLORS[tier], thickness=0.3), bgcolor="#fff", borderwidth=0,
            steps=[dict(range=[0, cut], color="#DCEFE5"), dict(range=[cut, 100], color="#F7E3C3")],
            threshold=dict(line=dict(color=INK, width=3), thickness=0.85, value=cut),
        ),
    ))
    fig.update_layout(height=250, margin=dict(l=40, r=40, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif"))
    return fig


def page_scorecard() -> None:
    page_header("Risk scorecard", "Score one customer, see the reasons behind the score, and get a suggested credit limit and next step.")

    left, right = st.columns([2, 3], gap="large")
    with left, card():
        st.markdown("#### Customer details")
        cur, rate = currency_controls("score", ("inc", "debt", "txn"))
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 80, 35, step=1, key="f_age")
        emp = c2.selectbox("Employment", core.EMPLOYMENT_OPTIONS, key="f_emp")
        inc = money_input("Monthly income", "inc", cur, rate, 56_000, 2_000_000, 1_000, 10)
        credit = st.slider("Credit score", 300, 850, 650, key="f_credit")
        loans = st.slider("Active BNPL loans", 0, 15, 2, key="f_loans")
        debt = money_input("Outstanding BNPL debt", "debt", cur, rate, 3_000, 5_000_000, 500, 5)
        txn = money_input("Average purchase value", "txn", cur, rate, 3_000, 500_000, 500, 5)
        c3, c4 = st.columns(2)
        late = c3.selectbox("Paid late before", core.YES_NO, key="f_late")
        cat = c4.selectbox("Usual purchase", core.SHOPPING_OPTIONS, key="f_cat")

    row = pd.DataFrame([{
        "Age": age, "Employment_Status": emp, "Monthly_Income_KES": inc, "Credit_Score": credit,
        "Total_BNPL_Active_Loans": loans, "Total_BNPL_Debt_KES": debt, "Late_Payment_History": late,
        "Shopping_Category_Most_Frequent": cat, "Average_Transaction_Value_KES": txn,
    }])
    res = core.score_customers(ART, row).iloc[0]
    tier = res["Risk_Tier"]

    q = PORT[["Monthly_Income_KES", "Total_BNPL_Debt_KES", "Average_Transaction_Value_KES"]].quantile([0.005, 0.995])
    outside = [
        f"{lab} ({money(v)} entered; most training customers were between {money(q.loc[0.005, col])} and {money(q.loc[0.995, col])})"
        for col, lab, v in [("Monthly_Income_KES", "monthly income", inc), ("Total_BNPL_Debt_KES", "BNPL debt", debt),
                            ("Average_Transaction_Value_KES", "average purchase value", txn)]
        if v < q.loc[0.005, col] or v > q.loc[0.995, col]
    ]
    with right:
        if outside:
            callout("<b>Outside the training range.</b> The model saw very few customers like this on: " + "; ".join(outside)
                    + ". The score may be unreliable.", warn=True)
        with card():
            h1, h2 = st.columns([3, 2])
            h1.markdown("#### Decision")
            h2.markdown(f"<div style='text-align:right'>{badge(tier)}</div>", unsafe_allow_html=True)
            g1, g2 = st.columns([3, 2])
            g1.plotly_chart(gauge(res["Risk_Score"], tier), width="stretch")
            with g2:
                st.markdown(f"<div class='small'>Probability of each class</div>", unsafe_allow_html=True)
                for t, key in zip(TIERS, ["P_Low", "P_Medium", "P_High"]):
                    st.markdown(f"<div style='margin:6px 0 0 0'><b>{t}</b> <span class='small'>{pct(res[key])}</span></div>", unsafe_allow_html=True)
                    st.progress(float(res[key]))
            st.caption(f"The line on the gauge is the Medium cutoff ({100 * TH['medium']:.0f}). A customer is High when the probability of High is at least {pct(TH['high'], 0)}.")

        with st.expander("Credit policy settings (illustrative)"):
            pc = st.columns(3)
            limits = {t: pc[i].slider(f"{t} tier: limit as share of monthly income", 0, 100, int(100 * core.POLICY[t]["limit_pct"]), key=f"lim_{t}") / 100 for i, t in enumerate(TIERS)}
        limit = max(0.0, limits[tier] * inc - debt)
        dti = debt / max(inc, 1)
        same = PORT[PORT["Employment_Status"] == emp]["Monthly_Income_KES"]
        kpi_strip([
            ("Suggested new BNPL limit", money(limit), f"{pct(limits[tier], 0)} of income, minus existing debt", core.TIER_COLORS[tier]),
            ("Debt relative to income", f"{dti:.2f}x", "BNPL debt divided by monthly income"),
            ("Income percentile", ordinal(int(round(100 * (same < inc).mean()))), f"Among {emp.lower()} customers"),
        ])
        callout(f"<b>Recommended next step.</b> {core.POLICY[tier]['action']}", warn=tier != "Low")

    contrib = core.explain(ART, row)
    with card():
        st.markdown("#### Why this score")
        if contrib is None:
            st.info("Customer-level reasons are available when the deployed model is XGBoost.")
        else:
            vals = contrib.iloc[0].sort_values()
            top = vals.reindex(vals.abs().sort_values(ascending=False).index[:7]).sort_values()
            feat_row = core.add_features(row).iloc[0]
            fig = go.Figure(go.Bar(
                x=top.values, y=[f"{core.FEATURE_LABELS[f]}: {fmt_value(f, feat_row)}" for f in top.index], orientation="h",
                marker_color=[RED if v > 0 else GREEN for v in top.values],
                hovertemplate="%{y}<br>Effect on risk: %{x:+.2f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            fig.update_layout(xaxis_title="Lowers risk  <  0  >  raises risk")
            st.plotly_chart(style(fig, 300, legend=False), width="stretch")
            up = [core.FEATURE_LABELS[f].lower() for f in vals[vals > 0.05].sort_values(ascending=False).index[:3]]
            dn = [core.FEATURE_LABELS[f].lower() for f in vals[vals < -0.05].sort_values().index[:3]]
            lines = []
            if up:
                lines.append("Raising the score most: " + ", ".join(up) + ".")
            if dn:
                lines.append("Working in the customer's favour: " + ", ".join(dn) + ".")
            st.markdown(" ".join(lines) if lines else "No single factor dominates this score.")

    out = row.assign(Risk_Score=round(res["Risk_Score"], 1), Risk_Tier=tier, Suggested_Limit_KES=round(limit))
    st.download_button("Download this decision (CSV)", out.to_csv(index=False).encode(), "decision_record.csv", "text/csv")


MONEY_COLS = ["Monthly_Income_KES", "Total_BNPL_Debt_KES", "Average_Transaction_Value_KES"]


def template_frame(cur: str = "KSh", rate: float = FX_DEFAULT) -> pd.DataFrame:
    df = pd.DataFrame([
        ["C001", 34, "Employed", 95000, 720, 1, 6000, "No", "Electronics", 4500],
        ["C002", 29, "Freelancer", 38000, 640, 3, 21000, "Yes", "Fashion", 3200],
        ["C003", 21, "Student", 9000, 610, 2, 6500, "Yes", "Fashion", 2000],
        ["C004", 41, "Employed", 32000, 540, 6, 75000, "Yes", "Travel", 5000],
        ["C005", 47, "Employed", 52000, 690, 2, 9000, "No", "Home/Furniture", 6000],
    ], columns=["Customer_ID", "Age", "Employment_Status", "Monthly_Income_KES", "Credit_Score", "Total_BNPL_Active_Loans",
                "Total_BNPL_Debt_KES", "Late_Payment_History", "Shopping_Category_Most_Frequent", "Average_Transaction_Value_KES"])
    if cur == "USD":
        df[MONEY_COLS] = (df[MONEY_COLS] / rate).round(2)
    return df


def validate_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    missing = [c for c in core.RAW_INPUT_COLS if c not in df.columns]
    if missing:
        return df.iloc[0:0], [f"Missing columns: {', '.join(missing)}. Download the template to see the expected format."]
    d = df.copy()
    for c in ["Age", "Monthly_Income_KES", "Credit_Score", "Total_BNPL_Active_Loans", "Total_BNPL_Debt_KES", "Average_Transaction_Value_KES"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = pd.Series(False, index=d.index)
    for col, allowed in [("Employment_Status", core.EMPLOYMENT_OPTIONS), ("Late_Payment_History", core.YES_NO), ("Shopping_Category_Most_Frequent", core.SHOPPING_OPTIONS)]:
        invalid = ~d[col].isin(allowed)
        if invalid.any():
            problems.append(f"{int(invalid.sum())} row(s) have a {col} outside: {', '.join(allowed)}.")
        bad |= invalid
    neg = (d[["Age", "Monthly_Income_KES", "Total_BNPL_Debt_KES", "Average_Transaction_Value_KES"]] < 0).any(axis=1)
    if neg.any():
        problems.append(f"{int(neg.sum())} row(s) have negative values.")
    bad |= neg
    if bad.any():
        problems.append(f"{int(bad.sum())} row(s) were skipped. Fix them and upload again to include them.")
    return d[~bad], problems


def page_batch() -> None:
    page_header("Batch scoring", "Upload a customer file, score every row, and download the results with tiers and suggested actions.")
    with card():
        cur, rate = currency_controls("batch")
        st.caption("Amounts in your file are read in this currency. Column names stay the same (the *_KES columns). Results are shown in KSh."
                   if cur == "USD" else "Amounts in your file are read as KSh.")
    c1, c2, c3 = st.columns([2, 1, 1])
    upload = c1.file_uploader("Customer file (CSV)", type=["csv"], label_visibility="collapsed")
    c2.download_button(f"Download CSV template ({cur})", template_frame(cur, rate).to_csv(index=False).encode(), "mkopo_batch_template.csv", "text/csv", width="stretch")
    if c3.button("Try a sample portfolio", width="stretch"):
        sample = pd.read_csv(BASE / "data" / "kenya_calibrated_dataset.csv").sample(250, random_state=7)
        sample = sample[["Customer_ID"] + core.RAW_INPUT_COLS].reset_index(drop=True)
        if cur == "USD":  # keep the sample consistent with the chosen currency
            sample[MONEY_COLS] = (sample[MONEY_COLS] / rate).round(2)
        st.session_state["batch_df"] = sample
    if upload is not None:
        try:
            st.session_state["batch_df"] = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"That file could not be read as a CSV ({exc}). Check the format and try again.")
            return

    raw = st.session_state.get("batch_df")
    if raw is None:
        callout("Upload a CSV, or choose <b>Try a sample portfolio</b> to see the output on 250 example customers. Required columns: "
                + ", ".join(f"<code>{c}</code>" for c in core.RAW_INPUT_COLS) + ". Customer_ID is optional.")
        return

    valid, problems = validate_batch(raw)
    for p in problems:
        st.warning(p)
    if valid.empty:
        return
    if cur == "USD":  # the model works in KSh
        valid = valid.copy()
        valid[MONEY_COLS] = valid[MONEY_COLS] * rate
    scored = core.score_customers(ART, valid)
    out = pd.concat([valid.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    out["Suggested_Action"] = out["Risk_Tier"].map(lambda t: core.POLICY[t]["action"].split(".")[0] + ".")
    counts = out["Risk_Tier"].value_counts().reindex(TIERS).fillna(0).astype(int)

    kpi_strip([
        ("Customers scored", f"{len(out):,}", f"{len(raw) - len(out)} skipped" if len(raw) != len(out) else "All rows valid"),
        ("Low", f"{counts['Low']:,}", pct(counts["Low"] / len(out))),
        ("Medium", f"{counts['Medium']:,}", pct(counts["Medium"] / len(out))),
        ("High", f"{counts['High']:,}", pct(counts["High"] / len(out))),
        ("Average risk score", f"{out['Risk_Score'].mean():.0f}", "Out of 100"),
    ])
    a, b = st.columns([1, 2])
    with a, card():
        fig = go.Figure(go.Pie(labels=TIERS, values=[counts[t] for t in TIERS], hole=0.62, marker_colors=[core.TIER_COLORS[t] for t in TIERS], sort=False, textinfo="percent"))
        st.plotly_chart(style(fig, 250), width="stretch")
    with b, card():
        show = out.sort_values("Risk_Score", ascending=False)
        cols = (["Customer_ID"] if "Customer_ID" in show.columns else []) + ["Risk_Tier", "Risk_Score", "P_High", "Employment_Status", "Monthly_Income_KES", "Total_BNPL_Debt_KES", "Late_Payment_History"]
        styled = show[cols].head(200).style.map(lambda v: f"background-color:{core.TIER_COLORS.get(v, 'white')};color:white;font-weight:600" if v in core.TIER_COLORS else "", subset=["Risk_Tier"])
        st.dataframe(
            styled, hide_index=True, height=250, width="stretch",
            column_config={
                "Risk_Score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
                "P_High": st.column_config.NumberColumn("P(High)", format="%.2f"),
                "Monthly_Income_KES": st.column_config.NumberColumn("Monthly income", format="KSh %d"),
                "Total_BNPL_Debt_KES": st.column_config.NumberColumn("BNPL debt", format="KSh %d"),
            },
        )
        st.caption("Showing the 200 highest-risk rows. The download contains every scored row, with amounts in KSh.")
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("Download scored file (CSV)", buf.getvalue().encode(), "mkopo_scored_customers.csv", "text/csv", type="primary")


def use_best_cutoff() -> None:
    st.session_state["cutoff"] = st.session_state["_best_cutoff"]


def page_business() -> None:
    page_header("Business impact", "Choose how strict to be. See how many customers you approve, how many bad loans slip through, and what that does to profit.")
    tp = REP.get("test_predictions")
    if tp is None:
        st.error("The test predictions file is missing. Re-run the notebook to regenerate it.")
        return
    base = META["business_assumptions"]
    with card():
        st.markdown("#### Assumptions")
        cur, rate = currency_controls("biz", ("exposure",))
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            exposure = money_input("Average exposure per customer", "exposure", cur, rate, float(base["exposure"]), 500_000, 500, 5, kes_min=500)
        lgd = a2.slider("Loss if a customer defaults", 10, 100, int(100 * base["lgd"]), help="Share of the exposure that is lost on a default.") / 100
        margin = a3.slider("Margin on a performing loan", 1, 40, int(100 * base["margin"]), help="Fees and interest earned, as a share of exposure.") / 100
        mw = a4.slider("Chance a Medium customer defaults", 0, 100, int(100 * base["medium_weight"])) / 100
        st.caption("These are placeholders for a demo. Replace them with your own economics. High customers are treated as defaults and Low customers as performing.")

    cutoffs = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    score, actual = tp["Risk_Score"].values / 100, tp["Actual_Risk"].values
    curve = core.business_curve(score, actual, cutoffs, exposure, lgd, margin, mw)
    best = curve.loc[curve["Expected_Profit"].idxmax()]
    everyone = core.business_curve(score, actual, np.array([1.01]), exposure, lgd, margin, mw).iloc[0]

    st.session_state["_best_cutoff"] = int(round(100 * best["Cutoff"]))
    if "cutoff" not in st.session_state:
        st.session_state["cutoff"] = st.session_state["_best_cutoff"]
    cc1, cc2 = st.columns([4, 1])
    cutoff = cc1.slider("Approve customers whose risk score is below", 1, 100, key="cutoff", help="Lower is stricter.") / 100
    cc2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    cc2.button("Use best cutoff", width="stretch", on_click=use_best_cutoff)
    sel = core.business_curve(score, actual, np.array([cutoff]), exposure, lgd, margin, mw).iloc[0]
    declined = score >= cutoff
    declined_bad = (actual[declined] != "Low").mean() if declined.any() else 0.0

    uplift = sel["Expected_Profit"] - everyone["Expected_Profit"]
    kpi_strip([
        ("Approval rate", pct(sel["Approval_Rate"]), f"{int(sel['Approved']):,} of {len(tp):,} test customers"),
        ("Share of approved that go bad", pct(sel["Bad_Rate"]), f"Approving everyone: {pct(everyone['Bad_Rate'])}"),
        ("Expected profit", money(sel["Expected_Profit"], True), f"{'+' if uplift >= 0 else '-'}{money(abs(uplift), True)} vs approving everyone"),
        ("Losses avoided", money(everyone["Expected_Loss"] - sel["Expected_Loss"], True), "Compared with approving everyone"),
        ("Declined that were risky", pct(declined_bad), "Medium or High among declined"),
    ])

    with card():
        st.markdown("#### Profit and approvals by cutoff")
        from plotly.subplots import make_subplots

        unit = 1 / rate if cur == "USD" else 1.0
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=curve["Cutoff"] * 100, y=curve["Expected_Profit"] * unit / 1e6, name=f"Expected profit ({cur} million)", line=dict(color=LEAF, width=3), secondary_y=False)
        fig.add_scatter(x=curve["Cutoff"] * 100, y=curve["Approval_Rate"], name="Approval rate", line=dict(color=MUTED, width=2, dash="dot"), secondary_y=True)
        fig.add_scatter(x=[100 * best["Cutoff"]], y=[best["Expected_Profit"] * unit / 1e6], mode="markers+text", name="Profit-maximising cutoff", marker=dict(color=INK, size=11),
                        text=[f"Best: {100 * best['Cutoff']:.0f}"], textposition="top center", secondary_y=False)
        fig.add_vline(x=100 * cutoff, line_dash="dash", line_color=AMBER, annotation_text="Your cutoff", annotation_position="bottom")
        fig.update_xaxes(title="Risk-score cutoff")
        fig.update_yaxes(title=f"Expected profit ({cur} million)", secondary_y=False)
        fig.update_yaxes(title="Approval rate", tickformat=".0%", secondary_y=True, showgrid=False)
        st.plotly_chart(style(fig, 360), width="stretch")
    callout("Profit rises as risky customers are screened out, then falls as good customers are turned away. The best cutoff is where those two forces balance "
            "under your assumptions. Risk classes in the data are labels, not recorded defaults, so treat the money figures as directional.", warn=True)


def page_portfolio() -> None:
    page_header("Portfolio explorer", "Filter the book, find where risk concentrates, and export a watchlist for follow-up.")
    df = PORT
    with card():
        cur, rate = currency_controls("port")
        f1, f2, f3, f4 = st.columns([2, 2, 3, 1.4])
        tiers = f1.multiselect("Risk tier", TIERS, default=TIERS)
        emps = f2.multiselect("Employment", core.EMPLOYMENT_OPTIONS, default=core.EMPLOYMENT_OPTIONS)
        if cur == "USD":
            lo, hi = f3.slider("Monthly income (USD)", 0, 3000, (0, 3000), step=50, key="inc_rng_USD")
            lo_kes, hi_kes = lo * rate, (hi * rate if hi < 3000 else 1e9)
        else:
            lo, hi = f3.slider("Monthly income (KSh '000)", 0, 400, (0, 400), step=5, key="inc_rng_KSh")
            lo_kes, hi_kes = lo * 1000, (hi * 1000 if hi < 400 else 1e9)
        late = f4.selectbox("Late payments", ["Any", "Yes", "No"])
    m = df["Risk_Tier"].isin(tiers) & df["Employment_Status"].isin(emps) & df["Monthly_Income_KES"].between(lo_kes, hi_kes)
    if late != "Any":
        m &= df["Late_Payment_History"] == late
    f = df[m]
    if f.empty:
        st.info("No customers match these filters. Widen the income range or add more tiers.")
        return
    kpi_strip([
        ("Customers", f"{len(f):,}", pct(len(f) / len(df)) + " of portfolio"),
        ("BNPL exposure", money(f["Total_BNPL_Debt_KES"].sum(), True), f"Median {money(f['Total_BNPL_Debt_KES'].median())}"),
        ("Average risk score", f"{f['Risk_Score'].mean():.0f}", "Out of 100"),
        ("Median debt relative to income", f"{f['Debt_to_Income'].median():.2f}x", "Months of income owed"),
        ("Paid late before", pct((f["Late_Payment_History"] == "Yes").mean(), 0), "Of filtered customers"),
    ])
    a, b = st.columns(2)
    with a, card():
        st.markdown("#### Share flagged, employment by income band")
        heat = df[m].assign(flag=lambda d: d["Risk_Tier"] != "Low").pivot_table(index="Employment_Status", columns="Income_Band", values="flag", aggfunc="mean", observed=False)
        heat = heat.reindex(index=[e for e in core.EMPLOYMENT_OPTIONS if e in heat.index], columns=BAND_LABELS)
        fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns.astype(str), y=heat.index, colorscale=[[0, "#E7F3EC"], [0.5, "#F2C879"], [1, RED]], zmin=0, zmax=1,
                                   text=np.where(np.isnan(heat.values), "", np.round(100 * heat.values).astype("float").astype(str)), hovertemplate="%{y}, %{x}: %{z:.0%}<extra></extra>",
                                   colorbar=dict(tickformat=".0%", thickness=10)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 330, legend=False), width="stretch")
        st.caption("Income bands are in KSh. Blank cells have no customers in the current filter.")
    with b, card():
        st.markdown("#### Income and debt load")
        s = f.sample(min(len(f), 2500), random_state=1)
        fig = go.Figure()
        for tier in TIERS:
            d = s[s["Risk_Tier"] == tier]
            fig.add_scatter(x=d["Monthly_Income_KES"], y=d["Debt_to_Income"], mode="markers", name=tier, marker=dict(color=core.TIER_COLORS[tier], size=5, opacity=0.65),
                            hovertemplate="Income KSh %{x:,.0f}<br>Debt/income %{y:.2f}x<extra>" + tier + "</extra>")
        fig.update_xaxes(type="log", title="Monthly income (KSh, log scale)")
        fig.update_yaxes(title="Debt relative to income (x)")
        st.plotly_chart(style(fig, 330), width="stretch")

    with card():
        st.markdown("#### Watchlist: highest-risk customers in this view")
        cols = ["Customer_ID", "Risk_Tier", "Risk_Score", "Employment_Status", "Age", "Monthly_Income_KES", "Total_BNPL_Debt_KES", "Debt_to_Income", "Total_BNPL_Active_Loans", "Late_Payment_History", "Credit_Score"]
        wl = f.sort_values("Risk_Score", ascending=False)[cols]
        st.dataframe(
            wl.head(50).style.map(lambda v: f"background-color:{core.TIER_COLORS.get(v, 'white')};color:white;font-weight:600" if v in core.TIER_COLORS else "", subset=["Risk_Tier"]),
            hide_index=True, height=320, width="stretch",
            column_config={
                "Risk_Score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
                "Monthly_Income_KES": st.column_config.NumberColumn("Monthly income", format="KSh %d"),
                "Total_BNPL_Debt_KES": st.column_config.NumberColumn("BNPL debt", format="KSh %d"),
                "Debt_to_Income": st.column_config.NumberColumn("Debt / income", format="%.2fx"),
            },
        )
        st.download_button("Download filtered customers (CSV)", wl.to_csv(index=False).encode(), "mkopo_watchlist.csv", "text/csv")
        st.caption("Scores cover the whole portfolio, including customers the model was trained on. Use the Model and governance page for held-out performance.")


def page_governance() -> None:
    page_header("Model and governance", "How the model performs on customers it has not seen, what drives it, how far to trust it, and what the data assumes.")
    tm = META["test_metrics"]
    cm = REP["confusion_matrix"]
    recall_high = cm.iloc[2, 2] / cm.iloc[2].sum()
    kpi_strip([
        ("Macro F1 (test)", f"{tm['macro_f1']:.2f}", "Balanced across the three tiers"),
        ("Accuracy (test)", pct(tm["accuracy"]), f"{META['test_size']:,} unseen customers"),
        ("High-risk recall", pct(recall_high, 0), "High customers correctly flagged"),
        ("ROC AUC", f"{tm['roc_auc_ovr']:.2f}", "One-vs-rest average"),
        ("Model", META["model_name"], f"Cross-validated F1 {META['cv_macro_f1']:.2f}"),
    ])
    t1, t2, t3, t4, t5 = st.tabs(["Performance", "Drivers", "Fairness", "Data and assumptions", "Model card"])

    with t1:
        a, b = st.columns(2)
        with a, card():
            st.markdown("#### Model comparison")
            mc = REP["model_comparison"].sort_values("CV macro F1")
            fig = go.Figure()
            fig.add_bar(y=mc["Model"], x=mc["CV macro F1"], name="Cross-validated macro F1", orientation="h", marker_color=LEAF)
            fig.add_bar(y=mc["Model"], x=mc["Test macro F1"], name="Test macro F1 (argmax)", orientation="h", marker_color="#9FB8AC")
            fig.update_layout(barmode="group", xaxis=dict(range=[0, 1], title="Macro F1"))
            st.plotly_chart(style(fig, 320), width="stretch")
            st.caption(f"Selection rule: {META['selection_rule']} The test set was used once, after selection.")
        with b, card():
            st.markdown("#### Confusion matrix (deployed decision rule)")
            norm = cm.values / cm.values.sum(axis=1, keepdims=True)
            fig = go.Figure(go.Heatmap(z=norm, x=TIERS, y=TIERS, colorscale=[[0, "#F3F6F4"], [1, LEAF]], zmin=0, zmax=1, showscale=False,
                                       text=[[f"{cm.values[i, j]:,}<br>{100 * norm[i, j]:.0f}%" for j in range(3)] for i in range(3)], texttemplate="%{text}",
                                       hovertemplate="Actual %{y}, predicted %{x}<extra></extra>"))
            fig.update_layout(xaxis_title="Predicted tier", yaxis_title="Actual risk class")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 320, legend=False), width="stretch")
        with card():
            st.markdown("#### Per-class results")
            pc = REP["per_class_report"].loc[TIERS, ["precision", "recall", "f1-score", "support"]].round(3)
            st.dataframe(pc, width="stretch")
            st.caption("Medium is the hardest class to separate from its neighbours. Expect more false alarms there than in High.")

    with t2:
        fi = REP["feature_importance"].copy()
        a, b = st.columns(2)
        with a, card():
            st.markdown("#### Permutation importance")
            st.caption("How much macro F1 falls when a field is shuffled on unseen customers.")
            d = fi.sort_values("permutation_importance")
            fig = go.Figure(go.Bar(x=d["permutation_importance"], y=d["label"], orientation="h", marker_color=LEAF, error_x=dict(type="data", array=d["permutation_std"], color=MUTED)))
            st.plotly_chart(style(fig, 360, legend=False), width="stretch")
        with b, card():
            st.markdown("#### Average pull on the score")
            if "mean_abs_contribution" in fi.columns:
                st.caption("Average size of each field's contribution to a customer's score (exact TreeSHAP).")
                d = fi.sort_values("mean_abs_contribution")
                fig = go.Figure(go.Bar(x=d["mean_abs_contribution"], y=d["label"], orientation="h", marker_color=AMBER))
                st.plotly_chart(style(fig, 360, legend=False), width="stretch")
            else:
                st.info("Contribution analysis is available when the deployed model is XGBoost.")

    with t3:
        a, b = st.columns(2)
        with a, card():
            st.markdown("#### Do the probabilities mean what they say?")
            rel = REP["reliability"]
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="Perfect")
            fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers", line=dict(color=LEAF, width=3), name="Model", marker=dict(size=np.clip(rel["customers"] ** 0.5, 6, 18)),
                            hovertemplate="Predicted %{x:.0%}, observed %{y:.0%}<extra></extra>")
            fig.update_layout(xaxis=dict(title="Predicted risk score", tickformat=".0%"), yaxis=dict(title="Observed share Medium or High", tickformat=".0%"))
            st.plotly_chart(style(fig, 320), width="stretch")
            st.dataframe(REP["calibration_metrics"], hide_index=True, width="stretch")
        with b, card():
            st.markdown("#### Group check at the deployed cutoffs")
            fa = REP["fairness"].copy()
            fa["group"] = fa["group"].astype(str)
            fa = fa.rename(columns={"dimension": "Dimension", "group": "Group", "customers": "Customers", "actual_non_low_rate": "Actually Medium/High",
                                    "flagged_rate": "Flagged", "recall": "Recall", "false_positive_rate": "False alarm rate"})
            st.dataframe(fa, hide_index=True, height=360, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.0f%%") for c in []})
            st.caption("Large gaps in the false alarm rate mean one group is declined more often without being riskier. Review before any real use.")

    with t4:
        cal = META["calibration"]
        callout("<b>What the data is.</b> Incomes were remapped to a Kenyan distribution, keeping each customer's rank within their employment group. ", warn=True)
        st.markdown("#### Monthly income assumptions by employment group")
        rows = []
        for g, v in cal["targets"].items():
            lo, hi = np.exp(np.log(v["median"]) - 1.2816 * v["sigma"]), np.exp(np.log(v["median"]) + 1.2816 * v["sigma"])
            rows.append({"Group": g, "Median (KSh)": round(v["median"]), "10th percentile": round(lo), "90th percentile": round(hi), "Floor": round(v["floor"]), "Basis": v["basis"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(f"Debt and purchase values were converted from USD at KSh {cal['fx_usd_kes']} and multiplied by {cal['money_scale_factor']:.4f}, so debt relative to income is unchanged from the source data.")
        st.markdown("#### Kenyan reference points")
        st.markdown(
            "- **Statutory minimum wage (from 1 May 2026):** KSh 18,047 a month for a general labourer in the five major cities, KSh 9,628 elsewhere. Regulation of Wages (General) (Amendment) Order, 2026.\n"
            "- **Average formal earnings, 2025:** about KSh 82,350 a month (KSh 988,200 a year). KNBS Economic Survey 2026.\n"
            "- **Formal pay distribution:** about 42% of formal employees earn under KSh 50,000 a month, and about 12% earn KSh 100,000 or more. KNBS Statistical Abstract 2025.\n"
            "- **Coverage gap:** KNBS publishes no median and no informal-sector wages, so freelancer, student and unemployed ranges are assumptions."
        )
        if "sensitivity" in REP:
            st.markdown("#### Does the result depend on these assumptions?")
            st.dataframe(REP["sensitivity"], hide_index=True, width="stretch")
            st.caption("Model macro F1 under alternative income assumptions. Stable scores mean the ranking the model learns does not hinge on the guesses; the income levels still matter for credibility.")

    with t5:
        st.markdown(f"""
**Intended use.** Decision support for BNPL credit analysts: screening applicants, setting starter limits and prioritising accounts for review. A person makes the decision.

**Not intended for.** Fully automated approvals or declines, regulatory capital calculations, or use on customers or markets unlike the training data.

**Model.** {META['model_name']} with calibrated probabilities (vector scaling on out-of-fold predictions). Tier cutoffs: High if P(High) is at least {TH['high']:.2f}; otherwise Medium if P(Medium) + P(High) is at least {TH['medium']:.2f}; otherwise Low.

**Data.** {META['train_size'] + META['test_size']:,} synthetic customers, split {META['train_size']:,} for training and {META['test_size']:,} for testing, stratified by risk class. Class balance: {', '.join(f"{k} {v:,}" for k, v in META['class_balance'].items())}.

**Performance (held-out).** Macro F1 {tm['macro_f1']:.2f}, accuracy {pct(tm['accuracy'])}, macro recall {pct(tm['macro_recall'])}, ROC AUC {tm['roc_auc_ovr']:.2f}.

**Known limitations.**
- The risk classes are labels, not observed repayment outcomes, and were generated on the source dataset's original scale.
- Freelancer, student and unemployed income distributions and the business-impact economics are assumptions.
- Medium-tier precision is lower than for the other tiers.

""")


PAGE_FUNCS = {
    "Overview": page_overview, "Risk scorecard": page_scorecard, "Batch scoring": page_batch,
    "Business impact": page_business, "Portfolio explorer": page_portfolio, "Model and governance": page_governance,
}
PAGE_FUNCS[page]()


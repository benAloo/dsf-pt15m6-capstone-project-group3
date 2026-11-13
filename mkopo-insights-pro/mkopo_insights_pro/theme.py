from __future__ import annotations

import base64
import functools
from pathlib import Path

import streamlit as st

import core

ASSETS = Path(__file__).parent / "assets" / "fonts"

INK, MUTED, LINE, CANVAS = "#0F1F19", "#5C6F66", "#E1E8E4", "#F4F7F5"
FOREST, EMERALD, MINT, GOLD = "#0B2A21", "#0F8A5F", "#E6F4EC", "#E0A528"
GREEN, AMBER, RED = core.TIER_COLORS["Low"], core.TIER_COLORS["Medium"], core.TIER_COLORS["High"]

BODY_FONT = "'Nunito', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
DISPLAY_FONT = BODY_FONT


@functools.lru_cache(maxsize=1)
def _font_faces() -> str:
    """Embed the self-hosted Nunito variable font, falling back to Google Fonts if the file is missing."""
    path = ASSETS / "nunito-latin-wght-normal.woff2"
    if not path.exists():
        return "@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f"@font-face{{font-family:'Nunito';src:url(data:font/woff2;base64,{b64}) format('woff2');"
        "font-weight:200 1000;font-style:normal;font-display:swap;}"
    )


CSS = f"""
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --canvas:{CANVAS}; --forest:{FOREST};
        --emerald:{EMERALD}; --mint:{MINT}; --gold:{GOLD}; }}
.stApp {{ background: var(--canvas); color: var(--ink); }}
.stApp, .stApp :where(p, label, li, td, th, a, input, textarea, button, div, span):not([data-testid="stIconMaterial"]) {{
  font-family: {BODY_FONT}; }}
.stApp :where(h1, h2, h3, h4, h5) {{ font-family: {DISPLAY_FONT}; color: var(--ink); letter-spacing: -0.01em; font-weight: 800; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1620px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: var(--forest); border-right: none; }}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{ color: #D5E7DD; }}
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  justify-content: flex-start; gap: .55rem; border: none; border-radius: 10px; padding: .62rem .85rem; box-shadow: none;
  background: transparent; color: #CFE3D9; font-weight: 700; }}
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{ background: #143C30; color: #fff; }}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  background: #174536; color: #fff; box-shadow: inset 3px 0 0 var(--gold); }}
section[data-testid="stSidebar"] button {{ justify-content: flex-start !important; }}
section[data-testid="stSidebar"] button > div, section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] {{
  width: 100%; justify-content: flex-start !important; text-align: left !important; }}
section[data-testid="stSidebar"] button p {{ color: inherit; text-align: left !important; font-weight: 700; font-size: .95rem; }}
.stApp .brand-name, .stApp .logo, .stApp .kv, .stApp .badge, .stApp .login-title {{ font-family: {DISPLAY_FONT}; }}
.brand {{ display: flex; align-items: center; gap: 12px; margin: 4px 0 4px 2px; }}
.logo {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: #fff;
  font-weight: 900; font-size: 1.25rem; background: var(--emerald); box-shadow: inset 0 -3px 0 var(--gold); }}
.brand-name {{ color: #fff; font-weight: 800; font-size: 1.25rem; line-height: 1.1; }}
.brand-sub {{ color: #9FBFB0; font-size: .78rem; font-weight: 600; margin: 2px 0 0 0; }}
.side-label {{ color: #8FB3A2; font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin: 18px 0 6px 4px; }}
.side-card {{ background: #12382C; border: 1px solid #1D4D3D; border-radius: 12px;
  padding: 12px 14px; color: #CFE3D9; font-size: .82rem; font-weight: 500; line-height: 1.5; margin-top: 10px; }}
.side-card b {{ color: #fff; font-weight: 800; }}

/* Page hero */
.hero {{ border-radius: 20px; padding: 26px 30px 24px; margin-bottom: 18px; color: #fff;
  background: var(--forest); border-bottom: 4px solid var(--gold); }}
.hero h1 {{ color: #fff !important; font-size: 2.05rem; font-weight: 900; margin: 0 0 6px 0; padding: 0; line-height: 1.15; }}
.hero p {{ color: #CFE3D9; margin: 0; max-width: 780px; font-size: 1rem; font-weight: 500; line-height: 1.55; }}
.hero .tag {{ display: inline-block; font-size: .72rem; font-weight: 800; letter-spacing: .06em; padding: 4px 12px;
  border-radius: 999px; background: var(--emerald); color: #fff; margin-bottom: 12px; }}

/* KPI cards */
.kgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 158px), 1fr)); gap: 14px; margin: 4px 0 18px 0; }}
.kcard {{ position: relative; background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 18px 18px 14px;
  box-shadow: 0 1px 2px rgba(15,31,25,.04), 0 8px 22px rgba(15,31,25,.05); }}
.kcard:before {{ content: ""; position: absolute; left: 18px; top: 0; width: 40px; height: 4px; border-radius: 0 0 4px 4px; background: var(--accent, var(--emerald)); }}
.kv {{ font-weight: 800; font-size: 1.65rem; line-height: 1.15; color: var(--ink); margin-top: 6px; font-variant-numeric: tabular-nums; }}
.kl {{ font-size: .86rem; font-weight: 800; color: var(--ink); margin-top: 3px; }}
.kn {{ font-size: .8rem; font-weight: 500; color: var(--muted); margin-top: 2px; line-height: 1.4; }}

/* Cards, widgets */
[class*="st-key-card"] {{ background: #fff; border-color: var(--line) !important; border-radius: 16px !important;
  box-shadow: 0 1px 2px rgba(15,31,25,.04), 0 8px 22px rgba(15,31,25,.05); }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; }}
[data-testid="stMetricValue"] {{ font-family: {DISPLAY_FONT}; font-weight: 800; }}
[data-testid="stMetricLabel"] p {{ color: var(--muted); font-weight: 700; }}
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"] {{ background: var(--emerald); border: 1px solid var(--emerald); color: #fff;
  border-radius: 10px; font-weight: 800; box-shadow: none; }}
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover {{ background: #0B6B4A; border-color: #0B6B4A; color: #fff; }}
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="stBaseButton-secondaryFormSubmit"] {{
  background: #fff; border: 1px solid var(--line); border-radius: 10px; font-weight: 700; color: var(--ink); }}
.stApp [data-testid="stBaseButton-secondary"]:hover, .stApp [data-testid="stBaseButton-secondaryFormSubmit"]:hover {{ border-color: var(--emerald); color: var(--emerald); }}
.stApp div[data-baseweb="input"], .stApp div[data-baseweb="input"] > div, .stApp div[data-baseweb="base-input"], .stApp [data-testid="stNumberInputContainer"], .stApp div[data-baseweb="select"] > div, .stApp div[data-baseweb="select"] > div > div, .stApp div[data-baseweb="textarea"] {{
  border-radius: 10px; background-color: #fff !important; border-color: #D3DDD7; }}
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] {{ display: none !important; }}
.stApp [data-testid="stFileUploaderDropzone"] {{ border-radius: 14px; background: #fff; border: 1.5px dashed #9DBBAD; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-weight: 800; color: var(--muted); padding: 10px 14px; }}
.stTabs [aria-selected="true"] {{ color: var(--emerald); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--emerald); height: 3px; border-radius: 3px; }}
[data-testid="stExpander"] {{ border-radius: 14px; border-color: var(--line); background: #fff; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
[data-testid="stCaptionContainer"] {{ color: var(--muted); font-weight: 500; }}

.callout {{ background: #fff; border: 1px solid var(--line); border-left: 5px solid var(--emerald); border-radius: 12px;
  padding: 13px 18px; margin: 8px 0 16px 0; color: var(--ink); font-size: .96rem; font-weight: 500; line-height: 1.55; }}
.callout.warn {{ border-left-color: var(--gold); background: #FFFAEE; }}
.badge {{ display: inline-block; padding: 6px 16px; border-radius: 999px; color: #fff; font-weight: 800;
  font-size: .95rem; letter-spacing: .01em; }}
.small {{ color: var(--muted); font-size: .85rem; font-weight: 600; }}
h4 {{ font-size: 1.08rem !important; margin-bottom: .2rem !important; }}

/* Sign-in */
.login-hero {{ border-radius: 24px; padding: 40px 42px; color: #fff; min-height: 610px; display: flex; flex-direction: column; justify-content: space-between;
  background: var(--forest); border-bottom: 5px solid var(--gold); box-shadow: 0 20px 50px rgba(11,42,33,.22); }}
.login-hero h2 {{ color: #fff !important; font-size: 2.6rem; font-weight: 900; line-height: 1.1; margin: 28px 0 14px 0; }}
.login-hero .lead {{ color: #CFE3D9; font-size: 1.06rem; font-weight: 500; line-height: 1.6; max-width: 460px; }}
.login-hero ul {{ list-style: none; padding: 0; margin: 26px 0 0 0; }}
.login-hero li {{ color: #E6F2EB; padding: 9px 0 9px 30px; position: relative; font-size: 1rem; font-weight: 700; }}
.login-hero li:before {{ content: ""; position: absolute; left: 0; top: 14px; width: 14px; height: 14px; border-radius: 50%;
  background: var(--gold); }}
.login-hero .foot {{ color: #9FBFB0; font-size: .8rem; font-weight: 700; }}
.login-title {{ font-size: 1.75rem; font-weight: 900; color: var(--ink); margin: 0 0 4px 0; }}
.login-sub {{ color: var(--muted); font-size: .96rem; font-weight: 500; margin: 0 0 14px 0; }}
"""

LOGIN_ONLY = """
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
"""


def inject(login: bool = False) -> None:
    css = _font_faces() + CSS + (LOGIN_ONLY if login else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, sub: str, tag: str = "Mkopo Insights") -> None:
    st.markdown(f"<div class='hero'><span class='tag'>{tag}</span><h1>{title}</h1><p>{sub}</p></div>", unsafe_allow_html=True)


def kpi_grid(items: list[tuple]) -> None:
    """items: (label, value, note) or (label, value, note, accent_colour)."""
    cells = []
    for it in items:
        label, value, note = it[0], it[1], it[2]
        accent = it[3] if len(it) > 3 else EMERALD
        cells.append(
            f"<div class='kcard' style='--accent:{accent}'><div class='kv'>{value}</div>"
            f"<div class='kl'>{label}</div><div class='kn'>{note}</div></div>"
        )
    st.markdown(f"<div class='kgrid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def callout(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='callout{' warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


def badge(tier: str) -> str:
    return f"<span class='badge' style='background:{core.TIER_COLORS[tier]}'>{tier} risk</span>"
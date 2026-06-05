
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Dashboard RH", page_icon="📊", layout="wide")

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"

PAGES = {
    "Executive": {"icon": "bi-speedometer2", "label": "Executive"},
    "Data Quality": {"icon": "bi-shield-check", "label": "Data Quality"},
    "RH Analytics": {"icon": "bi-people", "label": "RH Analytics"},
    "Forecast": {"icon": "bi-graph-up-arrow", "label": "Forecast"},
}

# =========================================================
# DATA
# =========================================================
@st.cache_data(show_spinner=False)
def read_csv_safe(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_csv_safe("gold_data_rh.csv", parse_dates=["mois"]),
        read_csv_safe("dashboard_data_quality.csv"),
        read_csv_safe("dashboard_executive.csv"),
        read_csv_safe("dashboard_rh_analytics.csv"),
        read_csv_safe("dashboard_forecast.csv", parse_dates=["mois"]),
        read_csv_safe("model_results.csv"),
    )


gold, quality, executive, analytics, forecast, model_results = load_data()

# =========================================================
# CSS
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #F5F7FB;
    --panel: #FFFFFF;
    --panel2: #F8FAFC;
    --navy: #06142D;
    --navy2: #0A1D3D;
    --text: #0F172A;
    --muted: #64748B;
    --line: #E5E7EB;
    --indigo: #4F46E5;
    --blue: #2563EB;
    --green: #16A34A;
    --orange: #F97316;
    --red: #DC2626;
}

/* Global */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
    overflow-x: hidden !important;
}

.stApp > header, footer, #MainMenu {
    visibility: hidden !important;
    height: 0 !important;
}

.block-container {
    max-width: 1600px !important;
    padding: 24px 34px 42px 34px !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 1rem !important;
}

div[data-testid="column"] {
    padding: 0 !important;
}

/* Sidebar: full dark, no inner white card */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06142D 0%, #081A35 55%, #030B1C 100%) !important;
    border-right: 1px solid rgba(255,255,255,.08) !important;
    width: 270px !important;
    min-width: 270px !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
    padding: 24px 18px !important;
}

section[data-testid="stSidebar"] * {
    font-family: Inter, "Segoe UI", sans-serif !important;
}

.sidebar-brand {
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,.10);
    margin-bottom: 20px;
}

.brand-logo {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(135deg, #4F46E5, #2563EB);
    box-shadow: 0 16px 34px rgba(37,99,235,.30);
    margin-bottom: 14px;
    font-size: 23px;
    font-weight: 900;
}

.brand-title {
    color: #fff;
    font-size: 22px;
    font-weight: 900;
    line-height: 1.1;
}

.brand-subtitle {
    color: #AFC2E8;
    font-size: 13px;
    font-weight: 600;
    margin-top: 6px;
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.sidebar-item {
    height: 48px;
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 0 14px;
    border-radius: 14px;
    color: #D8E5FF !important;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 800;
    box-sizing: border-box;
    border: 1px solid transparent;
    background: transparent;
}

.sidebar-item i {
    width: 20px;
    font-size: 18px;
    text-align: center;
    color: #EAF1FF;
}

.sidebar-item:hover {
    color: #FFFFFF !important;
    background: rgba(255,255,255,.075);
    border-color: rgba(255,255,255,.10);
    text-decoration: none !important;
}

.sidebar-item.active {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #4F46E5, #2563EB);
    box-shadow: 0 13px 30px rgba(37,99,235,.32);
}

.sidebar-footer {
    margin-top: 26px;
    padding: 16px;
    border-radius: 16px;
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.12);
}

.sidebar-footer-label {
    color: #91A9D5;
    font-size: 10px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.sidebar-footer-value {
    color: #FFFFFF;
    margin-top: 6px;
    font-size: 13px;
    font-weight: 800;
}

/* Compact filters row */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 18px;
    margin-bottom: 14px;
}

.topbar-title h1 {
    margin: 0;
    color: var(--text);
    font-size: 34px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.03em;
}

.topbar-title p {
    margin: 8px 0 0 0;
    color: var(--muted);
    font-size: 15px;
    font-weight: 600;
}

.aggregate-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-left: 14px;
    padding: 8px 12px;
    border-radius: 999px;
    background: #ECFDF5;
    color: #047857;
    font-size: 12px;
    font-weight: 900;
    vertical-align: middle;
}

/* Selectboxes: compact and no huge white block */
div[data-testid="stSelectbox"] label {
    color: #64748B !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 44px !important;
    border-radius: 14px !important;
    border: 1px solid #DDE3EE !important;
    background: #FFFFFF !important;
    box-shadow: 0 8px 22px rgba(15,23,42,.05) !important;
}

div[data-baseweb="popover"] {
    z-index: 999999 !important;
}

/* Hero */
.hero {
    margin-top: 4px;
    margin-bottom: 22px;
    padding: 24px 26px;
    border-radius: 24px;
    color: #FFFFFF;
    background:
      radial-gradient(circle at 8% 12%, rgba(79,70,229,.38), transparent 28%),
      linear-gradient(135deg, #101D54 0%, #07152E 52%, #0B2146 100%);
    box-shadow: 0 22px 50px rgba(15,23,42,.18);
    border: 1px solid rgba(255,255,255,.10);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
}

.hero-left h2 {
    margin: 0;
    font-size: 31px;
    font-weight: 900;
    letter-spacing: -0.02em;
}

.hero-left p {
    margin: 8px 0 0 0;
    color: #D8E5FF;
    font-size: 15px;
    font-weight: 600;
}

.hero-mini {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.hero-chip {
    min-width: 128px;
    padding: 13px 15px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.16);
    background: rgba(255,255,255,.08);
}

.hero-chip span {
    display: block;
    color: #BED3FB;
    font-size: 10px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.hero-chip strong {
    display: block;
    margin-top: 4px;
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 900;
}

/* Sections */
.section-head {
    display: flex;
    align-items: center;
    gap: 13px;
    margin: 8px 0 15px 0;
}

.section-icon {
    width: 40px;
    height: 40px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    color: #4F46E5;
    background: linear-gradient(135deg, rgba(79,70,229,.12), rgba(37,99,235,.14));
    font-size: 20px;
}

.section-head h2 {
    margin: 0;
    font-size: 25px;
    font-weight: 900;
    line-height: 1.05;
    color: var(--text);
    letter-spacing: -0.02em;
}

.section-head p {
    margin: 5px 0 0 0;
    color: var(--muted);
    font-size: 14px;
    font-weight: 600;
}

/* KPI cards */
.kpi-card {
    height: 146px;
    padding: 18px;
    border-radius: 20px;
    border: 1px solid var(--line);
    background: #FFFFFF;
    box-shadow: 0 10px 28px rgba(15,23,42,.075);
    overflow: hidden;
    position: relative;
}

.kpi-card::after {
    content: "";
    position: absolute;
    right: -28px;
    bottom: -42px;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(79,70,229,.08), rgba(37,99,235,.04));
}

.kpi-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    position: relative;
    z-index: 1;
}

.kpi-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .06em;
    line-height: 1.35;
}

.kpi-icon {
    flex: 0 0 42px;
    width: 42px;
    height: 42px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    color: #fff;
    font-size: 18px;
    background: linear-gradient(135deg, #4F46E5, #2563EB);
    box-shadow: 0 14px 26px rgba(79,70,229,.25);
}

.kpi-value {
    margin-top: 13px;
    color: #0F172A;
    font-size: 30px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 1;
}

.kpi-subtext {
    margin-top: 8px;
    color: #64748B;
    font-size: 12px;
    font-weight: 700;
    position: relative;
    z-index: 1;
}

.kpi-delta {
    margin-top: 6px;
    font-size: 11px;
    font-weight: 900;
    position: relative;
    z-index: 1;
}

.positive { color: #16A34A; }
.warning { color: #F97316; }
.negative { color: #DC2626; }
.neutral { color: #2563EB; }

/* Streamlit bordered containers used as chart cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 1px solid var(--line) !important;
    background: #FFFFFF !important;
    box-shadow: 0 10px 28px rgba(15,23,42,.075) !important;
    overflow: hidden !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 18px 18px 14px 18px !important;
}

.card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.card-title i {
    color: #4F46E5;
    font-size: 18px;
}

.card-title h3 {
    margin: 0;
    color: #0F172A;
    font-size: 16px;
    font-weight: 900;
}

.card-subtitle {
    margin: 0 0 8px 28px;
    color: #64748B;
    font-size: 12px;
    font-weight: 650;
}

.js-plotly-plot, .plotly {
    border-radius: 14px !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}

.governance-card {
    margin-top: 2px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 20px 22px;
    border-radius: 20px;
    background: linear-gradient(135deg, #07152E, #0B2146);
    color: #FFFFFF;
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 14px 32px rgba(15,23,42,.16);
}

.governance-card i {
    color: #93C5FD;
    font-size: 22px;
}

.governance-card strong {
    display: block;
    font-size: 16px;
    margin-bottom: 5px;
}

.governance-card span {
    color: #D8E5FF;
    font-size: 14px;
    line-height: 1.45;
}

/* Plotly modebar clean */
.modebar { display: none !important; }

@media (max-width: 1200px) {
    .block-container { padding: 18px !important; }
    .hero { display: block; }
    .hero-mini { justify-content: flex-start; margin-top: 16px; }
    .kpi-value { font-size: 25px; }
}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# HELPERS
# =========================================================
def get_page() -> str:
    page = st.query_params.get("page", "Executive")
    return page if page in PAGES else "Executive"


def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return not df.empty and all(c in df.columns for c in cols)


def fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}".replace(",", " ")


def fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def fmt_num(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def delta_text(current, previous, percent: bool = False, lower_is_better: bool = False) -> tuple[str, str]:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous) or previous == 0:
        return "Référence indisponible", "neutral"
    delta = float(current) - float(previous)
    txt = f"{delta:+.1%} vs période précédente" if percent else f"{delta:+,.0f} vs période précédente".replace(",", " ")
    good = delta >= 0
    if lower_is_better:
        good = not good
    return txt, "positive" if good else "negative"


def style_fig(fig: go.Figure, height: int = 310, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#0F172A"),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)") if legend else None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#64748B"))
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, tickfont=dict(color="#64748B"))
    return fig


def latest_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not has_cols(df, ["mois"]):
        return pd.DataFrame(), pd.DataFrame()
    latest_month = df["mois"].max()
    previous_month = df.loc[df["mois"] < latest_month, "mois"].max()
    latest = df[df["mois"] == latest_month].copy()
    previous = df[df["mois"] == previous_month].copy() if pd.notna(previous_month) else pd.DataFrame()
    return latest, previous


def kpi_card(icon: str, label: str, value: str, subtext: str, delta: str, delta_class: str = "neutral") -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-top">
    <div class="kpi-label">{label}</div>
    <div class="kpi-icon"><i class="bi {icon}"></i></div>
  </div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-subtext">{subtext}</div>
  <div class="kpi-delta {delta_class}">{delta}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_title(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="section-head">
  <div class="section-icon"><i class="bi {icon}"></i></div>
  <div>
    <h2>{title}</h2>
    <p>{subtitle}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def chart_card(title: str, subtitle: str, icon: str, fig: go.Figure | None, height: int = 310, legend: bool = True) -> None:
    if fig is None:
        return
    with st.container(border=True):
        st.markdown(
            f"""
<div class="card-title"><i class="bi {icon}"></i><h3>{title}</h3></div>
<div class="card-subtitle">{subtitle}</div>
""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(style_fig(fig, height=height, legend=legend), use_container_width=True, config={"displayModeBar": False, "responsive": True})


def table_card(title: str, subtitle: str, icon: str, df: pd.DataFrame, height: int = 260) -> None:
    if df.empty:
        return
    with st.container(border=True):
        st.markdown(
            f"""
<div class="card-title"><i class="bi {icon}"></i><h3>{title}</h3></div>
<div class="card-subtitle">{subtitle}</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def render_governance(title: str, text: str) -> None:
    st.markdown(
        f"""
<div class="governance-card">
  <i class="bi bi-info-circle"></i>
  <div><strong>{title}</strong><span>{text}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# NAVIGATION + HEADER
# =========================================================
def render_sidebar(page: str, latest_month) -> None:
    links = []
    for name, meta in PAGES.items():
        active = " active" if name == page else ""
        links.append(
            f"""
<a class="sidebar-item{active}" href="?page={quote(name)}" target="_self">
  <i class="bi {meta['icon']}"></i>
  <span>{meta['label']}</span>
</a>
"""
        )

    date = latest_month.strftime("%d/%m/%Y") if latest_month is not None and pd.notna(latest_month) else "Non disponible"

    st.sidebar.markdown(
        f"""
<div class="sidebar-brand">
  <div class="brand-logo">RH</div>
  <div class="brand-title">Dashboard RH</div>
  <div class="brand-subtitle">Pilotage RH agrégé</div>
</div>

<nav class="sidebar-nav">
  {''.join(links)}
</nav>

<div class="sidebar-footer">
  <div class="sidebar-footer-label">Dernière mise à jour</div>
  <div class="sidebar-footer-value">{date}</div>
  <div class="sidebar-footer-label" style="margin-top:14px;">Usage</div>
  <div class="sidebar-footer-value">Données agrégées uniquement</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_filters(df: pd.DataFrame) -> tuple[str, str, str]:
    sites = ["Tous"] + sorted(df["site"].dropna().astype(str).unique().tolist()) if "site" in df.columns else ["Tous"]

    left, f1, f2, f3 = st.columns([2.7, 1, 1, 1], gap="medium")
    with left:
        st.markdown(
            """
<div class="topbar-title">
  <h1>Dashboard RH <span class="aggregate-pill"><i class="bi bi-shield-check"></i> Données agrégées uniquement</span></h1>
  <p>Pilotage & Analyse des Ressources Humaines</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with f1:
        trimestre = st.selectbox("Trimestre", ["T2 2026", "T1 2026", "T4 2025", "T3 2025", "Tous"])
    with f2:
        perimetre = st.selectbox("Périmètre", ["Tous", "Sites", "Métiers", "Équipes"])
    with f3:
        site = st.selectbox("Site", sites)

    return trimestre, perimetre, site


def render_hero(trimestre: str, perimetre: str, site: str, latest_month) -> None:
    ref = latest_month.strftime("%Y-%m-%d") if latest_month is not None and pd.notna(latest_month) else "N/A"
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-left">
    <h2><i class="bi bi-bar-chart-line" style="color:#93C5FD;margin-right:10px;"></i>Dashboard RH</h2>
    <p>Pilotage des dynamiques RH collectives, qualité des données et prévision agrégée.</p>
  </div>
  <div class="hero-mini">
    <div class="hero-chip"><span>Trimestre</span><strong>{trimestre}</strong></div>
    <div class="hero-chip"><span>Périmètre</span><strong>{perimetre}</strong></div>
    <div class="hero-chip"><span>Site</span><strong>{site}</strong></div>
    <div class="hero-chip"><span>Référence</span><strong>{ref}</strong></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# PAGES
# =========================================================
def render_executive(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    section_title("bi-speedometer2", "Vue Executive", "Synthèse direction des indicateurs RH collectifs")

    latest, previous = latest_frames(df)
    if latest.empty:
        st.error("DATA GOLD indisponible ou colonne `mois` manquante.")
        return

    total_real = latest.get("effectif_reel", pd.Series(dtype=float)).sum()
    total_plan = latest.get("effectif_planifie", pd.Series(dtype=float)).sum()
    total_gap = latest.get("ecart_effectif", pd.Series(dtype=float)).sum()
    total_depart = latest.get("departs_volontaires", pd.Series(dtype=float)).sum()
    attrition = total_depart / max(total_real, 1)
    tension = latest.get("tension_recrutement", pd.Series(dtype=float)).mean()

    prev_real = previous.get("effectif_reel", pd.Series(dtype=float)).sum() if not previous.empty else 0
    prev_plan = previous.get("effectif_planifie", pd.Series(dtype=float)).sum() if not previous.empty else 0
    prev_gap = previous.get("ecart_effectif", pd.Series(dtype=float)).sum() if not previous.empty else 0
    prev_depart = previous.get("departs_volontaires", pd.Series(dtype=float)).sum() if not previous.empty else 0
    prev_attrition = prev_depart / max(prev_real, 1) if prev_real else 0
    prev_tension = previous.get("tension_recrutement", pd.Series(dtype=float)).mean() if not previous.empty else 0

    cards = [
        ("bi-people-fill", "Effectif réel", fmt_int(total_real), "Population observée", *delta_text(total_real, prev_real)),
        ("bi-calendar2-check", "Effectif planifié", fmt_int(total_plan), "Cible workforce planning", *delta_text(total_plan, prev_plan)),
        ("bi-diagram-3", "Écart effectif", fmt_int(total_gap), "Réel moins planifié", *delta_text(total_gap, prev_gap)),
        ("bi-arrow-repeat", "Taux attrition", fmt_pct(attrition), "Départs / effectif", *delta_text(attrition, prev_attrition, percent=True, lower_is_better=True)),
        ("bi-box-arrow-right", "Départs volontaires", fmt_int(total_depart), "Volume agrégé", *delta_text(total_depart, prev_depart, lower_is_better=True)),
        ("bi-activity", "Tension globale", fmt_num(tension), "Indice recrutement", *delta_text(tension, prev_tension, percent=False, lower_is_better=True)),
    ]

    cols = st.columns(6, gap="small")
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    if has_cols(df, ["mois", "effectif_reel", "effectif_planifie"]):
        monthly = df.groupby("mois", as_index=False).agg(
            effectif_reel=("effectif_reel", "sum"),
            effectif_planifie=("effectif_planifie", "sum"),
            departs_volontaires=("departs_volontaires", "sum"),
            recrutements_ouverts=("recrutements_ouverts", "sum") if "recrutements_ouverts" in df.columns else ("effectif_reel", "count"),
        )
        monthly["ecart_effectif"] = monthly["effectif_reel"] - monthly["effectif_planifie"]
    else:
        monthly = pd.DataFrame()

    c1, c2 = st.columns([1.35, 1], gap="medium")

    with c1:
        fig = None
        if not monthly.empty:
            fig = go.Figure()
            fig.add_bar(x=monthly["mois"], y=monthly["effectif_reel"], name="Réel", marker_color="#4F46E5")
            fig.add_bar(x=monthly["mois"], y=monthly["effectif_planifie"], name="Planifié", marker_color="#C4B5FD")
            fig.add_scatter(x=monthly["mois"], y=monthly["ecart_effectif"], name="Écart", mode="lines+markers", yaxis="y2", line=dict(color="#2563EB", width=3))
            fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", showgrid=False))
        chart_card("Effectif réel vs planifié", "Barres mensuelles et ligne d'écart", "bi-bar-chart", fig, 330)

    with c2:
        fig = None
        if has_cols(latest, ["site", "tension_recrutement"]):
            tension_site = latest.groupby("site", as_index=False)["tension_recrutement"].mean().sort_values("tension_recrutement")
            fig = px.bar(tension_site, x="tension_recrutement", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")
        chart_card("Tension par site", "Indice moyen de tension recrutement", "bi-geo-alt", fig, 330, legend=False)

    c3, c4, c5 = st.columns(3, gap="medium")

    with c3:
        fig = None
        if has_cols(forecast_df, ["mois", "reel", "prediction_retenue"]):
            fc = forecast_df.groupby("mois", as_index=False).agg(reel=("reel", "sum"), prediction_retenue=("prediction_retenue", "sum")).tail(12)
            fig = px.line(fc, x="mois", y=["reel", "prediction_retenue"], markers=True, color_discrete_sequence=["#4F46E5", "#F97316"])
        elif not monthly.empty:
            fig = px.bar(monthly.tail(12), x="mois", y="departs_volontaires", color_discrete_sequence=["#4F46E5"])
        chart_card("Départs volontaires", "Réel vs prévision agrégée", "bi-graph-up", fig, 285)

    with c4:
        fig = None
        if has_cols(latest, ["equipe", "taux_attrition"]):
            top_attr = latest.groupby("equipe", as_index=False)["taux_attrition"].mean().sort_values("taux_attrition", ascending=False).head(5)
            fig = px.bar(top_attr, x="taux_attrition", y="equipe", orientation="h", color="taux_attrition", color_continuous_scale="OrRd")
        chart_card("Top 5 équipes attrition", "Collectifs à surveiller", "bi-exclamation-triangle", fig, 285, legend=False)

    with c5:
        fig = None
        if has_cols(latest, ["site", "ecart_effectif"]):
            gap_site = latest.groupby("site", as_index=False)["ecart_effectif"].sum().sort_values("ecart_effectif")
            fig = px.bar(gap_site, x="ecart_effectif", y="site", orientation="h", color="ecart_effectif", color_continuous_scale="RdYlGn")
        chart_card("Écart effectif par site", "Réel moins planifié", "bi-sliders", fig, 285, legend=False)

    c6, c7 = st.columns(2, gap="medium")

    with c6:
        fig = None
        if "couverture_competences_critiques" in latest.columns:
            coverage = latest["couverture_competences_critiques"].mean()
            fig = go.Figure(data=[go.Pie(
                labels=["Couvert", "À renforcer"],
                values=[coverage, max(0, 1 - coverage)],
                hole=.62,
                marker_colors=["#16A34A", "#F97316"],
            )])
            fig.update_layout(annotations=[dict(text=fmt_pct(coverage), x=.5, y=.5, showarrow=False, font=dict(size=28, color="#0F172A"))])
        chart_card("Couverture compétences critiques", "Niveau de couverture moyen", "bi-award", fig, 280)

    with c7:
        fig = None
        if not monthly.empty and "recrutements_ouverts" in monthly.columns:
            fig = px.line(monthly.tail(12), x="mois", y="recrutements_ouverts", markers=True, color_discrete_sequence=["#2563EB"])
        chart_card("Recrutements ouverts", "Postes ouverts sur 12 mois", "bi-briefcase", fig, 280, legend=False)

    render_governance(
        "Usage responsable",
        "Tous les indicateurs sont calculés au niveau agrégé : équipe, site et métier. Aucune décision individuelle ou automatisée n'est prévue.",
    )


def render_data_quality(df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    section_title("bi-shield-check", "Data Quality", "Contrôle des sources, anomalies détectées et règles de fiabilisation")

    if quality_df.empty:
        st.error("dashboard_data_quality.csv est indisponible.")
        return

    score = quality_df["score_qualite_estime"].mean() if "score_qualite_estime" in quality_df.columns else 0
    missing = quality_df["valeurs_manquantes"].sum() if "valeurs_manquantes" in quality_df.columns else 0
    duplicates = quality_df["doublons"].sum() if "doublons" in quality_df.columns else 0
    issues = quality_df["incoherences_detectees"].sum() if "incoherences_detectees" in quality_df.columns else 0

    cols = st.columns(5, gap="small")
    cards = [
        ("bi-patch-check", "Score qualité global", f"{score:.1f}/100", "Plus haut = plus fiable", "Contrôle qualité source", "positive"),
        ("bi-question-circle", "Valeurs manquantes", fmt_int(missing), "Contrôle exhaustivité", "À surveiller", "warning" if missing else "positive"),
        ("bi-copy", "Doublons", fmt_int(duplicates), "Contrôle unicité", "Doublons stricts", "warning" if duplicates else "positive"),
        ("bi-bug", "Incohérences", fmt_int(issues), "Règles métier", "Valeurs hors bornes", "negative" if issues else "positive"),
        ("bi-database-check", "Lignes GOLD DATA", fmt_int(len(df)), "Dataset final RH", "Table agrégée", "neutral"),
    ]

    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig = None
        if has_cols(quality_df, ["fichier", "valeurs_manquantes"]):
            q = quality_df.sort_values("valeurs_manquantes")
            fig = px.bar(q, x="valeurs_manquantes", y="fichier", orientation="h", color="valeurs_manquantes", color_continuous_scale="Blues")
        chart_card("Missing values par fichier", "Volume total de valeurs manquantes", "bi-list-check", fig, 250, legend=False)

    with c2:
        fig = None
        if has_cols(quality_df, ["fichier", "doublons"]):
            q = quality_df.sort_values("doublons")
            fig = px.bar(q, x="doublons", y="fichier", orientation="h", color="doublons", color_continuous_scale="Purples")
        chart_card("Doublons par fichier", "Lignes strictement dupliquées", "bi-files", fig, 250, legend=False)

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        fig = None
        if has_cols(quality_df, ["fichier", "score_qualite_estime"]):
            q = quality_df.sort_values("score_qualite_estime")
            fig = px.bar(q, x="score_qualite_estime", y="fichier", orientation="h", color="score_qualite_estime", color_continuous_scale="RdYlGn")
        chart_card("Score qualité par source", "Score estimé après contrôles", "bi-stars", fig, 250, legend=False)

    with c4:
        rules = quality_df[["fichier", "regles_appliquees"]].drop_duplicates() if has_cols(quality_df, ["fichier", "regles_appliquees"]) else pd.DataFrame()
        table_card("Règles qualité appliquées", "Principes de fiabilisation DATA GOLD", "bi-tools", rules, height=250)

    anomaly = quality_df.copy()
    if "incoherences_detectees" in anomaly.columns:
        anomaly["gravite"] = pd.cut(
            anomaly["incoherences_detectees"],
            bins=[-1, 0, 200, float("inf")],
            labels=["faible", "moyen", "élevé"],
        ).astype(str)
    table_card("Anomalies détectées", "Lecture par source et niveau de vigilance", "bi-clipboard-data", anomaly, height=300)


def render_rh_analytics(df: pd.DataFrame, analytics_df: pd.DataFrame) -> None:
    section_title("bi-people", "RH Analytics", "Dynamiques collectives par équipe, site et métier")

    latest, _ = latest_frames(df)
    if latest.empty:
        st.error("DATA GOLD indisponible.")
        return

    cols = st.columns(5, gap="small")
    cards = [
        ("bi-diagram-2", "Équipes analysées", fmt_int(latest["equipe"].nunique()) if "equipe" in latest.columns else "-", "Collectifs actifs", "Granularité équipe", "neutral"),
        ("bi-radioactive", "Risque moyen", fmt_num(latest["indicateur_risque_collectif"].mean()) if "indicateur_risque_collectif" in latest.columns else "-", "Indice collectif", "Composite RH", "warning"),
        ("bi-geo", "Sites en tension", fmt_int((latest.groupby("site")["tension_recrutement"].mean() > .5).sum()) if has_cols(latest, ["site", "tension_recrutement"]) else "-", "Tension > 0.50", "Recrutement", "warning"),
        ("bi-award", "Couverture skills", fmt_pct(latest["couverture_competences_critiques"].mean()) if "couverture_competences_critiques" in latest.columns else "-", "Moyenne collective", "Compétences critiques", "positive"),
        ("bi-calendar-heart", "Absentéisme moyen", fmt_pct(latest["absenteeism_rate"].mean()) if "absenteeism_rate" in latest.columns else "-", "Donnée agrégée", "Signal RH", "neutral"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig = None
        if has_cols(latest, ["equipe", "indicateur_risque_collectif"]):
            top_risk = latest.sort_values("indicateur_risque_collectif", ascending=False).head(12)
            fig = px.bar(top_risk, x="indicateur_risque_collectif", y="equipe", orientation="h", color="site" if "site" in latest.columns else None)
        chart_card("Classement équipes instables", "Top collectifs par risque agrégé", "bi-sort-down", fig, 330)

    with c2:
        fig = None
        if has_cols(latest, ["tension_recrutement", "taux_attrition", "effectif_reel"]):
            fig = px.scatter(
                latest,
                x="tension_recrutement",
                y="taux_attrition",
                size="effectif_reel",
                color="metier" if "metier" in latest.columns else None,
                hover_data=[c for c in ["site", "equipe"] if c in latest.columns],
            )
        chart_card("Profils d'équipes", "Tension recrutement x attrition", "bi-bounding-box", fig, 330)

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        fig = None
        if has_cols(latest, ["site", "metier", "indicateur_risque_collectif"]):
            heat = latest.pivot_table(index="site", columns="metier", values="indicateur_risque_collectif", aggfunc="mean")
            fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r", labels=dict(color="Risque"))
        chart_card("Heatmap site x métier", "Risque collectif moyen", "bi-grid-3x3-gap", fig, 315, legend=False)

    with c4:
        fig = None
        if has_cols(latest, ["site", "ecart_effectif", "tension_recrutement"]):
            site_gap = latest.groupby("site", as_index=False).agg(ecart_effectif=("ecart_effectif", "sum"), tension_recrutement=("tension_recrutement", "mean"))
            fig = px.bar(site_gap.sort_values("ecart_effectif"), x="ecart_effectif", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")
        chart_card("Écarts effectif réel vs planifié", "Sous-effectif et tension par site", "bi-bar-chart-steps", fig, 315, legend=False)

    table_card("Tableau analytique agrégé", "Aucun niveau individuel, uniquement équipe/site/métier", "bi-table", analytics_df if not analytics_df.empty else latest, height=330)


def render_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    section_title("bi-graph-up-arrow", "Forecast", "Prévision agrégée des départs volontaires")

    if model_df.empty:
        st.error("model_results.csv est indisponible.")
        return

    best = model_df.sort_values("MAE").iloc[0] if "MAE" in model_df.columns else model_df.iloc[0]
    predicted = forecast_df["prediction_retenue"].sum() if "prediction_retenue" in forecast_df.columns else 0

    cols = st.columns(5, gap="small")
    cards = [
        ("bi-box-arrow-up-right", "Départs prévus", fmt_int(predicted), "Volume agrégé", "Prévision collective", "neutral"),
        ("bi-bullseye", "Erreur MAE", fmt_num(best.get("MAE", 0)), "Plus faible = mieux", "Erreur moyenne", "positive"),
        ("bi-crosshair", "RMSE", fmt_num(best.get("RMSE", 0)), "Pénalise gros écarts", "Validation modèle", "positive"),
        ("bi-cpu", "Meilleur modèle", str(best.get("modele", "-")), "Sélection par MAE", "Benchmark", "neutral"),
        ("bi-shield-lock", "Confiance", "Contrôlée", "Usage encadré", "Limites documentées", "warning"),
    ]

    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    c1, c2 = st.columns([1.2, 1], gap="medium")

    with c1:
        fig = None
        if has_cols(forecast_df, ["mois", "reel", "prediction_retenue"]):
            fc = forecast_df.groupby("mois", as_index=False).agg(reel=("reel", "sum"), prediction_retenue=("prediction_retenue", "sum"))
            fig = px.line(fc, x="mois", y=["reel", "prediction_retenue"], markers=True, color_discrete_sequence=["#4F46E5", "#F97316"])
        chart_card("Réel vs prédit", "Volumes agrégés de départs volontaires", "bi-graph-up", fig, 330)

    with c2:
        fig = None
        if has_cols(model_df, ["modele", "MAE", "RMSE"]):
            fig = px.bar(model_df, x="modele", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=["#4F46E5", "#2563EB"])
        chart_card("Baseline vs modèles", "Comparaison des performances", "bi-columns-gap", fig, 330)

    fig = None
    cols_corr = ["tension_recrutement", "couverture_competences_critiques", "effectif_reel", "absenteeism_rate", "engagement_score_avg"]
    if has_cols(df, cols_corr + ["departs_volontaires"]):
        imp = df[cols_corr + ["departs_volontaires"]].corr(numeric_only=True)["departs_volontaires"].drop("departs_volontaires").abs().reset_index()
        imp.columns = ["variable", "importance_proxy"]
        fig = px.bar(imp.sort_values("importance_proxy"), x="importance_proxy", y="variable", orientation="h", color="importance_proxy", color_continuous_scale="Blues")
    chart_card("Facteurs explicatifs agrégés", "Proxy par corrélation absolue", "bi-node-plus", fig, 285, legend=False)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        table_card("Tableau model_results", "Scores de validation des modèles", "bi-table", model_df, height=300)
    with c4:
        render_governance(
            "Gouvernance du modèle",
            "Ces prévisions sont agrégées et ne doivent jamais être utilisées pour scorer ou classer des collaborateurs individuellement.",
        )


# =========================================================
# APP
# =========================================================
def main() -> None:
    inject_css()

    if gold.empty:
        st.error("Impossible de charger la DATA GOLD. Vérifie `data/processed/gold_data_rh.csv`.")
        return

    latest_month = gold["mois"].max() if "mois" in gold.columns else None
    page = get_page()

    render_sidebar(page, latest_month)
    trimestre, perimetre, site = render_filters(gold)

    view = gold.copy()
    if site != "Tous" and "site" in view.columns:
        view = view[view["site"].astype(str) == site].copy()

    render_hero(trimestre, perimetre, site, latest_month)

    if page == "Executive":
        render_executive(view, forecast)
    elif page == "Data Quality":
        render_data_quality(view, quality)
    elif page == "RH Analytics":
        render_rh_analytics(view, analytics)
    elif page == "Forecast":
        render_forecast(view, forecast, model_results)


if __name__ == "__main__":
    main()

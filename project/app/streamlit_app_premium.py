from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Dashboard RH", page_icon="RH", layout="wide")

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"

PAGES = {
    "Executive": {"icon": "bi-speedometer2", "label": "Executive"},
    "Data Quality": {"icon": "bi-shield-check", "label": "Data Quality"},
    "RH Analytics": {"icon": "bi-people", "label": "RH Analytics"},
    "Forecast": {"icon": "bi-graph-up-arrow", "label": "Forecast"},
}


@st.cache_data
def read_csv_safe(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()


@st.cache_data
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


def inject_css() -> None:
    st.html(
        """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #F5F7FB;
    --panel: #FFFFFF;
    --navy: #07152E;
    --navy-soft: #0B2146;
    --text: #0F172A;
    --muted: #64748B;
    --line: #E5E7EB;
    --indigo: #4F46E5;
    --blue: #2563EB;
    --green: #16A34A;
    --orange: #F97316;
    --red: #DC2626;
}

html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    overflow-x: hidden;
}

.stApp > header,
footer,
#MainMenu {
    display: none !important;
}

.block-container {
    max-width: 1540px;
    padding: 14px 22px 30px 22px;
    overflow-x: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07152E 0%, #081A33 58%, #041027 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
}

[data-testid="stSidebar"] > div {
    background: transparent !important;
    padding: 22px 16px !important;
}

[data-testid="stSidebar"] .stMarkdown {
    margin: 0;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    background: transparent !important;
}

.sidebar-brand {
    padding: 4px 4px 18px 4px;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    background: transparent;
}

.brand-logo {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, var(--indigo), var(--blue));
    color: #FFFFFF;
    box-shadow: 0 14px 32px rgba(37,99,235,0.28);
    margin-bottom: 14px;
}

.brand-logo i {
    font-size: 24px;
}

.brand-title {
    color: #FFFFFF;
    font-size: 21px;
    font-weight: 900;
    line-height: 1.1;
}

.brand-subtitle {
    color: #AFC2E8;
    font-size: 13px;
    font-weight: 600;
    margin-top: 5px;
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 12px;
}

.sidebar-item {
    height: 48px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 14px;
    border-radius: 14px;
    color: #E5EDFF !important;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 800;
    border: 1px solid transparent;
    background: rgba(255,255,255,0.035);
    transition: background .15s ease, border-color .15s ease, box-shadow .15s ease;
    box-sizing: border-box;
}

.sidebar-item i {
    width: 20px;
    text-align: center;
    font-size: 17px;
    color: #FFFFFF;
}

.sidebar-item:hover {
    color: #FFFFFF !important;
    background: rgba(255,255,255,0.085);
    border-color: rgba(255,255,255,0.12);
}

.sidebar-item.active {
    background: linear-gradient(135deg, var(--indigo), var(--blue));
    color: #FFFFFF !important;
    box-shadow: 0 12px 28px rgba(37,99,235,0.30);
}

.sidebar-footer {
    margin-top: 24px;
    padding: 15px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
}

.footer-label {
    color: #AFC2E8;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.footer-value {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 900;
    margin-top: 6px;
}

/* Header and filters */
.header-marker {
    display: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.header-marker) {
    background:
        radial-gradient(circle at 8% 24%, rgba(79,70,229,0.24), transparent 30%),
        linear-gradient(135deg, #07152E 0%, #0B2146 58%, #111827 100%) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 22px !important;
    box-shadow: 0 18px 46px rgba(15,23,42,0.18) !important;
    margin-bottom: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.header-marker) > div {
    padding: 18px 22px !important;
}

.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 0;
    margin: 0;
    border-radius: 0;
    color: #FFFFFF;
    background: transparent;
    box-shadow: none;
}

.hero-title {
    display: flex;
    align-items: center;
    gap: 14px;
}

.hero h1 {
    margin: 0;
    font-size: 31px;
    line-height: 1.05;
    font-weight: 900;
}

.hero p {
    margin: 7px 0 0 0;
    color: #D8E5FF;
    font-size: 15px;
    font-weight: 500;
}

.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.header-filter-label {
    color: #BFD7FF;
    font-size: 10px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 0 0 6px 2px;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.header-marker) div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 42px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.09);
    color: #FFFFFF;
    box-shadow: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.header-marker) div[data-baseweb="select"] span,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.header-marker) div[data-baseweb="select"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

.hero-badge {
    min-width: 136px;
    padding: 12px 15px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.08);
}

.hero-badge span {
    display: block;
    color: #BFDBFE;
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.hero-badge strong {
    display: block;
    margin-top: 4px;
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 900;
}

.filter-title {
    color: var(--muted);
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin: 2px 0 8px 2px;
}

/* Streamlit components */
div[data-testid="stSelectbox"] > div {
    min-height: 42px;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border-radius: 14px;
    border-color: #DDE3EE;
    background: #FFFFFF;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}

div[data-testid="stVerticalBlock"] {
    gap: 0.9rem;
}

div[data-testid="column"] {
    padding-top: 0 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--panel);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    overflow: hidden;
    max-width: 100%;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 16px 16px 12px 16px;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

.js-plotly-plot .plotly {
    border-radius: 14px;
}

/* Content atoms */
.section-head {
    display: flex;
    align-items: center;
    gap: 13px;
    margin: 12px 0 12px 0;
}

.section-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 15px;
    color: var(--indigo);
    background: linear-gradient(135deg, rgba(79,70,229,0.12), rgba(37,99,235,0.16));
}

.section-icon i {
    font-size: 20px;
}

.section-head h2 {
    margin: 0;
    color: var(--text);
    font-size: 23px;
    line-height: 1.08;
    font-weight: 900;
}

.section-head p {
    margin: 4px 0 0 0;
    color: var(--muted);
    font-size: 14px;
    font-weight: 500;
}

.kpi-card {
    height: 138px;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: var(--panel);
    box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    overflow: hidden;
}

.kpi-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
}

.kpi-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .06em;
    line-height: 1.3;
}

.kpi-icon {
    flex: 0 0 38px;
    width: 38px;
    height: 38px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    color: #FFFFFF;
    background: linear-gradient(135deg, var(--indigo), var(--blue));
    box-shadow: 0 10px 22px rgba(79,70,229,0.25);
}

.kpi-icon i {
    font-size: 17px;
}

.kpi-value {
    margin-top: 10px;
    color: var(--text);
    font-size: 27px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: 0;
    overflow-wrap: anywhere;
}

.kpi-subtext {
    margin-top: 7px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 650;
    min-height: 17px;
}

.kpi-delta {
    margin-top: 5px;
    font-size: 11px;
    font-weight: 900;
}

.positive {color: var(--green);}
.warning {color: var(--orange);}
.negative {color: var(--red);}
.neutral {color: var(--blue);}

.card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}

.card-title i {
    color: var(--indigo);
    font-size: 18px;
}

.card-title h3 {
    margin: 0;
    color: var(--text);
    font-size: 16px;
    font-weight: 900;
}

.card-subtitle {
    margin: -2px 0 6px 28px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}

.governance-card {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 20px 22px;
    border-radius: 18px;
    color: #FFFFFF;
    background: linear-gradient(135deg, #07152E, #0B2146);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 28px rgba(15,23,42,0.15);
}

.governance-card i {
    font-size: 22px;
    color: #93C5FD;
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

.empty-message {
    padding: 16px;
    border-radius: 14px;
    border: 1px dashed #CBD5E1;
    color: var(--muted);
    background: #F8FAFC;
    font-weight: 650;
}

@media (max-width: 1200px) {
    .block-container {
        padding-left: 18px;
        padding-right: 18px;
    }
    .hero {
        display: block;
    }
    .hero-badges {
        justify-content: flex-start;
        margin-top: 18px;
    }
    .kpi-value {
        font-size: 24px;
    }
}
</style>
"""
    )


def get_page() -> str:
    page = st.query_params.get("page", "Executive")
    return page if page in PAGES else "Executive"


def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return not df.empty and all(col in df.columns for col in cols)


def fmt_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}".replace(",", " ")


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def delta_text(current: float, previous: float, percent: bool = False, lower_is_better: bool = False) -> tuple[str, str]:
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "Reference indisponible", "neutral"
    delta = current - previous
    label = f"{delta:+.1%} vs periode precedente" if percent else f"{delta:+,.0f} vs periode precedente".replace(",", " ")
    good = delta >= 0
    if lower_is_better:
        good = not good
    return label, "positive" if good else "negative"


def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
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


def chart_card(title: str, subtitle: str, icon: str, fig: go.Figure | None, height: int = 320, legend: bool = True) -> None:
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
        st.plotly_chart(style_fig(fig, height=height, legend=legend), use_container_width=True, config={"displayModeBar": False})


def table_card(title: str, subtitle: str, icon: str, df: pd.DataFrame) -> None:
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
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_sidebar(page: str, latest_month: pd.Timestamp | None) -> None:
    links = []
    for name, meta in PAGES.items():
        active = " active" if name == page else ""
        links.append(
            f"""
<a class="sidebar-item{active}" href="?page={quote(name)}" target="_self">
  <i class="bi {meta["icon"]}"></i><span>{meta["label"]}</span>
</a>
"""
        )
    date = latest_month.strftime("%d/%m/%Y") if latest_month is not None and pd.notna(latest_month) else "Non disponible"
    st.sidebar.markdown(
        f"""
<div class="sidebar-brand">
  <div class="brand-logo"><i class="bi bi-person-workspace"></i></div>
  <div class="brand-title">Dashboard RH</div>
  <div class="brand-subtitle">Pilotage RH agrege</div>
</div>
<nav class="sidebar-nav">
  {''.join(links)}
</nav>
<div class="sidebar-footer">
  <div class="footer-label">Derniere mise a jour</div>
  <div class="footer-value">{date}</div>
  <div class="footer-label" style="margin-top:12px;">Usage</div>
  <div class="footer-value" style="font-size:13px;">Agrege uniquement</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_header(df: pd.DataFrame, latest_month: pd.Timestamp | None) -> tuple[str, str, str]:
    ref = latest_month.strftime("%Y-%m-%d") if latest_month is not None and pd.notna(latest_month) else "NA"
    sites = ["Tous"] + sorted(df["site"].dropna().unique().tolist()) if "site" in df.columns else ["Tous"]

    with st.container(border=True):
        st.markdown("<span class='header-marker'></span>", unsafe_allow_html=True)
        left, right = st.columns([1.25, 1.55], gap="large")
        with left:
            st.markdown(
                """
<div class="hero">
  <div class="hero-title">
    <div>
      <h1><i class="bi bi-bar-chart-line" style="font-size:28px;margin-right:10px;color:#93C5FD;"></i>Dashboard RH</h1>
      <p>Pilotage & Analyse des Ressources Humaines</p>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
            with c1:
                st.markdown("<div class='header-filter-label'>Trimestre</div>", unsafe_allow_html=True)
                trimestre = st.selectbox("Trimestre", ["T2 2026", "T1 2026", "T4 2025", "T3 2025", "Tous"], label_visibility="collapsed")
            with c2:
                st.markdown("<div class='header-filter-label'>Perimetre</div>", unsafe_allow_html=True)
                perimetre = st.selectbox("Perimetre", ["Tous", "Sites", "Metiers", "Equipes"], label_visibility="collapsed")
            with c3:
                st.markdown("<div class='header-filter-label'>Site</div>", unsafe_allow_html=True)
                site = st.selectbox("Site", sites, label_visibility="collapsed")
            with c4:
                st.markdown(
                    f"<div class='header-filter-label'>Reference</div><div class='hero-badge' style='min-width:0;width:100%;box-sizing:border-box;'><strong>{ref}</strong></div>",
                    unsafe_allow_html=True,
                )
    return trimestre, perimetre, site


def latest_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not has_cols(df, ["mois"]):
        return pd.DataFrame(), pd.DataFrame()
    latest_month = df["mois"].max()
    previous_month = df.loc[df["mois"] < latest_month, "mois"].max()
    latest = df[df["mois"] == latest_month].copy()
    previous = df[df["mois"] == previous_month].copy() if pd.notna(previous_month) else pd.DataFrame()
    return latest, previous


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


def render_executive(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    section_title("bi-speedometer2", "Vue Executive", "Synthese direction des indicateurs RH collectifs")
    latest, previous = latest_frames(df)
    if latest.empty:
        st.error("DATA GOLD indisponible ou colonne mois manquante.")
        return

    attrition = latest["departs_volontaires"].sum() / max(latest["effectif_reel"].sum(), 1)
    prev_attrition = previous["departs_volontaires"].sum() / max(previous["effectif_reel"].sum(), 1) if not previous.empty else 0
    tension = latest["tension_recrutement"].mean()
    prev_tension = previous["tension_recrutement"].mean() if not previous.empty else 0

    kpis = [
        ("bi-people-fill", "Effectif reel", latest["effectif_reel"].sum(), previous["effectif_reel"].sum() if not previous.empty else 0, "Population observee", False, False),
        ("bi-calendar2-check", "Effectif planifie", latest["effectif_planifie"].sum(), previous["effectif_planifie"].sum() if not previous.empty else 0, "Cible workforce planning", False, False),
        ("bi-diagram-3", "Ecart effectif", latest["ecart_effectif"].sum(), previous["ecart_effectif"].sum() if not previous.empty else 0, "Reel moins planifie", False, False),
        ("bi-arrow-repeat", "Taux attrition", attrition, prev_attrition, "Departs / effectif", True, True),
        ("bi-box-arrow-right", "Departs volontaires", latest["departs_volontaires"].sum(), previous["departs_volontaires"].sum() if not previous.empty else 0, "Volume mensuel agrege", False, True),
        ("bi-activity", "Tension globale", tension, prev_tension, "Indice recrutement", True, True),
    ]
    cols = st.columns(6, gap="small")
    for col, (icon, label, current, prev, subtext, is_pct, lower_is_better) in zip(cols, kpis):
        d_text, d_class = delta_text(current, prev, percent=is_pct, lower_is_better=lower_is_better)
        value = fmt_pct(current) if is_pct else fmt_int(current)
        if label == "Tension globale":
            value = fmt_num(current)
        with col:
            kpi_card(icon, label, value, subtext, d_text, d_class)

    monthly = df.groupby("mois", as_index=False).agg(
        effectif_reel=("effectif_reel", "sum"),
        effectif_planifie=("effectif_planifie", "sum"),
        departs_volontaires=("departs_volontaires", "sum"),
        recrutements_ouverts=("recrutements_ouverts", "sum"),
    )
    monthly["ecart_effectif"] = monthly["effectif_reel"] - monthly["effectif_planifie"]

    fig_headcount = go.Figure()
    fig_headcount.add_bar(x=monthly["mois"], y=monthly["effectif_reel"], name="Reel", marker_color="#4F46E5")
    fig_headcount.add_bar(x=monthly["mois"], y=monthly["effectif_planifie"], name="Planifie", marker_color="#C4B5FD")
    fig_headcount.add_scatter(x=monthly["mois"], y=monthly["ecart_effectif"], name="Ecart", mode="lines+markers", yaxis="y2", line=dict(color="#2563EB", width=3))
    fig_headcount.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", showgrid=False))

    tension_site = latest.groupby("site", as_index=False)["tension_recrutement"].mean().sort_values("tension_recrutement")
    fig_tension = px.bar(tension_site, x="tension_recrutement", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")

    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        chart_card("Effectif reel vs planifie", "Barres mensuelles et ligne d'ecart", "bi-bar-chart", fig_headcount, 330)
    with c2:
        chart_card("Tension par site", "Indice moyen de tension recrutement", "bi-geo-alt", fig_tension, 330, legend=False)

    c3, c4, c5 = st.columns(3, gap="medium")
    with c3:
        if has_cols(forecast_df, ["mois", "reel", "prediction_retenue"]):
            fc = forecast_df.groupby("mois", as_index=False).agg(reel=("reel", "sum"), prediction_retenue=("prediction_retenue", "sum")).tail(12)
            fig = px.line(fc, x="mois", y=["reel", "prediction_retenue"], markers=True, color_discrete_sequence=["#4F46E5", "#F97316"])
        else:
            fig = px.bar(monthly.tail(12), x="mois", y="departs_volontaires", color_discrete_sequence=["#4F46E5"])
        chart_card("Departs volontaires", "Reel vs prevision agregee", "bi-graph-up", fig, 285)
    with c4:
        top_attr = latest.groupby("equipe", as_index=False)["taux_attrition"].mean().sort_values("taux_attrition", ascending=False).head(5)
        fig = px.bar(top_attr, x="taux_attrition", y="equipe", orientation="h", color="taux_attrition", color_continuous_scale="OrRd")
        chart_card("Top 5 equipes attrition", "Collectifs a surveiller", "bi-exclamation-triangle", fig, 285, legend=False)
    with c5:
        gap_site = latest.groupby("site", as_index=False)["ecart_effectif"].sum().sort_values("ecart_effectif")
        fig = px.bar(gap_site, x="ecart_effectif", y="site", orientation="h", color="ecart_effectif", color_continuous_scale="RdYlGn")
        chart_card("Ecart effectif par site", "Reel moins planifie", "bi-sliders", fig, 285, legend=False)

    c6, c7 = st.columns(2, gap="medium")
    with c6:
        coverage = latest["couverture_competences_critiques"].mean()
        donut = go.Figure(data=[go.Pie(labels=["Couvert", "A renforcer"], values=[coverage, max(0, 1 - coverage)], hole=.62, marker_colors=["#16A34A", "#F97316"])])
        donut.update_layout(annotations=[dict(text=fmt_pct(coverage), x=.5, y=.5, showarrow=False, font=dict(size=28, color="#0F172A"))])
        chart_card("Couverture competences critiques", "Niveau de couverture moyen", "bi-award", donut, 280)
    with c7:
        fig = px.line(monthly.tail(12), x="mois", y="recrutements_ouverts", markers=True, color_discrete_sequence=["#2563EB"])
        chart_card("Recrutements ouverts", "Postes ouverts sur 12 mois", "bi-briefcase", fig, 280, legend=False)

    render_governance("Usage responsable", "Tous les indicateurs sont agreges au niveau equipe, site et metier. Aucun scoring individuel n'est produit.")


def render_data_quality(df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    section_title("bi-shield-check", "Data Quality", "Fiabilite des sources et regles appliquees")
    if quality_df.empty:
        st.error("dashboard_data_quality.csv est indisponible.")
        return

    score = quality_df["score_qualite_estime"].mean() if "score_qualite_estime" in quality_df.columns else 0
    missing = quality_df["valeurs_manquantes"].sum() if "valeurs_manquantes" in quality_df.columns else 0
    duplicates = quality_df["doublons"].sum() if "doublons" in quality_df.columns else 0
    issues = quality_df["incoherences_detectees"].sum() if "incoherences_detectees" in quality_df.columns else 0

    cols = st.columns(5, gap="small")
    cards = [
        ("bi-patch-check", "Score qualite global", f"{score:.1f}/100", "Plus haut = donnees plus fiables", "Controle qualite source", "positive"),
        ("bi-question-circle", "Valeurs manquantes", fmt_int(missing), "Controle exhaustivite", "A corriger si critique", "warning" if missing else "positive"),
        ("bi-copy", "Doublons", fmt_int(duplicates), "Controle unicite", "Doublons stricts", "warning" if duplicates else "positive"),
        ("bi-bug", "Incoherences", fmt_int(issues), "Regles metier", "Valeurs hors bornes", "negative" if issues else "positive"),
        ("bi-database-check", "Lignes GOLD DATA", fmt_int(len(df)), "Dataset final RH", "Table agregee", "neutral"),
    ]
    for col, (icon, label, value, subtext, delta, klass) in zip(cols, cards):
        with col:
            kpi_card(icon, label, value, subtext, delta, klass)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = px.bar(quality_df.sort_values("valeurs_manquantes"), x="valeurs_manquantes", y="fichier", orientation="h", color="valeurs_manquantes", color_continuous_scale="Blues")
        chart_card("Missing values par fichier", "Labels longs controles en horizontal", "bi-list-check", fig, 245, legend=False)
    with c2:
        fig = px.bar(quality_df.sort_values("doublons"), x="doublons", y="fichier", orientation="h", color="doublons", color_continuous_scale="Purples")
        chart_card("Doublons par fichier", "Lignes strictement dupliquees", "bi-files", fig, 245, legend=False)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        fig = px.bar(quality_df.sort_values("score_qualite_estime"), x="score_qualite_estime", y="fichier", orientation="h", color="score_qualite_estime", color_continuous_scale="RdYlGn")
        chart_card("Score qualite par source", "Score estime apres controles", "bi-stars", fig, 245, legend=False)
    with c4:
        rules = quality_df[["fichier", "regles_appliquees"]].drop_duplicates() if has_cols(quality_df, ["fichier", "regles_appliquees"]) else pd.DataFrame()
        table_card("Regles qualite appliquees", "Normalisation, bornage, agregation", "bi-tools", rules)

    table = quality_df.copy()
    if "incoherences_detectees" in table.columns:
        table["gravite"] = pd.cut(table["incoherences_detectees"], [-1, 0, 200, float("inf")], labels=["faible", "moyen", "eleve"]).astype(str)
    table_card("Anomalies detectees", "Lecture par source et niveau de vigilance", "bi-clipboard-data", table)


def render_rh_analytics(df: pd.DataFrame, analytics_df: pd.DataFrame) -> None:
    section_title("bi-people", "RH Analytics", "Dynamiques collectives par equipe, site et metier")
    latest, _ = latest_frames(df)
    if latest.empty:
        st.error("DATA GOLD indisponible.")
        return

    cols = st.columns(5, gap="small")
    kpis = [
        ("bi-diagram-2", "Equipes analysees", fmt_int(latest["equipe"].nunique()), "Collectifs actifs", "Granularite equipe", "neutral"),
        ("bi-radioactive", "Risque moyen", fmt_num(latest["indicateur_risque_collectif"].mean()), "Indice collectif", "Composite RH", "warning"),
        ("bi-geo", "Sites en tension", fmt_int((latest.groupby("site")["tension_recrutement"].mean() > .5).sum()), "Tension > 0.50", "Recrutement", "warning"),
        ("bi-award", "Couverture skills", fmt_pct(latest["couverture_competences_critiques"].mean()), "Moyenne collective", "Competences critiques", "positive"),
        ("bi-calendar-heart", "Absenteisme moyen", fmt_pct(latest["absenteeism_rate"].mean()), "Donnee agregee", "Signal RH", "neutral"),
    ]
    for col, item in zip(cols, kpis):
        with col:
            kpi_card(*item)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        top_risk = latest.sort_values("indicateur_risque_collectif", ascending=False).head(12)
        fig = px.bar(top_risk, x="indicateur_risque_collectif", y="equipe", orientation="h", color="site")
        chart_card("Classement equipes instables", "Top collectifs par risque agrege", "bi-sort-down", fig, 330)
    with c2:
        fig = px.scatter(latest, x="tension_recrutement", y="taux_attrition", size="effectif_reel", color="metier", hover_data=["site", "equipe"])
        chart_card("Profils d'equipes", "Tension recrutement x attrition", "bi-bounding-box", fig, 330)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        heat = latest.pivot_table(index="site", columns="metier", values="indicateur_risque_collectif", aggfunc="mean")
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r", labels=dict(color="Risque"))
        chart_card("Heatmap site x metier", "Risque collectif moyen", "bi-grid-3x3-gap", fig, 315, legend=False)
    with c4:
        site_gap = latest.groupby("site", as_index=False).agg(ecart_effectif=("ecart_effectif", "sum"), tension_recrutement=("tension_recrutement", "mean"))
        fig = px.bar(site_gap.sort_values("ecart_effectif"), x="ecart_effectif", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")
        chart_card("Ecarts effectif reel vs planifie", "Sous-effectif et tension par site", "bi-bar-chart-steps", fig, 315, legend=False)

    table_card("Tableau analytique agrege", "Aucun niveau individuel, uniquement equipe/site/metier", "bi-table", analytics_df if not analytics_df.empty else latest)


def render_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    section_title("bi-graph-up-arrow", "Forecast", "Prevision agregee des departs volontaires")
    if model_df.empty:
        st.error("model_results.csv est indisponible.")
        return

    best = model_df.sort_values("MAE").iloc[0] if "MAE" in model_df.columns else model_df.iloc[0]
    predicted = forecast_df["prediction_retenue"].sum() if "prediction_retenue" in forecast_df.columns else 0

    cols = st.columns(5, gap="small")
    kpis = [
        ("bi-box-arrow-up-right", "Departs prevus", fmt_int(predicted), "Volume agrege", "Prediction collective", "neutral"),
        ("bi-bullseye", "Erreur MAE", fmt_num(best.get("MAE", 0)), "Plus faible = mieux", "Erreur moyenne", "positive"),
        ("bi-crosshair", "RMSE", fmt_num(best.get("RMSE", 0)), "Penalise gros ecarts", "Validation modele", "positive"),
        ("bi-cpu", "Meilleur modele", str(best.get("modele", "-")), "Selection par MAE", "Benchmark", "neutral"),
        ("bi-shield-lock", "Confiance", "Controlee", "Usage encadre", "Limites documentees", "warning"),
    ]
    for col, item in zip(cols, kpis):
        with col:
            kpi_card(*item)

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        if has_cols(forecast_df, ["mois", "reel", "prediction_retenue"]):
            fc = forecast_df.groupby("mois", as_index=False).agg(reel=("reel", "sum"), prediction_retenue=("prediction_retenue", "sum"))
            fig = px.line(fc, x="mois", y=["reel", "prediction_retenue"], markers=True, color_discrete_sequence=["#4F46E5", "#F97316"])
        else:
            fig = None
        chart_card("Reel vs predit", "Volumes agreges de departs volontaires", "bi-graph-up", fig, 330)
    with c2:
        fig = px.bar(model_df, x="modele", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=["#4F46E5", "#2563EB"]) if has_cols(model_df, ["modele", "MAE", "RMSE"]) else None
        chart_card("Baseline vs modeles", "Comparaison des performances", "bi-columns-gap", fig, 330)

    if has_cols(df, ["departs_volontaires", "tension_recrutement", "couverture_competences_critiques", "effectif_reel", "absenteeism_rate", "engagement_score_avg"]):
        corr_cols = ["tension_recrutement", "couverture_competences_critiques", "effectif_reel", "absenteeism_rate", "engagement_score_avg"]
        imp = df[corr_cols + ["departs_volontaires"]].corr(numeric_only=True)["departs_volontaires"].drop("departs_volontaires").abs().reset_index()
        imp.columns = ["variable", "importance_proxy"]
        fig = px.bar(imp.sort_values("importance_proxy"), x="importance_proxy", y="variable", orientation="h", color="importance_proxy", color_continuous_scale="Blues")
    else:
        fig = None
    chart_card("Facteurs explicatifs agreges", "Proxy par correlation absolue", "bi-node-plus", fig, 280, legend=False)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        table_card("Tableau model_results", "Scores de validation des modeles", "bi-table", model_df)
    with c4:
        render_governance(
            "Gouvernance du modele",
            "Ces previsions sont agregees et ne doivent jamais etre utilisees pour scorer ou classer des collaborateurs individuellement.",
        )


def main() -> None:
    inject_css()
    if gold.empty:
        st.error("Impossible de charger la DATA GOLD. Verifiez data/processed/gold_data_rh.csv.")
        return

    latest_month = gold["mois"].max() if "mois" in gold.columns else None
    page = get_page()
    render_sidebar(page, latest_month)

    trimestre, perimetre, site = render_header(gold, latest_month)

    view = gold.copy()
    if site != "Tous" and "site" in view.columns:
        view = view[view["site"] == site].copy()

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

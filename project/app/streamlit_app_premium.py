from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
from typing import Iterable
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

st.set_page_config(page_title="Dashboard RH", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
PROCESSED = PROJECT_DIR / "data" / "processed"

PAGES = {
    "Executive": {"icon": "🏠", "label": "Executive"},
    "Data Quality": {"icon": "🛡️", "label": "Data Quality"},
    "RH Analytics": {"icon": "👥", "label": "RH Analytics"},
    "Forecast": {"icon": "📈", "label": "Forecast"},
}

@st.cache_data(show_spinner=False)
def read_csv_safe(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:
        st.warning(f"Impossible de charger {filename}: {exc}")
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

def inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#F5F7FB;--panel:#fff;--text:#0F172A;--muted:#64748B;--line:#E5E7EB;--navy:#07152E;--navy2:#0B2146;--blue:#4F46E5;--blue2:#2563EB;--green:#16A34A;--orange:#F97316;--red:#EF4444;--purple:#8B5CF6}
html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif;overflow-x:hidden}.stApp>header,footer,#MainMenu{display:none!important}.block-container{max-width:1640px;padding:18px 24px 34px 24px!important}div[data-testid="stVerticalBlock"]{gap:.85rem!important}div[data-testid="column"]{padding-top:0!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07152E 0%,#081A33 55%,#041027 100%)!important;border-right:1px solid rgba(255,255,255,.08);min-width:270px!important;max-width:270px!important}[data-testid="stSidebar"]>div{background:transparent!important;padding:22px 14px!important}.sidebar-brand{padding:6px 8px 20px;border-bottom:1px solid rgba(255,255,255,.11);margin-bottom:16px}.logo-tile{width:54px;height:54px;border-radius:18px;display:grid;place-items:center;background:linear-gradient(135deg,#4F46E5,#2563EB);color:white;font-size:28px;font-weight:900;box-shadow:0 18px 38px rgba(79,70,229,.33);margin-bottom:14px}.brand-title{color:white;font-size:22px;line-height:1.05;font-weight:900}.brand-subtitle{color:#AFC2E8;font-size:13px;font-weight:600;margin-top:7px}.nav-title{color:#8EA7D5;font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.09em;margin:16px 8px 10px}.sidebar-nav{display:flex;flex-direction:column;gap:9px}.sidebar-link{height:48px;display:flex;align-items:center;gap:13px;padding:0 14px;border-radius:15px;color:#DDE8FF!important;text-decoration:none!important;font-size:14px;font-weight:800;border:1px solid transparent;background:transparent;box-sizing:border-box}.sidebar-link:hover{color:white!important;background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.10)}.sidebar-link.active{color:white!important;background:linear-gradient(135deg,#4F46E5,#7C3AED);box-shadow:0 14px 30px rgba(79,70,229,.34)}.nav-icon{width:23px;height:23px;display:grid;place-items:center;font-size:17px}.sidebar-footer{margin-top:330px;padding:16px;border-radius:18px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.045)}.footer-line{display:flex;align-items:center;gap:10px;color:white;font-size:13px;font-weight:800;margin-bottom:6px}.footer-small{color:#AFC2E8;font-size:12px;line-height:1.45}
.hero-wrap{min-height:92px;border-radius:22px;padding:20px 24px;background:radial-gradient(circle at 18% 20%,rgba(79,70,229,.28),transparent 34%),linear-gradient(135deg,#07152E 0%,#0B2146 58%,#111827 100%);border:1px solid rgba(255,255,255,.10);box-shadow:0 18px 44px rgba(15,23,42,.18);display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:16px}.hero-main{display:flex;align-items:center;gap:16px}.hero-icon{width:50px;height:50px;border-radius:16px;display:grid;place-items:center;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.12);color:#93C5FD;font-size:26px}.hero-title{margin:0;color:white;font-size:31px;line-height:1.05;font-weight:900}.hero-subtitle{margin-top:7px;color:#D8E5FF;font-size:14px;font-weight:600}.hero-badge{display:inline-flex;align-items:center;gap:8px;margin-left:12px;padding:8px 12px;border-radius:999px;background:rgba(22,163,74,.13);color:#BBF7D0;border:1px solid rgba(22,163,74,.22);font-size:12px;font-weight:800}.filter-card{border-radius:16px;padding:10px 14px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.08);color:white;min-width:130px}.filter-card span{display:block;color:#BFD7FF;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.07em}.filter-card strong{display:block;margin-top:4px;color:white;font-size:14px;font-weight:900}.filter-row{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{min-height:42px;border-radius:14px;border:1px solid #DDE3EE;background:white;box-shadow:0 4px 14px rgba(15,23,42,.04)}.section-head{display:flex;align-items:center;gap:12px;margin:12px 0}.section-icon{width:42px;height:42px;border-radius:16px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(79,70,229,.12),rgba(37,99,235,.16));color:#4F46E5;font-size:21px}.section-title{margin:0;color:var(--text);font-size:25px;line-height:1.05;font-weight:900}.section-subtitle{margin-top:5px;color:var(--muted);font-size:14px;font-weight:600}.kpi-card{min-height:150px;border-radius:20px;padding:17px 18px;background:white;border:1px solid #E5E7EB;box-shadow:0 10px 28px rgba(15,23,42,.075);position:relative;overflow:hidden}.kpi-card:after{content:"";position:absolute;width:120px;height:85px;right:-48px;bottom:-45px;background:radial-gradient(circle,rgba(79,70,229,.12),transparent 68%)}.kpi-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.kpi-label{color:#64748B;font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.065em;line-height:1.35}.kpi-icon{width:42px;height:42px;flex:0 0 42px;border-radius:15px;display:grid;place-items:center;font-size:21px;color:white;box-shadow:0 12px 24px rgba(79,70,229,.25)}.kpi-blue{background:linear-gradient(135deg,#4F46E5,#2563EB)}.kpi-green{background:linear-gradient(135deg,#16A34A,#22C55E)}.kpi-orange{background:linear-gradient(135deg,#F97316,#F59E0B)}.kpi-red{background:linear-gradient(135deg,#EF4444,#F97316)}.kpi-purple{background:linear-gradient(135deg,#8B5CF6,#4F46E5)}.kpi-value{margin-top:13px;color:var(--text);font-size:30px;font-weight:900;line-height:1;letter-spacing:-.02em}.kpi-text{margin-top:8px;color:#64748B;font-size:12px;font-weight:650}.kpi-delta{margin-top:7px;font-size:11px;font-weight:900}.positive{color:var(--green)}.negative{color:var(--red)}.warning{color:var(--orange)}.neutral{color:var(--blue2)}
div[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid #E5E7EB!important;border-radius:20px!important;background:white!important;box-shadow:0 10px 28px rgba(15,23,42,.075)!important;overflow:hidden!important}div[data-testid="stVerticalBlockBorderWrapper"]>div{padding:18px 18px 14px!important}.card-title{display:flex;align-items:center;gap:10px;margin-bottom:2px}.card-icon{width:28px;height:28px;border-radius:10px;display:grid;place-items:center;background:rgba(79,70,229,.10);color:#4F46E5;font-size:16px}.card-title h3{margin:0;color:var(--text);font-size:16px;font-weight:900}.card-subtitle{margin:2px 0 10px 38px;color:#64748B;font-size:12px;font-weight:600}.governance-card{display:flex;align-items:flex-start;gap:14px;padding:18px 20px;border-radius:20px;background:linear-gradient(135deg,#07152E,#0B2146);color:white;border:1px solid rgba(255,255,255,.10);box-shadow:0 12px 28px rgba(15,23,42,.16)}.gov-icon{width:34px;height:34px;border-radius:12px;display:grid;place-items:center;background:rgba(255,255,255,.10);color:#93C5FD;font-size:18px}.gov-title{font-size:15px;font-weight:900;margin-bottom:4px}.gov-text{color:#D8E5FF;font-size:13px;line-height:1.45;font-weight:550}@media(max-width:1200px){.hero-wrap{display:block}.filter-row{justify-content:flex-start;margin-top:16px}.sidebar-footer{margin-top:80px}}
</style>
""", unsafe_allow_html=True)

def page_from_query() -> str:
    page = st.query_params.get("page", "Executive")
    return page if page in PAGES else "Executive"

def has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return not df.empty and all(c in df.columns for c in cols)

def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)

def fmt_int(value) -> str:
    try:
        if pd.isna(value): return "-"
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "-"

def fmt_num(value, digits: int = 2) -> str:
    try:
        if pd.isna(value): return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"

def fmt_pct(value) -> str:
    try:
        if pd.isna(value): return "-"
        return f"{float(value):.1%}"
    except Exception:
        return "-"

def delta_text(current: float, previous: float, percent: bool = False, lower_is_better: bool = False) -> tuple[str, str]:
    try:
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return "Référence indisponible", "neutral"
        delta = current - previous
        label = f"{delta:+.1%} vs période précédente" if percent else f"{delta:+,.0f} vs période précédente".replace(",", " ")
        good = delta >= 0
        if lower_is_better: good = not good
        return label, "positive" if good else "negative"
    except Exception:
        return "Référence indisponible", "neutral"

def latest_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "mois" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    latest_month = df["mois"].max()
    previous_month = df.loc[df["mois"] < latest_month, "mois"].max()
    latest = df[df["mois"] == latest_month].copy()
    previous = df[df["mois"] == previous_month].copy() if pd.notna(previous_month) else pd.DataFrame()
    return latest, previous

def chart_source(df: pd.DataFrame, index_col: str, value_cols: list[str]) -> pd.DataFrame:
    if df.empty or index_col not in df.columns: return pd.DataFrame()
    available = [c for c in value_cols if c in df.columns]
    if not available: return pd.DataFrame()
    return df[[index_col] + available].copy().set_index(index_col)

def esc(text: str) -> str:
    return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def render_sidebar(page: str, latest_month) -> None:
    nav_items = []
    for name, meta in PAGES.items():
        active = " active" if name == page else ""
        nav_items.append(f'<a class="sidebar-link{active}" href="?page={quote(name)}" target="_self"><span class="nav-icon">{meta["icon"]}</span><span>{meta["label"]}</span></a>')
    date = latest_month.strftime("%d/%m/%Y") if pd.notna(latest_month) else "Non disponible"
    st.sidebar.markdown(f"""
<div class="sidebar-brand"><div class="logo-tile">RH</div><div class="brand-title">Dashboard RH</div><div class="brand-subtitle">Pilotage RH agrégé</div></div>
<div class="nav-title">Navigation</div><nav class="sidebar-nav">{''.join(nav_items)}</nav>
<div class="sidebar-footer"><div class="footer-line">🔄 Dernière mise à jour</div><div class="footer-small">{date}</div><div style="height:12px"></div><div class="footer-line">🛡️ Usage</div><div class="footer-small">Données agrégées uniquement</div></div>
""", unsafe_allow_html=True)

def render_filters(df: pd.DataFrame) -> tuple[str, str, str]:
    sites = ["Tous"] + (sorted([str(x) for x in df["site"].dropna().unique().tolist()]) if "site" in df.columns else [])
    c1,c2,c3 = st.columns([1,1,1], gap="medium")
    with c1: trimestre = st.selectbox("Trimestre", ["T2 2026", "T1 2026", "T4 2025", "T3 2025", "Tous"])
    with c2: perimetre = st.selectbox("Périmètre", ["Tous", "Sites", "Métiers", "Équipes"])
    with c3: site = st.selectbox("Site", sites)
    return trimestre, perimetre, site

def render_header(latest_month, trimestre: str, perimetre: str, site: str) -> None:
    ref = latest_month.strftime("%Y-%m-%d") if pd.notna(latest_month) else "NA"
    st.markdown(f"""
<div class="hero-wrap"><div class="hero-main"><div class="hero-icon">📊</div><div><div><span class="hero-title">Dashboard RH</span><span class="hero-badge">🛡️ Données agrégées uniquement</span></div><div class="hero-subtitle">Pilotage & Analyse des Ressources Humaines</div></div></div>
<div class="filter-row"><div class="filter-card"><span>Trimestre</span><strong>{esc(trimestre)}</strong></div><div class="filter-card"><span>Périmètre</span><strong>{esc(perimetre)}</strong></div><div class="filter-card"><span>Site</span><strong>{esc(site)}</strong></div><div class="filter-card"><span>Référence</span><strong>{esc(ref)}</strong></div></div></div>
""", unsafe_allow_html=True)

def section_title(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-head"><div class="section-icon">{icon}</div><div><div class="section-title">{esc(title)}</div><div class="section-subtitle">{esc(subtitle)}</div></div></div>', unsafe_allow_html=True)

def kpi_card(icon: str, label: str, value: str, subtext: str, delta: str, delta_class: str = "neutral", color: str = "blue") -> None:
    st.markdown(f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">{esc(label)}</div><div class="kpi-icon kpi-{color}">{icon}</div></div><div class="kpi-value">{esc(value)}</div><div class="kpi-text">{esc(subtext)}</div><div class="kpi-delta {delta_class}">{esc(delta)}</div></div>', unsafe_allow_html=True)

def chart_title(title: str, subtitle: str, icon: str) -> None:
    st.markdown(f'<div class="card-title"><div class="card-icon">{icon}</div><h3>{esc(title)}</h3></div><div class="card-subtitle">{esc(subtitle)}</div>', unsafe_allow_html=True)

def chart_card(title: str, subtitle: str, icon: str, data: pd.DataFrame, chart_type: str = "bar", height: int = 320) -> None:
    if data.empty: return
    with st.container(border=True):
        chart_title(title, subtitle, icon)
        if chart_type == "line": st.line_chart(data, height=height, use_container_width=True)
        elif chart_type == "area": st.area_chart(data, height=height, use_container_width=True)
        else: st.bar_chart(data, height=height, use_container_width=True)

def table_card(title: str, subtitle: str, icon: str, df: pd.DataFrame, height: int | None = None) -> None:
    if df.empty: return
    with st.container(border=True):
        chart_title(title, subtitle, icon)
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)

def governance_card(title: str, text: str) -> None:
    st.markdown(f'<div class="governance-card"><div class="gov-icon">ℹ️</div><div><div class="gov-title">{esc(title)}</div><div class="gov-text">{esc(text)}</div></div></div>', unsafe_allow_html=True)

def render_executive(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    section_title("📈", "Vue Executive", "Synthèse direction des indicateurs RH collectifs")
    latest, previous = latest_frames(df)
    if latest.empty:
        st.error("DATA GOLD indisponible ou colonne 'mois' manquante."); return
    effectif_reel = safe_col(latest,"effectif_reel").sum(); effectif_planifie = safe_col(latest,"effectif_planifie").sum()
    ecart_effectif = safe_col(latest,"ecart_effectif").sum() if "ecart_effectif" in latest.columns else effectif_reel-effectif_planifie
    departs = safe_col(latest,"departs_volontaires").sum(); attrition = departs/max(effectif_reel,1); tension = safe_col(latest,"tension_recrutement").mean()
    prev_effectif_reel = safe_col(previous,"effectif_reel").sum() if not previous.empty else 0; prev_effectif_planifie = safe_col(previous,"effectif_planifie").sum() if not previous.empty else 0
    prev_ecart = safe_col(previous,"ecart_effectif").sum() if not previous.empty else 0; prev_departs = safe_col(previous,"departs_volontaires").sum() if not previous.empty else 0
    prev_attrition = prev_departs/max(prev_effectif_reel,1) if not previous.empty else 0; prev_tension = safe_col(previous,"tension_recrutement").mean() if not previous.empty else 0
    cards=[("👥","Effectif réel",fmt_int(effectif_reel),"Population observée",*delta_text(effectif_reel,prev_effectif_reel),"blue"),("🗓️","Effectif planifié",fmt_int(effectif_planifie),"Cible workforce planning",*delta_text(effectif_planifie,prev_effectif_planifie),"purple"),("⚖️","Écart effectif",fmt_int(ecart_effectif),"Réel moins planifié",*delta_text(ecart_effectif,prev_ecart),"orange"),("🔁","Taux d’attrition",fmt_pct(attrition),"Départs / effectif",*delta_text(attrition,prev_attrition,percent=True,lower_is_better=True),"green"),("↗️","Départs volontaires",fmt_int(departs),"Volume agrégé",*delta_text(departs,prev_departs,lower_is_better=True),"purple"),("🚦","Tension globale",fmt_num(tension),"Indice recrutement",*delta_text(tension,prev_tension,percent=True,lower_is_better=True),"red")]
    cols=st.columns(6,gap="small")
    for col,card in zip(cols,cards):
        with col: kpi_card(*card)
    monthly=pd.DataFrame()
    if "mois" in df.columns:
        agg={}
        for col in ["effectif_reel","effectif_planifie","departs_volontaires","recrutements_ouverts"]:
            if col in df.columns: agg[col]=(col,"sum")
        monthly=df.groupby("mois",as_index=False).agg(**agg) if agg else pd.DataFrame()
        if has_cols(monthly,["effectif_reel","effectif_planifie"]): monthly["ecart_effectif"]=monthly["effectif_reel"]-monthly["effectif_planifie"]
    c1,c2,c3=st.columns([1.2,1,1.1],gap="medium")
    with c1: chart_card("Effectif réel vs planifié","Évolution mensuelle agrégée","📊",chart_source(monthly,"mois",["effectif_reel","effectif_planifie","ecart_effectif"]),"line",320)
    with c2:
        if has_cols(latest,["site","tension_recrutement"]): chart_card("Tension par site","Indice moyen de recrutement","📍",latest.groupby("site")["tension_recrutement"].mean().sort_values().to_frame(),"bar",320)
    with c3:
        if has_cols(forecast_df,["mois","reel","prediction_retenue"]): chart_card("Départs volontaires","Réel vs prévision agrégée","📈",forecast_df.groupby("mois")[["reel","prediction_retenue"]].sum().tail(12),"line",320)
        else: chart_card("Départs volontaires","Volume mensuel agrégé","📈",chart_source(monthly.tail(12),"mois",["departs_volontaires"]),"bar",320)
    c4,c5,c6=st.columns(3,gap="medium")
    with c4:
        if has_cols(latest,["equipe","taux_attrition"]): chart_card("Top 5 équipes","Taux d’attrition moyen","🔥",latest.groupby("equipe")["taux_attrition"].mean().sort_values(ascending=False).head(5).sort_values().to_frame(),"bar",280)
    with c5:
        if has_cols(latest,["site","ecart_effectif"]): chart_card("Écart effectif par site","Réel moins planifié","⚖️",latest.groupby("site")["ecart_effectif"].sum().sort_values().to_frame(),"bar",280)
    with c6: chart_card("Recrutements ouverts","Postes ouverts sur 12 mois","💼",chart_source(monthly.tail(12),"mois",["recrutements_ouverts"]),"line",280)
    if has_cols(latest,["couverture_competences_critiques"]): governance_card("Couverture compétences critiques", f"Couverture moyenne collective : {fmt_pct(safe_col(latest,'couverture_competences_critiques').mean())}.")
    governance_card("Usage responsable","Tous les indicateurs sont calculés au niveau agrégé : équipe, site et métier. Aucune décision individuelle automatisée n’est prévue ou permise.")

def render_data_quality(df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    section_title("🛡️","Data Quality","Contrôle des sources, anomalies détectées et règles de fiabilisation")
    if quality_df.empty: st.error("dashboard_data_quality.csv est indisponible."); return
    score=safe_col(quality_df,"score_qualite_estime").mean() if "score_qualite_estime" in quality_df.columns else 0; missing=safe_col(quality_df,"valeurs_manquantes").sum() if "valeurs_manquantes" in quality_df.columns else 0; duplicates=safe_col(quality_df,"doublons").sum() if "doublons" in quality_df.columns else 0; issues=safe_col(quality_df,"incoherences_detectees").sum() if "incoherences_detectees" in quality_df.columns else 0
    kpis=[("✅","Score qualité global",f"{score:.1f}/100","Plus haut = données plus fiables","Contrôle source","positive","green"),("❔","Valeurs manquantes",fmt_int(missing),"Exhaustivité","À corriger si critique","warning" if missing else "positive","orange"),("📄","Doublons",fmt_int(duplicates),"Unicité","Doublons stricts","warning" if duplicates else "positive","purple"),("🐞","Incohérences",fmt_int(issues),"Règles métier","Valeurs hors bornes","negative" if issues else "positive","red"),("🗄️","Lignes GOLD DATA",fmt_int(len(df)),"Dataset final RH","Table agrégée","neutral","blue")]
    cols=st.columns(5,gap="small")
    for col,item in zip(cols,kpis):
        with col: kpi_card(*item)
    c1,c2=st.columns(2,gap="medium")
    with c1:
        if has_cols(quality_df,["fichier","valeurs_manquantes"]): chart_card("Missing values par fichier","Volume total de valeurs manquantes","🔎",quality_df.groupby("fichier")["valeurs_manquantes"].sum().sort_values().to_frame(),"bar",260)
    with c2:
        if has_cols(quality_df,["fichier","doublons"]): chart_card("Doublons par fichier","Lignes strictement dupliquées","📚",quality_df.groupby("fichier")["doublons"].sum().sort_values().to_frame(),"bar",260)
    c3,c4=st.columns(2,gap="medium")
    with c3:
        if has_cols(quality_df,["fichier","score_qualite_estime"]): chart_card("Score qualité par source","Score estimé après contrôles","⭐",quality_df.groupby("fichier")["score_qualite_estime"].mean().sort_values().to_frame(),"bar",260)
    with c4: table_card("Règles qualité appliquées","Principes de fiabilisation DATA GOLD","🛠️",quality_df[[c for c in ["fichier","regles_appliquees"] if c in quality_df.columns]].drop_duplicates(),260)
    table_card("Anomalies détectées","Lecture par source et niveau de vigilance","📋",quality_df,320)

def render_rh_analytics(df: pd.DataFrame, analytics_df: pd.DataFrame) -> None:
    section_title("👥","RH Analytics","Dynamiques collectives par équipe, site et métier")
    latest,_=latest_frames(df)
    if latest.empty: st.error("DATA GOLD indisponible."); return
    kpis=[("🧩","Équipes analysées",fmt_int(latest["equipe"].nunique() if "equipe" in latest.columns else len(latest)),"Collectifs actifs","Granularité équipe","neutral","blue"),("⚠️","Risque moyen",fmt_num(safe_col(latest,"indicateur_risque_collectif").mean()),"Indice collectif","Composite RH","warning","orange"),("📍","Sites",fmt_int(latest["site"].nunique() if "site" in latest.columns else 0),"Périmètre observé","Analyse multi-site","neutral","purple"),("🎯","Couverture skills",fmt_pct(safe_col(latest,"couverture_competences_critiques").mean()),"Compétences critiques","Moyenne collective","positive","green"),("🗓️","Absenteisme moyen",fmt_pct(safe_col(latest,"absenteeism_rate").mean()),"Donnée agrégée","Signal RH","neutral","blue")]
    cols=st.columns(5,gap="small")
    for col,item in zip(cols,kpis):
        with col: kpi_card(*item)
    c1,c2=st.columns(2,gap="medium")
    with c1:
        if has_cols(latest,["equipe","indicateur_risque_collectif"]): chart_card("Équipes les plus instables","Top risque collectif agrégé","🔥",latest.groupby("equipe")["indicateur_risque_collectif"].mean().sort_values(ascending=False).head(12).sort_values().to_frame(),"bar",330)
    with c2:
        if has_cols(latest,["site","tension_recrutement"]): chart_card("Sites en tension","Indice moyen de tension recrutement","📍",latest.groupby("site")["tension_recrutement"].mean().sort_values().to_frame(),"bar",330)
    c3,c4=st.columns(2,gap="medium")
    with c3:
        if has_cols(latest,["site","ecart_effectif"]): chart_card("Écarts effectifs","Réel vs planifié par site","⚖️",latest.groupby("site")["ecart_effectif"].sum().sort_values().to_frame(),"bar",310)
    with c4:
        if has_cols(latest,["metier","couverture_competences_critiques"]): chart_card("Couverture compétences","Couverture moyenne par métier","🎯",latest.groupby("metier")["couverture_competences_critiques"].mean().sort_values().to_frame(),"bar",310)
    table_card("Tableau analytique agrégé","Aucun niveau individuel, uniquement équipe/site/métier","📊",analytics_df if not analytics_df.empty else latest,380)

def render_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    section_title("📈","Forecast","Prévision agrégée des départs volontaires")
    if model_df.empty: st.error("model_results.csv est indisponible."); return
    best=model_df.sort_values("MAE").iloc[0] if "MAE" in model_df.columns else model_df.iloc[0]; predicted=safe_col(forecast_df,"prediction_retenue").sum() if "prediction_retenue" in forecast_df.columns else 0
    kpis=[("↗️","Départs prévus",fmt_int(predicted),"Volume agrégé","Prévision collective","neutral","purple"),("🎯","Erreur MAE",fmt_num(best.get("MAE",0)),"Plus faible = mieux","Erreur moyenne","positive","green"),("📐","RMSE",fmt_num(best.get("RMSE",0)),"Pénalise gros écarts","Validation modèle","positive","green"),("🤖","Meilleur modèle",str(best.get("modele","-")),"Sélection par MAE","Benchmark","neutral","blue"),("🛡️","Confiance","Contrôlée","Usage encadré","Limites documentées","warning","orange")]
    cols=st.columns(5,gap="small")
    for col,item in zip(cols,kpis):
        with col: kpi_card(*item)
    c1,c2=st.columns([1.2,1],gap="medium")
    with c1:
        if has_cols(forecast_df,["mois","reel","prediction_retenue"]): chart_card("Réel vs prédit","Volumes agrégés de départs volontaires","📈",forecast_df.groupby("mois")[["reel","prediction_retenue"]].sum(),"line",330)
    with c2:
        available=[c for c in ["MAE","RMSE","R2"] if c in model_df.columns]
        if "modele" in model_df.columns and available: chart_card("Baseline vs modèles","Comparaison des performances","📊",model_df.set_index("modele")[available],"bar",330)
    table_card("Tableau model_results","Scores de validation des modèles","📋",model_df,310)
    governance_card("Gouvernance du modèle","Ces prévisions sont agrégées et ne doivent jamais être utilisées pour scorer, classer ou sanctionner des collaborateurs individuellement.")


def _pdf_clean(value) -> str:
    """Keep PDF text compatible without crashing on accents/symbols."""
    txt = str(value) if value is not None else ""
    repl = {
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a",
        "î": "i", "ï": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c",
        "É": "E", "È": "E", "À": "A", "Ç": "C", "%": "%"
    }
    for a, b in repl.items():
        txt = txt.replace(a, b)
    return txt.encode("latin-1", "ignore").decode("latin-1")


def _pdf_section(pdf, title: str) -> None:
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 9, _pdf_clean(title), ln=True)
    pdf.set_draw_color(79, 70, 229)
    pdf.set_line_width(0.6)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(4)


def _pdf_kpi(pdf, label: str, value: str, note: str = "") -> None:
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(56, 6, _pdf_clean(label), border=0)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(38, 6, _pdf_clean(value), border=0)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 6, _pdf_clean(note))


def _df_preview_for_pdf(pdf, df: pd.DataFrame, title: str, max_rows: int = 8, max_cols: int = 5) -> None:
    if df is None or df.empty:
        return
    _pdf_section(pdf, title)
    preview = df.copy().head(max_rows)
    preview = preview[[c for c in preview.columns[:max_cols]]]
    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    col_width = 186 / max(1, len(preview.columns))
    for col in preview.columns:
        pdf.cell(col_width, 7, _pdf_clean(col)[:22], border=1, fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 7)
    for _, row in preview.iterrows():
        for col in preview.columns:
            pdf.cell(col_width, 6, _pdf_clean(row[col])[:22], border=1)
        pdf.ln()
    pdf.ln(2)


def generate_pdf_report(gold_df: pd.DataFrame, quality_df: pd.DataFrame, analytics_df: pd.DataFrame, forecast_df: pd.DataFrame, model_df: pd.DataFrame) -> bytes:
    """Generate a complete aggregated HR report as PDF bytes for Streamlit download."""
    if FPDF is None:
        raise RuntimeError("La librairie fpdf2 n'est pas installee. Ajoute `fpdf2` dans requirements.txt.")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Cover
    pdf.set_fill_color(7, 21, 46)
    pdf.rect(0, 0, 210, 48, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 24)
    pdf.set_xy(12, 14)
    pdf.cell(0, 10, _pdf_clean("Rapport Final - Dashboard RH"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_x(12)
    pdf.cell(0, 8, _pdf_clean("Pilotage & Analyse des Ressources Humaines - Donnees agregees uniquement"), ln=True)
    pdf.set_text_color(100, 116, 139)
    pdf.set_font("Arial", "", 10)
    pdf.set_xy(12, 55)
    pdf.multi_cell(0, 6, _pdf_clean(
        "Ce rapport synthetise les indicateurs RH collectifs issus de la DATA GOLD : effectifs, attrition, tension recrutement, qualite des donnees, analyses RH et forecast. "
        "Il ne doit jamais etre utilise pour scorer, classer ou automatiser une decision individuelle sur un collaborateur."
    ))

    latest = pd.DataFrame()
    previous = pd.DataFrame()
    if gold_df is not None and not gold_df.empty and "mois" in gold_df.columns:
        df = gold_df.copy()
        df["mois"] = pd.to_datetime(df["mois"], errors="coerce")
        lm = df["mois"].max()
        pm = df.loc[df["mois"] < lm, "mois"].max()
        latest = df[df["mois"] == lm].copy()
        previous = df[df["mois"] == pm].copy() if pd.notna(pm) else pd.DataFrame()

    _pdf_section(pdf, "1. Executive Summary")
    if not latest.empty:
        effectif_reel = latest["effectif_reel"].sum() if "effectif_reel" in latest.columns else 0
        effectif_plan = latest["effectif_planifie"].sum() if "effectif_planifie" in latest.columns else 0
        ecart = effectif_reel - effectif_plan
        departs = latest["departs_volontaires"].sum() if "departs_volontaires" in latest.columns else 0
        attrition = departs / max(effectif_reel, 1)
        tension = latest["tension_recrutement"].mean() if "tension_recrutement" in latest.columns else 0
        _pdf_kpi(pdf, "Effectif reel", fmt_int(effectif_reel), "Population observee")
        _pdf_kpi(pdf, "Effectif planifie", fmt_int(effectif_plan), "Cible workforce planning")
        _pdf_kpi(pdf, "Ecart effectif", fmt_int(ecart), "Reel moins planifie")
        _pdf_kpi(pdf, "Taux attrition", fmt_pct(attrition), "Departs volontaires / effectif reel")
        _pdf_kpi(pdf, "Departs volontaires", fmt_int(departs), "Volume agrege")
        _pdf_kpi(pdf, "Tension globale", fmt_num(tension), "Indice moyen de recrutement")
    else:
        pdf.multi_cell(0, 6, _pdf_clean("DATA GOLD indisponible ou colonne mois manquante."))

    _pdf_section(pdf, "2. Data Quality")
    if quality_df is not None and not quality_df.empty:
        score = safe_col(quality_df, "score_qualite_estime").mean() if "score_qualite_estime" in quality_df.columns else 0
        missing = safe_col(quality_df, "valeurs_manquantes").sum() if "valeurs_manquantes" in quality_df.columns else 0
        duplicates = safe_col(quality_df, "doublons").sum() if "doublons" in quality_df.columns else 0
        issues = safe_col(quality_df, "incoherences_detectees").sum() if "incoherences_detectees" in quality_df.columns else 0
        _pdf_kpi(pdf, "Score qualite", f"{score:.1f}/100", "Score estime apres controles")
        _pdf_kpi(pdf, "Valeurs manquantes", fmt_int(missing), "Controle exhaustivite")
        _pdf_kpi(pdf, "Doublons", fmt_int(duplicates), "Controle unicite")
        _pdf_kpi(pdf, "Incoherences", fmt_int(issues), "Regles metier")
    else:
        pdf.multi_cell(0, 6, _pdf_clean("Fichier dashboard_data_quality.csv indisponible."))

    _pdf_section(pdf, "3. RH Analytics")
    if not latest.empty:
        equipes = latest["equipe"].nunique() if "equipe" in latest.columns else len(latest)
        sites = latest["site"].nunique() if "site" in latest.columns else 0
        couverture = latest["couverture_competences_critiques"].mean() if "couverture_competences_critiques" in latest.columns else 0
        risque = latest["indicateur_risque_collectif"].mean() if "indicateur_risque_collectif" in latest.columns else 0
        _pdf_kpi(pdf, "Equipes analysees", fmt_int(equipes), "Collectifs actifs")
        _pdf_kpi(pdf, "Sites observes", fmt_int(sites), "Perimetre multi-site")
        _pdf_kpi(pdf, "Couverture skills", fmt_pct(couverture), "Competences critiques")
        _pdf_kpi(pdf, "Risque moyen", fmt_num(risque), "Indice collectif")

    _pdf_section(pdf, "4. Forecast")
    if model_df is not None and not model_df.empty:
        best = model_df.sort_values("MAE").iloc[0] if "MAE" in model_df.columns else model_df.iloc[0]
        predicted = safe_col(forecast_df, "prediction_retenue").sum() if forecast_df is not None and "prediction_retenue" in forecast_df.columns else 0
        _pdf_kpi(pdf, "Departs prevus", fmt_int(predicted), "Volume agrege prevu")
        _pdf_kpi(pdf, "Meilleur modele", best.get("modele", "-"), "Selection selon performance")
        _pdf_kpi(pdf, "MAE", fmt_num(best.get("MAE", 0)), "Erreur moyenne absolue")
        _pdf_kpi(pdf, "RMSE", fmt_num(best.get("RMSE", 0)), "Erreur quadratique moyenne")
    else:
        pdf.multi_cell(0, 6, _pdf_clean("Fichier model_results.csv indisponible."))

    _df_preview_for_pdf(pdf, quality_df, "Annexe A - Apercu Data Quality")
    _df_preview_for_pdf(pdf, analytics_df, "Annexe B - Apercu RH Analytics")
    _df_preview_for_pdf(pdf, model_df, "Annexe C - Resultats Modeles")

    _pdf_section(pdf, "5. Gouvernance et usage responsable")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 6, _pdf_clean(
        "Les analyses et previsions presentees sont strictement agregees aux niveaux equipe, site et metier. "
        "Le dashboard et le modele ne doivent pas etre utilises pour noter, classer, sanctionner ou automatiser une decision individuelle. "
        "Les resultats doivent etre interpretes comme une aide au pilotage RH, avec validation humaine et prise en compte du contexte metier."
    ))

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)


def render_pdf_export_button() -> None:
    try:
        pdf_bytes = generate_pdf_report(gold, quality, analytics, forecast, model_results)
        st.download_button(
            label="📄 Exporter le rapport complet en PDF",
            data=pdf_bytes,
            file_name=f"rapport_dashboard_rh_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Export PDF indisponible : {exc}")


def main() -> None:
    inject_css()
    if gold.empty:
        st.error("Impossible de charger la DATA GOLD. Vérifie que `data/processed/gold_data_rh.csv` existe sur Streamlit Cloud.")
        st.stop()
    latest_month = gold["mois"].max() if "mois" in gold.columns else pd.NaT
    page=page_from_query(); render_sidebar(page,latest_month)
    trimestre,perimetre,site=render_filters(gold); view=gold.copy()
    if site!="Tous" and "site" in view.columns: view=view[view["site"].astype(str)==site].copy()
    render_header(latest_month,trimestre,perimetre,site)
    render_pdf_export_button()
    if page=="Executive": render_executive(view,forecast)
    elif page=="Data Quality": render_data_quality(view,quality)
    elif page=="RH Analytics": render_rh_analytics(view,analytics)
    elif page=="Forecast": render_forecast(view,forecast,model_results)

if __name__ == "__main__":
    main()

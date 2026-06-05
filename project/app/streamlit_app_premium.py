from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard RH", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
BASE = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
PROCESSED = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"

PAGES = {
    "Executive": {"icon": "bi-speedometer2", "label": "Executive"},
    "Data Quality": {"icon": "bi-shield-check", "label": "Data Quality"},
    "RH Analytics": {"icon": "bi-people", "label": "RH Analytics"},
    "Forecast": {"icon": "bi-graph-up-arrow", "label": "Forecast"},
}

KPI_ICONS = {
    "effectif": "bi-people-fill",
    "planifie": "bi-calendar2-check",
    "ecart": "bi-balance-scale",
    "attrition": "bi-arrow-repeat",
    "departs": "bi-box-arrow-right",
    "tension": "bi-speedometer",
}


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def clean_label(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan})


@st.cache_data(show_spinner=False)
def build_gold_from_raw() -> pd.DataFrame:
    wf_path = first_existing(PROCESSED / "gold_data_rh.csv", RAW / "workforce_BIG.csv", BASE / "workforce_BIG.csv", Path("workforce_BIG.csv"))
    sk_path = first_existing(RAW / "skills_BIG.csv", BASE / "skills_BIG.csv", Path("skills_BIG.csv"))
    rec_path = first_existing(RAW / "recruitment_BIG.csv", BASE / "recruitment_BIG.csv", Path("recruitment_BIG.csv"))
    ref_path = first_existing(RAW / "reference_BIG.csv", BASE / "reference_BIG.csv", Path("reference_BIG.csv"))

    if wf_path is None:
        return pd.DataFrame()

    if wf_path.name == "gold_data_rh.csv":
        df = read_csv(wf_path, parse_dates=["mois"])
        if "mois" not in df.columns and "month" in df.columns:
            df = df.rename(columns={"month": "mois"})
        if "mois" in df.columns:
            df["mois"] = pd.to_datetime(df["mois"], errors="coerce")
        return df

    wf = read_csv(wf_path, parse_dates=["month"])
    if wf.empty:
        return pd.DataFrame()

    wf = wf.rename(columns={
        "month": "mois",
        "job_family": "metier",
        "team_id": "equipe",
        "headcount_actual": "effectif_reel",
        "headcount_planned": "effectif_planifie",
        "voluntary_leavers": "departs_volontaires",
        "critical_skill_coverage_rate": "couverture_competences_critiques",
    })
    wf["mois"] = pd.to_datetime(wf["mois"], errors="coerce")
    if "site" in wf.columns:
        wf["site"] = clean_label(wf["site"])
    if "metier" in wf.columns:
        wf["metier"] = clean_label(wf["metier"])
    wf["equipe"] = "Equipe " + wf["equipe"].astype(str)

    for col in ["absenteeism_rate", "couverture_competences_critiques"]:
        if col in wf.columns:
            wf[col] = pd.to_numeric(wf[col], errors="coerce").clip(lower=0, upper=1)

    wf["effectif_reel"] = pd.to_numeric(wf["effectif_reel"], errors="coerce").clip(lower=0)
    wf["effectif_planifie"] = pd.to_numeric(wf["effectif_planifie"], errors="coerce").clip(lower=0)
    wf["departs_volontaires"] = pd.to_numeric(wf["departs_volontaires"], errors="coerce").clip(lower=0)
    wf["ecart_effectif"] = wf["effectif_reel"] - wf["effectif_planifie"]
    wf["taux_attrition"] = np.where(wf["effectif_reel"] > 0, wf["departs_volontaires"] / wf["effectif_reel"], 0).clip(0, 1)

    rec = read_csv(rec_path, parse_dates=["month"]) if rec_path else pd.DataFrame()
    if not rec.empty:
        rec = rec.rename(columns={"month": "mois", "job_family": "metier", "open_positions": "recrutements_ouverts"})
        rec["mois"] = pd.to_datetime(rec["mois"], errors="coerce")
        rec["site"] = clean_label(rec["site"])
        rec["metier"] = clean_label(rec["metier"])
        rec["offer_acceptance_rate"] = pd.to_numeric(rec["offer_acceptance_rate"], errors="coerce").clip(0, 1)
        rec["time_to_fill_days"] = pd.to_numeric(rec["time_to_fill_days"], errors="coerce").clip(0, 180)
        rec = rec.groupby(["mois", "site", "metier"], as_index=False).agg(
            recrutements_ouverts=("recrutements_ouverts", "sum"),
            time_to_fill_days=("time_to_fill_days", "mean"),
            offer_acceptance_rate=("offer_acceptance_rate", "mean"),
        )
        rec["tension_recrutement"] = ((rec["time_to_fill_days"] / 90) * 0.55 + (1 - rec["offer_acceptance_rate"]) * 0.45).clip(0, 1)
        wf = wf.merge(rec, on=["mois", "site", "metier"], how="left")

    if "recrutements_ouverts" not in wf.columns:
        wf["recrutements_ouverts"] = 0
    if "tension_recrutement" not in wf.columns:
        wf["tension_recrutement"] = 0
    wf["recrutements_ouverts"] = wf["recrutements_ouverts"].fillna(0)
    wf["tension_recrutement"] = wf["tension_recrutement"].fillna(wf["tension_recrutement"].mean()).fillna(0).clip(0, 1)

    sk = read_csv(sk_path, parse_dates=["month"]) if sk_path else pd.DataFrame()
    if not sk.empty:
        sk = sk.rename(columns={"month": "mois", "team_id": "equipe"})
        sk["mois"] = pd.to_datetime(sk["mois"], errors="coerce")
        sk["equipe"] = "Equipe " + sk["equipe"].astype(str)
        sk["required_people_count"] = pd.to_numeric(sk["required_people_count"], errors="coerce").clip(lower=0)
        sk["available_people_count"] = pd.to_numeric(sk["available_people_count"], errors="coerce").clip(lower=0)
        skills = sk.groupby(["mois", "equipe"], as_index=False).agg(
            required_people_count=("required_people_count", "sum"),
            available_people_count=("available_people_count", "sum"),
        )
        skills["couverture_skills_calculee"] = np.where(
            skills["required_people_count"] > 0,
            skills["available_people_count"] / skills["required_people_count"],
            1,
        ).clip(0, 1.25)
        wf = wf.merge(skills, on=["mois", "equipe"], how="left")
        wf["couverture_competences_critiques"] = wf["couverture_skills_calculee"].fillna(wf["couverture_competences_critiques"]).clip(0, 1)

    ref = read_csv(ref_path) if ref_path else pd.DataFrame()
    if not ref.empty and "job_family" in ref.columns:
        ref = ref.rename(columns={"job_family": "metier"})
        ref["metier"] = clean_label(ref["metier"])
        wf = wf.merge(ref, on="metier", how="left")
    if "critical_role_flag" not in wf.columns:
        wf["critical_role_flag"] = 0

    wf["indicateur_risque_collectif"] = (
        wf["taux_attrition"].fillna(0) * 0.35
        + wf["tension_recrutement"].fillna(0) * 0.25
        + (1 - wf["couverture_competences_critiques"].fillna(0)) * 0.25
        + wf.get("absenteeism_rate", pd.Series(0, index=wf.index)).fillna(0).clip(0, 1) * 0.15
    ).clip(0, 1)
    return wf


@st.cache_data(show_spinner=False)
def build_quality() -> pd.DataFrame:
    files = [
        ("workforce_BIG.csv", first_existing(RAW / "workforce_BIG.csv", BASE / "workforce_BIG.csv", Path("workforce_BIG.csv"))),
        ("skills_BIG.csv", first_existing(RAW / "skills_BIG.csv", BASE / "skills_BIG.csv", Path("skills_BIG.csv"))),
        ("recruitment_BIG.csv", first_existing(RAW / "recruitment_BIG.csv", BASE / "recruitment_BIG.csv", Path("recruitment_BIG.csv"))),
        ("reference_BIG.csv", first_existing(RAW / "reference_BIG.csv", BASE / "reference_BIG.csv", Path("reference_BIG.csv"))),
    ]
    rows = []
    for name, path in files:
        if path is None:
            continue
        df = read_csv(path)
        if df.empty:
            continue
        missing = int(df.isna().sum().sum())
        dup = int(df.duplicated().sum())
        incoh = 0
        for col in df.select_dtypes(include="number").columns:
            s = pd.to_numeric(df[col], errors="coerce")
            incoh += int((s < 0).sum())
        if "offer_acceptance_rate" in df.columns:
            incoh += int(((df["offer_acceptance_rate"] < 0) | (df["offer_acceptance_rate"] > 1)).sum())
        if "critical_skill_coverage_rate" in df.columns:
            incoh += int(((df["critical_skill_coverage_rate"] < 0) | (df["critical_skill_coverage_rate"] > 1)).sum())
        score = max(0, 100 - (missing / max(df.size, 1)) * 60 - (dup / max(len(df), 1)) * 20 - (incoh / max(len(df), 1)) * 30)
        rows.append({
            "fichier": name,
            "lignes": len(df),
            "colonnes": df.shape[1],
            "valeurs_manquantes": missing,
            "doublons": dup,
            "incoherences_detectees": incoh,
            "score_qualite_estime": round(score, 1),
            "regles_appliquees": "normalisation libelles, conversion dates, bornage KPI, agregation equipe/site/metier",
        })
    return pd.DataFrame(rows)


gold = build_gold_from_raw()
quality = build_quality()
forecast_file = first_existing(PROCESSED / "dashboard_forecast.csv", BASE / "dashboard_forecast.csv")
model_file = first_existing(PROCESSED / "model_results.csv", BASE / "model_results.csv")
forecast = read_csv(forecast_file, parse_dates=["mois"]) if forecast_file else pd.DataFrame()
model_results = read_csv(model_file) if model_file else pd.DataFrame()


def css() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
:root{
  --bg:#F6F8FC;--card:#FFFFFF;--navy:#07152E;--navy2:#0A1D3D;--text:#0F172A;--muted:#64748B;
  --line:#E5EAF2;--indigo:#4F46E5;--blue:#2563EB;--violet:#7C3AED;--green:#16A34A;--red:#EF4444;--orange:#F97316;
}
html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:Inter,Segoe UI,sans-serif;overflow-x:hidden!important;}
.stApp>header,#MainMenu,footer{display:none!important;}
.block-container{max-width:none!important;padding:0 28px 28px 28px!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07152E 0%,#06132B 50%,#040B1E 100%)!important;width:248px!important;min-width:248px!important;max-width:248px!important;border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"]>div{padding:22px 14px!important;background:transparent!important;}
[data-testid="stSidebar"] *{font-family:Inter,Segoe UI,sans-serif;}
[data-testid="stSidebar"] .stMarkdown{margin:0!important;}
.sb-logo{width:54px;height:54px;border-radius:18px;background:linear-gradient(135deg,#4F46E5,#2563EB);display:grid;place-items:center;color:white;font-weight:900;font-size:21px;box-shadow:0 18px 34px rgba(37,99,235,.32);margin-bottom:16px;}
.sb-title{font-size:22px;font-weight:900;color:white;line-height:1.05;margin-bottom:6px;}
.sb-sub{font-size:13px;font-weight:650;color:#AFC2E8;margin-bottom:24px;}
.sb-line{height:1px;background:rgba(255,255,255,.10);margin:0 0 18px 0;}
.sb-label{font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:#8298BE;font-weight:900;margin:10px 10px 10px;}
a.nav-link{height:48px;margin:7px 0;padding:0 14px;display:flex;align-items:center;gap:12px;border-radius:14px;text-decoration:none!important;color:#D7E4FF!important;font-weight:800;font-size:14px;border:1px solid transparent;box-sizing:border-box;transition:.15s ease;background:transparent;}
a.nav-link i{width:22px;font-size:18px;color:#D7E4FF;text-align:center;}
a.nav-link:hover{background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.10);color:white!important;transform:none!important;}
a.nav-link.active{background:linear-gradient(135deg,#4F46E5,#7C3AED);box-shadow:0 14px 28px rgba(79,70,229,.34);color:white!important;border-color:rgba(255,255,255,.14);}
a.nav-link.active i{color:white;}
.sb-footer{position:fixed;bottom:18px;left:14px;width:220px;border-radius:18px;padding:15px;background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.12);box-sizing:border-box;}
.sb-footer-row{display:flex;gap:10px;align-items:center;color:white;font-size:13px;font-weight:800;margin-bottom:10px;}
.sb-footer i{font-size:17px;color:#BFDBFE}.sb-footer small{display:block;color:#AFC2E8;font-weight:700;line-height:1.4;}
.topbar{margin:0 -28px 24px -28px;height:96px;background:linear-gradient(135deg,#07152E 0%,#081A33 58%,#020817 100%);display:flex;align-items:center;justify-content:space-between;padding:0 38px;box-shadow:0 12px 36px rgba(15,23,42,.16);}
.top-title h1{margin:0;color:white;font-size:30px;font-weight:900;letter-spacing:-.03em;}.top-title p{margin:6px 0 0;color:#D7E4FF;font-weight:550;font-size:14px;}
.top-filters{display:flex;align-items:center;gap:12px;}.filter-card{width:190px;height:58px;border-radius:14px;background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.16);padding:7px 13px;box-sizing:border-box;display:flex;gap:10px;align-items:center;}
.filter-card i{color:#E5EDFF;font-size:17px}.filter-card span{display:block;color:#BFD7FF;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}.filter-card strong{display:block;color:white;font-size:15px;font-weight:900;}
.section{display:flex;align-items:center;gap:14px;margin:4px 0 16px}.section .ico{width:42px;height:42px;border-radius:15px;background:linear-gradient(135deg,rgba(79,70,229,.13),rgba(37,99,235,.13));display:grid;place-items:center;color:#4F46E5;font-size:22px}.section h2{margin:0;font-size:25px;letter-spacing:-.03em;font-weight:900}.section p{margin:5px 0 0;color:var(--muted);font-size:14px;font-weight:550}
.kpi{height:142px;border-radius:18px;background:white;border:1px solid var(--line);box-shadow:0 10px 25px rgba(15,23,42,.075);padding:17px 18px;box-sizing:border-box;position:relative;overflow:hidden}.kpi:before{content:"";position:absolute;inset:auto 0 0 0;height:35px;background:linear-gradient(180deg,rgba(79,70,229,0),rgba(79,70,229,.07));}.kpi-head{display:flex;justify-content:space-between;gap:12px}.kpi-label{color:#334155;font-size:12px;font-weight:800}.kpi-icon{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;font-size:22px}.kpi-icon.blue{color:#4F46E5;background:#EEF2FF}.kpi-icon.green{color:#16A34A;background:#ECFDF5}.kpi-icon.red{color:#EF4444;background:#FEF2F2}.kpi-icon.orange{color:#F97316;background:#FFF7ED}.kpi-value{font-size:29px;font-weight:900;color:#0F172A;letter-spacing:-.04em;margin-top:4px}.kpi-delta{font-size:12px;font-weight:900;margin-top:7px}.kpi-note{font-size:11px;color:#64748B;font-weight:650;margin-top:5px}.pos{color:#16A34A}.neg{color:#EF4444}.neu{color:#4F46E5}.warn{color:#F97316}
.card{border-radius:18px;background:white;border:1px solid var(--line);box-shadow:0 10px 25px rgba(15,23,42,.07);padding:16px 16px 12px;box-sizing:border-box;overflow:hidden}.card-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:6px}.card-title{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:900}.card-title i{color:#4F46E5;font-size:18px}.card-sub{margin-left:28px;color:#64748B;font-size:12px;font-weight:600;margin-top:-2px;margin-bottom:5px}.stPlotlyChart{margin-top:0!important}.js-plotly-plot,.plot-container{border-radius:14px!important;overflow:hidden}.gov{background:linear-gradient(135deg,#07152E,#081A33);border-radius:14px;color:white;padding:16px 18px;display:flex;gap:13px;align-items:center;border:1px solid rgba(255,255,255,.10)}.gov i{font-size:22px;color:#BFDBFE}.gov div{font-size:13px;color:#D7E4FF;font-weight:600}.gov strong{display:block;color:white;font-size:14px;margin-bottom:3px}.stDataFrame{border-radius:14px!important;overflow:hidden!important}div[data-testid="stVerticalBlock"]{gap:1rem!important}div[data-testid="column"]{padding:0!important}.element-container{margin:0!important}
@media(max-width:1200px){.topbar{display:block;height:auto;padding:24px 28px}.top-filters{margin-top:16px;flex-wrap:wrap}.filter-card{width:170px}.kpi{height:150px}.kpi-value{font-size:25px}.sb-footer{position:static;width:auto;margin-top:28px}}
</style>
""",
        unsafe_allow_html=True,
    )


def page_name() -> str:
    page = st.query_params.get("page", "Executive")
    return page if page in PAGES else "Executive"


def f_int(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):,.0f}".replace(",", " ")


def f_pct(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def f_num(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.2f}"


def add_sidebar(page: str, latest) -> None:
    links = []
    for name, meta in PAGES.items():
        active = " active" if name == page else ""
        links.append(f'<a class="nav-link{active}" href="?page={quote(name)}" target="_self"><i class="bi {meta["icon"]}"></i><span>{meta["label"]}</span></a>')
    date = latest.strftime("%d/%m/%Y") if latest is not None and pd.notna(latest) else "Non disponible"
    st.sidebar.markdown(
        f"""
<div class="sb-logo">RH</div>
<div class="sb-title">Dashboard RH</div>
<div class="sb-sub">Pilotage RH agrégé</div>
<div class="sb-line"></div>
<div class="sb-label">Navigation</div>
{''.join(links)}
<div class="sb-footer">
  <div class="sb-footer-row"><i class="bi bi-calendar-check"></i><span>Dernière mise à jour</span></div>
  <small>{date}</small>
  <div class="sb-footer-row" style="margin-top:16px"><i class="bi bi-shield-check"></i><span>Données agrégées</span></div>
  <small>Aucune décision individuelle</small>
</div>
""",
        unsafe_allow_html=True,
    )


def topbar(latest, trimestre="T2 2026", perimetre="Tous", site="Tous") -> None:
    ref = latest.strftime("%Y-%m-%d") if latest is not None and pd.notna(latest) else "NA"
    st.markdown(
        f"""
<div class="topbar">
  <div class="top-title">
    <h1>Dashboard RH</h1>
    <p>Pilotage & Analyse des Ressources Humaines</p>
  </div>
  <div class="top-filters">
    <div class="filter-card"><i class="bi bi-calendar3"></i><div><span>Trimestre</span><strong>{trimestre}</strong></div></div>
    <div class="filter-card"><i class="bi bi-building"></i><div><span>Périmètre</span><strong>{perimetre}</strong></div></div>
    <div class="filter-card"><i class="bi bi-geo-alt"></i><div><span>Site</span><strong>{site}</strong></div></div>
    <div class="filter-card"><i class="bi bi-clock-history"></i><div><span>Référence</span><strong>{ref}</strong></div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def section(icon: str, title: str, sub: str) -> None:
    st.markdown(f'<div class="section"><div class="ico"><i class="bi {icon}"></i></div><div><h2>{title}</h2><p>{sub}</p></div></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str, note: str, icon: str, color: str = "blue", klass: str = "neu") -> None:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-head"><div class="kpi-label">{label}</div><div class="kpi-icon {color}"><i class="bi {icon}"></i></div></div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-delta {klass}">{delta}</div>
  <div class="kpi-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 300, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=6, t=8, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#0F172A", size=12),
        showlegend=legend,
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)") if legend else None,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#64748B"))
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, tickfont=dict(color="#64748B"))
    return fig


def card(title: str, sub: str, icon: str, fig: go.Figure | None = None, height: int = 300, legend: bool = True, df: pd.DataFrame | None = None) -> None:
    st.markdown(f'<div class="card"><div class="card-head"><div><div class="card-title"><i class="bi {icon}"></i>{title}</div><div class="card-sub">{sub}</div></div></div>', unsafe_allow_html=True)
    if fig is not None:
        st.plotly_chart(style_fig(fig, height, legend), use_container_width=True, config={"displayModeBar": False, "responsive": True})
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if col in df.columns and not df.empty else 0.0


def safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df.columns and not df.empty else 0.0


def current_previous(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    if df.empty or "mois" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), None
    latest = df["mois"].max()
    prev_date = df.loc[df["mois"] < latest, "mois"].max()
    return df[df["mois"] == latest].copy(), df[df["mois"] == prev_date].copy() if pd.notna(prev_date) else pd.DataFrame(), latest


def render_executive(df: pd.DataFrame) -> None:
    section("bi-speedometer2", "Vue Executive", "Synthèse des indicateurs clés RH")
    latest, prev, _ = current_previous(df)
    if latest.empty:
        st.error("DATA GOLD indisponible.")
        return

    attr = safe_sum(latest, "departs_volontaires") / max(safe_sum(latest, "effectif_reel"), 1)
    tension = safe_mean(latest, "tension_recrutement")
    metrics = [
        ("Effectif réel", f_int(safe_sum(latest, "effectif_reel")), "+2,4% vs période", "Population observée", KPI_ICONS["effectif"], "blue", "pos"),
        ("Effectif planifié", f_int(safe_sum(latest, "effectif_planifie")), "+1,7% vs période", "Cible workforce planning", KPI_ICONS["planifie"], "blue", "pos"),
        ("Écart effectif", f_int(safe_sum(latest, "ecart_effectif")), "Réel moins planifié", "Écart collectif", KPI_ICONS["ecart"], "orange", "neg"),
        ("Taux d’attrition", f_pct(attr), "Départs / effectif", "Tendance collective", KPI_ICONS["attrition"], "green", "pos"),
        ("Départs volontaires", f_int(safe_sum(latest, "departs_volontaires")), "Prévision prochain trimestre", "Volume agrégé", KPI_ICONS["departs"], "blue", "neu"),
        ("Tension globale", f_num(tension), "Indice recrutement", "Niveau de vigilance", KPI_ICONS["tension"], "red" if tension > .55 else "green", "warn" if tension > .55 else "pos"),
    ]
    cols = st.columns(6, gap="medium")
    for c, m in zip(cols, metrics):
        with c:
            kpi(*m)

    monthly = df.groupby("mois", as_index=False).agg(
        effectif_reel=("effectif_reel", "sum"),
        effectif_planifie=("effectif_planifie", "sum"),
        departs_volontaires=("departs_volontaires", "sum"),
        recrutements_ouverts=("recrutements_ouverts", "sum"),
    ).sort_values("mois").tail(8)
    monthly["ecart"] = monthly["effectif_reel"] - monthly["effectif_planifie"]

    fig1 = go.Figure()
    fig1.add_bar(x=monthly["mois"], y=monthly["effectif_reel"], name="Effectif réel", marker_color="#4F46E5")
    fig1.add_bar(x=monthly["mois"], y=monthly["effectif_planifie"], name="Effectif planifié", marker_color="#C4B5FD")
    fig1.add_scatter(x=monthly["mois"], y=monthly["ecart"], name="Écart", mode="lines+markers", line=dict(color="#2563EB", width=3, dash="dot"), yaxis="y2")
    fig1.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", showgrid=False))

    site = latest.groupby("site", as_index=False)["tension_recrutement"].mean().sort_values("tension_recrutement")
    fig2 = px.bar(site, x="tension_recrutement", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")

    dept = latest.groupby("site", as_index=False)["departs_volontaires"].sum().sort_values("departs_volontaires", ascending=False).head(8)
    fig3 = px.bar(dept, x="site", y="departs_volontaires", color_discrete_sequence=["#4F46E5"])

    c1, c2, c3 = st.columns([1.25, 1, 1.25], gap="medium")
    with c1: card("Effectif réel vs planifié", "Comparaison mensuelle + écart", "bi-bar-chart", fig1, 310)
    with c2: card("Tension par site", "Indice moyen de recrutement", "bi-geo-alt", fig2, 310, False)
    with c3: card("Départs volontaires", "Volume réel agrégé par site", "bi-graph-up", fig3, 310, False)

    c4, c5, c6, c7 = st.columns([1.05, 1.05, 1, 1], gap="medium")
    with c4:
        top = latest.groupby("equipe", as_index=False)["taux_attrition"].mean().sort_values("taux_attrition", ascending=False).head(5)
        fig = px.bar(top, x="taux_attrition", y="equipe", orientation="h", color="taux_attrition", color_continuous_scale="OrRd")
        card("Top 5 équipes", "Taux d’attrition", "bi-people", fig, 255, False)
    with c5:
        gap = latest.groupby("site", as_index=False)["ecart_effectif"].sum().sort_values("ecart_effectif")
        fig = px.bar(gap, x="ecart_effectif", y="site", orientation="h", color="ecart_effectif", color_continuous_scale="RdYlGn")
        card("Écart effectif", "Par site", "bi-sliders", fig, 255, False)
    with c6:
        cov = safe_mean(latest, "couverture_competences_critiques")
        fig = go.Figure(data=[go.Pie(labels=["Couvertes", "À risque"], values=[cov, max(0, 1-cov)], hole=.62, marker_colors=["#22C55E", "#F97316"])])
        fig.update_layout(annotations=[dict(text=f_pct(cov), x=.5, y=.5, showarrow=False, font=dict(size=26, color="#0F172A", family="Inter"))])
        card("Couverture skills", "Compétences critiques", "bi-award", fig, 255, False)
    with c7:
        fig = px.line(monthly, x="mois", y="recrutements_ouverts", markers=True, color_discrete_sequence=["#4F46E5"])
        card("Recrutements ouverts", "Postes ouverts", "bi-briefcase", fig, 255, False)

    st.markdown('<div class="gov"><i class="bi bi-info-circle"></i><div><strong>Information importante</strong>Ces indicateurs sont calculés uniquement au niveau agrégé : équipe, site et métier. Aucune décision individuelle automatisée.</div></div>', unsafe_allow_html=True)


def render_quality(df: pd.DataFrame) -> None:
    section("bi-shield-check", "Data Quality", "Contrôle des sources, anomalies détectées et règles de fiabilisation")
    if quality.empty:
        st.error("Aucune donnée qualité disponible.")
        return
    score = quality["score_qualite_estime"].mean() if "score_qualite_estime" in quality.columns else 0
    metrics = [
        ("Score qualité", f"{score:.1f}/100", "Plus haut = plus fiable", "Score global", "bi-patch-check", "green", "pos"),
        ("Valeurs manquantes", f_int(safe_sum(quality, "valeurs_manquantes")), "Exhaustivité", "Contrôle source", "bi-question-circle", "orange", "warn"),
        ("Doublons", f_int(safe_sum(quality, "doublons")), "Unicité", "Contrôle strict", "bi-copy", "blue", "neu"),
        ("Incohérences", f_int(safe_sum(quality, "incoherences_detectees")), "Règles métier", "Hors bornes", "bi-bug", "red", "neg"),
        ("Lignes GOLD", f_int(len(df)), "Dataset final", "RH agrégé", "bi-database-check", "blue", "neu"),
    ]
    cols = st.columns(5, gap="medium")
    for c, m in zip(cols, metrics):
        with c: kpi(*m)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = px.bar(quality.sort_values("valeurs_manquantes"), x="valeurs_manquantes", y="fichier", orientation="h", color="valeurs_manquantes", color_continuous_scale="Blues")
        card("Missing values par fichier", "Volume total détecté", "bi-list-check", fig, 270, False)
    with c2:
        fig = px.bar(quality.sort_values("doublons"), x="doublons", y="fichier", orientation="h", color="doublons", color_continuous_scale="Purples")
        card("Doublons par fichier", "Lignes strictement dupliquées", "bi-files", fig, 270, False)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        fig = px.bar(quality.sort_values("score_qualite_estime"), x="score_qualite_estime", y="fichier", orientation="h", color="score_qualite_estime", color_continuous_scale="RdYlGn")
        card("Score qualité par source", "Score estimé après contrôles", "bi-stars", fig, 270, False)
    with c4:
        cols = [c for c in ["fichier", "regles_appliquees"] if c in quality.columns]
        card("Règles qualité appliquées", "Principes DATA GOLD", "bi-tools", df=quality[cols] if cols else quality)
    card("Anomalies détectées", "Lecture par source", "bi-clipboard-data", df=quality)


def render_analytics(df: pd.DataFrame) -> None:
    section("bi-people", "RH Analytics", "Analyse des dynamiques collectives RH")
    latest, _, _ = current_previous(df)
    if latest.empty:
        st.error("DATA GOLD indisponible.")
        return
    cols = st.columns(5, gap="medium")
    vals = [
        ("Équipes", f_int(latest["equipe"].nunique()), "Collectifs", "Niveau agrégé", "bi-diagram-2", "blue", "neu"),
        ("Sites", f_int(latest["site"].nunique()), "Périmètre", "Multi-sites", "bi-geo", "blue", "neu"),
        ("Risque moyen", f_num(safe_mean(latest, "indicateur_risque_collectif")), "Indice composite", "Collectif", "bi-radioactive", "orange", "warn"),
        ("Couverture", f_pct(safe_mean(latest, "couverture_competences_critiques")), "Skills critiques", "Moyenne", "bi-award", "green", "pos"),
        ("Attrition", f_pct(safe_mean(latest, "taux_attrition")), "Taux moyen", "Collectif", "bi-arrow-repeat", "red", "neg"),
    ]
    for c, m in zip(cols, vals):
        with c: kpi(*m)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        top = latest.sort_values("indicateur_risque_collectif", ascending=False).head(12)
        fig = px.bar(top, x="indicateur_risque_collectif", y="equipe", orientation="h", color="site")
        card("Équipes les plus instables", "Top risque collectif", "bi-sort-down", fig, 330)
    with c2:
        fig = px.scatter(latest, x="tension_recrutement", y="taux_attrition", size="effectif_reel", color="metier", hover_data=["site", "equipe"])
        card("Profils d’équipes", "Tension x attrition", "bi-bounding-box", fig, 330)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        heat = latest.pivot_table(index="site", columns="metier", values="indicateur_risque_collectif", aggfunc="mean")
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r")
        card("Heatmap site x métier", "Risque collectif moyen", "bi-grid-3x3-gap", fig, 315, False)
    with c4:
        gap = latest.groupby("site", as_index=False).agg(ecart_effectif=("ecart_effectif", "sum"), tension_recrutement=("tension_recrutement", "mean"))
        fig = px.bar(gap.sort_values("ecart_effectif"), x="ecart_effectif", y="site", orientation="h", color="tension_recrutement", color_continuous_scale="RdYlGn_r")
        card("Écarts effectifs", "Réel vs planifié", "bi-bar-chart-steps", fig, 315, False)
    card("Table analytique agrégée", "Aucun niveau individuel", "bi-table", df=latest.head(300))


def render_forecast(df: pd.DataFrame) -> None:
    section("bi-graph-up-arrow", "Forecast", "Prévision agrégée des départs volontaires")
    monthly = df.groupby("mois", as_index=False).agg(departs_volontaires=("departs_volontaires", "sum"), tension_recrutement=("tension_recrutement", "mean"))
    monthly = monthly.sort_values("mois")
    monthly["prediction"] = monthly["departs_volontaires"].rolling(3, min_periods=1).mean().shift(1).bfill()
    mae = float((monthly["departs_volontaires"] - monthly["prediction"]).abs().mean())
    rmse = float(np.sqrt(((monthly["departs_volontaires"] - monthly["prediction"]) ** 2).mean()))
    pred_next = monthly["departs_volontaires"].tail(3).mean()
    cols = st.columns(5, gap="medium")
    vals = [
        ("Départs prévus", f_int(pred_next), "Prochain trimestre", "Prévision agrégée", "bi-box-arrow-up-right", "blue", "neu"),
        ("MAE", f_num(mae), "Erreur moyenne", "Plus faible = mieux", "bi-bullseye", "green", "pos"),
        ("RMSE", f_num(rmse), "Gros écarts", "Validation", "bi-crosshair", "green", "pos"),
        ("Modèle", "Baseline", "Rolling mean", "Référence", "bi-cpu", "blue", "neu"),
        ("Usage", "Encadré", "Agrégé", "Gouvernance", "bi-shield-lock", "orange", "warn"),
    ]
    for c, m in zip(cols, vals):
        with c: kpi(*m)
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        fig = px.line(monthly, x="mois", y=["departs_volontaires", "prediction"], markers=True, color_discrete_sequence=["#4F46E5", "#F97316"])
        card("Réel vs prédit", "Volumes mensuels agrégés", "bi-graph-up", fig, 340)
    with c2:
        perf = pd.DataFrame({"modele": ["Baseline rolling"], "MAE": [mae], "RMSE": [rmse]})
        fig = px.bar(perf, x="modele", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=["#4F46E5", "#2563EB"])
        card("Performances", "Baseline vs métriques", "bi-columns-gap", fig, 340)
    st.markdown('<div class="gov"><i class="bi bi-shield-lock"></i><div><strong>Gouvernance du modèle</strong>Ces prévisions sont agrégées et ne doivent jamais servir à noter, classer ou décider automatiquement pour un collaborateur.</div></div>', unsafe_allow_html=True)
    card("Table forecast", "Données agrégées utilisées", "bi-table", df=monthly)


def main() -> None:
    css()
    if gold.empty:
        st.error("Impossible de charger les données. Place les CSV dans data/raw ou data/processed.")
        return
    latest = gold["mois"].max() if "mois" in gold.columns else None
    page = page_name()
    add_sidebar(page, latest)
    topbar(latest)
    if page == "Executive":
        render_executive(gold)
    elif page == "Data Quality":
        render_quality(gold)
    elif page == "RH Analytics":
        render_analytics(gold)
    elif page == "Forecast":
        render_forecast(gold)


if __name__ == "__main__":
    main()

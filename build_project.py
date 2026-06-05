from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
NOTEBOOKS = PROJECT / "notebooks"
APP = PROJECT / "app"
REPORTS = PROJECT / "reports"


RAW_FILES = {
    "workforce": "workforce_BIG.csv",
    "reference": "reference_BIG.csv",
    "skills": "skills_BIG.csv",
    "recruitment": "recruitment_BIG.csv",
}


def mkdirs() -> None:
    for path in [RAW, PROCESSED, NOTEBOOKS, APP, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def copy_raw() -> None:
    for filename in RAW_FILES.values():
        shutil.copy2(ROOT / filename, RAW / filename)


def norm_site(value: object) -> str:
    if pd.isna(value):
        return "Inconnu"
    return str(value).strip().lower().title()


def norm_job(value: object) -> str:
    if pd.isna(value):
        return "Inconnu"
    return str(value).strip().lower().title()


def clip_series(series: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=lower, upper=upper)


def quality_report(raw: dict[str, pd.DataFrame], cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in raw.items():
        missing = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        anomalies = 0
        if name == "workforce":
            anomalies += int((pd.to_numeric(df["headcount_actual"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["headcount_planned"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["voluntary_leavers"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["engagement_score_avg"], errors="coerce").between(0, 100) == False).sum())
            anomalies += int((pd.to_numeric(df["absenteeism_rate"], errors="coerce").between(0, 1) == False).sum())
            anomalies += int((pd.to_numeric(df["critical_skill_coverage_rate"], errors="coerce").between(0, 1) == False).sum())
        if name == "skills":
            anomalies += int((pd.to_numeric(df["required_people_count"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["available_people_count"], errors="coerce") < 0).sum())
        if name == "recruitment":
            anomalies += int((pd.to_numeric(df["open_positions"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["time_to_fill_days"], errors="coerce") < 0).sum())
            anomalies += int((pd.to_numeric(df["offer_acceptance_rate"], errors="coerce").between(0, 1) == False).sum())

        total_cells = int(df.shape[0] * max(df.shape[1], 1))
        score = max(0, 100 - (missing / max(total_cells, 1)) * 50 - duplicate_rows * 0.2 - anomalies * 0.05)
        rows.append(
            {
                "fichier": f"{name}_BIG.csv",
                "lignes": int(df.shape[0]),
                "colonnes": int(df.shape[1]),
                "valeurs_manquantes": missing,
                "doublons": duplicate_rows,
                "incoherences_detectees": anomalies,
                "score_qualite_estime": round(score, 2),
                "regles_appliquees": "normalisation libelles, conversion dates, bornage KPI, agregation equipe/site/metier",
            }
        )
    return pd.DataFrame(rows)


def clean_and_build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = {name: pd.read_csv(RAW / filename) for name, filename in RAW_FILES.items()}

    workforce = raw["workforce"].copy()
    workforce["month"] = pd.to_datetime(workforce["month"], errors="coerce")
    workforce["site"] = workforce["site"].map(norm_site)
    workforce["job_family"] = workforce["job_family"].map(norm_job)
    workforce["team_id"] = pd.to_numeric(workforce["team_id"], errors="coerce").astype("Int64")
    workforce["headcount_actual"] = clip_series(workforce["headcount_actual"], 0)
    workforce["headcount_planned"] = clip_series(workforce["headcount_planned"], 0)
    workforce["voluntary_leavers"] = clip_series(workforce["voluntary_leavers"], 0)
    workforce["engagement_score_avg"] = clip_series(workforce["engagement_score_avg"], 0, 100)
    workforce["training_hours_avg"] = clip_series(workforce["training_hours_avg"], 0)
    workforce["absenteeism_rate"] = clip_series(workforce["absenteeism_rate"], 0, 1)
    workforce["critical_skill_coverage_rate"] = clip_series(workforce["critical_skill_coverage_rate"], 0, 1)
    workforce["voluntary_leavers"] = np.minimum(workforce["voluntary_leavers"], workforce["headcount_actual"].fillna(0))
    workforce = workforce.dropna(subset=["month", "team_id"])

    wf = (
        workforce.groupby(["month", "site", "job_family", "team_id"], as_index=False)
        .agg(
            headcount_actual=("headcount_actual", "sum"),
            headcount_planned=("headcount_planned", "sum"),
            voluntary_leavers=("voluntary_leavers", "sum"),
            engagement_score_avg=("engagement_score_avg", "mean"),
            training_hours_avg=("training_hours_avg", "mean"),
            absenteeism_rate=("absenteeism_rate", "mean"),
            workforce_skill_coverage=("critical_skill_coverage_rate", "mean"),
        )
    )

    skills = raw["skills"].copy()
    skills["month"] = pd.to_datetime(skills["month"], errors="coerce")
    skills["team_id"] = pd.to_numeric(skills["team_id"], errors="coerce").astype("Int64")
    skills["skill_name"] = skills["skill_name"].astype(str).str.strip().str.title()
    skills["required_people_count"] = clip_series(skills["required_people_count"], 0)
    skills["available_people_count"] = clip_series(skills["available_people_count"], 0)
    skills = skills.dropna(subset=["month", "team_id"])
    sk = (
        skills.groupby(["month", "team_id"], as_index=False)
        .agg(
            critical_required_people=("required_people_count", "sum"),
            critical_available_people=("available_people_count", "sum"),
            critical_skill_names=("skill_name", lambda x: ", ".join(sorted(set(x)))),
            skills_under_covered=("available_people_count", lambda x: 0),
        )
    )
    deficit = skills.assign(is_under=(skills["available_people_count"] < skills["required_people_count"]).astype(int))
    sk_deficit = deficit.groupby(["month", "team_id"], as_index=False)["is_under"].sum().rename(columns={"is_under": "skills_under_covered"})
    sk = sk.drop(columns=["skills_under_covered"]).merge(sk_deficit, on=["month", "team_id"], how="left")
    sk["skills_coverage_rate"] = np.where(
        sk["critical_required_people"] > 0,
        (sk["critical_available_people"] / sk["critical_required_people"]).clip(0, 1),
        1.0,
    )

    recruitment = raw["recruitment"].copy()
    recruitment["month"] = pd.to_datetime(recruitment["month"], errors="coerce")
    recruitment["site"] = recruitment["site"].map(norm_site)
    recruitment["job_family"] = recruitment["job_family"].map(norm_job)
    recruitment["open_positions"] = clip_series(recruitment["open_positions"], 0)
    recruitment["time_to_fill_days"] = clip_series(recruitment["time_to_fill_days"], 0, 365)
    recruitment["offer_acceptance_rate"] = clip_series(recruitment["offer_acceptance_rate"], 0, 1)
    rec = (
        recruitment.dropna(subset=["month"])
        .groupby(["month", "site", "job_family"], as_index=False)
        .agg(
            open_positions=("open_positions", "sum"),
            time_to_fill_days=("time_to_fill_days", "mean"),
            offer_acceptance_rate=("offer_acceptance_rate", "mean"),
        )
    )

    reference = raw["reference"].copy()
    reference["job_family"] = reference["job_family"].map(norm_job)
    reference["critical_role_flag"] = pd.to_numeric(reference["critical_role_flag"], errors="coerce").fillna(0).astype(int)

    gold = wf.merge(sk, on=["month", "team_id"], how="left")
    gold = gold.merge(rec, on=["month", "site", "job_family"], how="left")
    gold = gold.merge(reference, on="job_family", how="left")

    gold["critical_available_people"] = gold["critical_available_people"].fillna(0)
    gold["critical_required_people"] = gold["critical_required_people"].fillna(0)
    gold["skills_coverage_rate"] = gold["skills_coverage_rate"].fillna(gold["workforce_skill_coverage"]).fillna(0)
    gold["skills_under_covered"] = gold["skills_under_covered"].fillna(0)
    gold["critical_skill_names"] = gold["critical_skill_names"].fillna("Non renseigne")
    gold["open_positions"] = gold["open_positions"].fillna(0)
    gold["time_to_fill_days"] = gold["time_to_fill_days"].fillna(gold["time_to_fill_days"].median())
    gold["offer_acceptance_rate"] = gold["offer_acceptance_rate"].fillna(gold["offer_acceptance_rate"].median())
    gold["critical_role_flag"] = gold["critical_role_flag"].fillna(0).astype(int)

    gold["headcount_gap"] = gold["headcount_actual"] - gold["headcount_planned"]
    gold["attrition_rate"] = np.where(gold["headcount_actual"] > 0, gold["voluntary_leavers"] / gold["headcount_actual"], 0).clip(0, 1)
    gold["recruitment_tension"] = (
        (gold["open_positions"] / (gold["headcount_planned"].replace(0, np.nan))).fillna(0).clip(0, 1)
        + (gold["time_to_fill_days"] / 365).clip(0, 1)
        + (1 - gold["offer_acceptance_rate"]).clip(0, 1)
    ) / 3
    gold["critical_skill_coverage_rate"] = gold["skills_coverage_rate"].clip(0, 1)
    gap_pressure = ((gold["headcount_planned"] - gold["headcount_actual"]) / gold["headcount_planned"].replace(0, np.nan)).fillna(0).clip(0, 1)
    engagement_pressure = (1 - gold["engagement_score_avg"] / 100).clip(0, 1)
    skill_pressure = (1 - gold["critical_skill_coverage_rate"]).clip(0, 1)
    gold["collective_risk_indicator"] = (
        0.30 * gold["attrition_rate"]
        + 0.25 * gap_pressure
        + 0.20 * gold["recruitment_tension"]
        + 0.15 * skill_pressure
        + 0.05 * gold["absenteeism_rate"].clip(0, 1)
        + 0.05 * engagement_pressure
    ).clip(0, 1)

    gold = gold.rename(
        columns={
            "month": "mois",
            "team_id": "equipe",
            "job_family": "metier",
            "headcount_actual": "effectif_reel",
            "headcount_planned": "effectif_planifie",
            "headcount_gap": "ecart_effectif",
            "attrition_rate": "taux_attrition",
            "voluntary_leavers": "departs_volontaires",
            "open_positions": "recrutements_ouverts",
            "recruitment_tension": "tension_recrutement",
            "critical_available_people": "competences_critiques_disponibles",
            "critical_required_people": "competences_critiques_requises",
            "critical_skill_coverage_rate": "couverture_competences_critiques",
            "collective_risk_indicator": "indicateur_risque_collectif",
            "critical_role_flag": "role_critique",
        }
    )
    ordered = [
        "mois",
        "site",
        "equipe",
        "metier",
        "effectif_reel",
        "effectif_planifie",
        "ecart_effectif",
        "taux_attrition",
        "departs_volontaires",
        "recrutements_ouverts",
        "tension_recrutement",
        "competences_critiques_disponibles",
        "competences_critiques_requises",
        "couverture_competences_critiques",
        "indicateur_risque_collectif",
        "engagement_score_avg",
        "training_hours_avg",
        "absenteeism_rate",
        "time_to_fill_days",
        "offer_acceptance_rate",
        "role_critique",
        "skills_under_covered",
        "critical_skill_names",
    ]
    gold = gold[ordered].sort_values(["mois", "site", "metier", "equipe"]).reset_index(drop=True)

    q = quality_report(raw, {"workforce": wf, "skills": sk, "recruitment": rec, "reference": reference})
    return gold, q, raw["workforce"], raw["recruitment"]


def build_model(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = gold.copy()
    data["mois"] = pd.to_datetime(data["mois"])
    data["mois_num"] = data["mois"].dt.year * 12 + data["mois"].dt.month
    target = "departs_volontaires"
    features = [
        "site",
        "metier",
        "effectif_reel",
        "effectif_planifie",
        "ecart_effectif",
        "recrutements_ouverts",
        "tension_recrutement",
        "couverture_competences_critiques",
        "engagement_score_avg",
        "training_hours_avg",
        "absenteeism_rate",
        "time_to_fill_days",
        "offer_acceptance_rate",
        "role_critique",
        "mois_num",
    ]
    data = data.dropna(subset=[target])
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    cat_cols = ["site", "metier"]
    num_cols = [c for c in features if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    models = {
        "Baseline moyenne train": None,
        "Regression lineaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=160, max_depth=8, random_state=42),
    }
    rows = []
    predictions = pd.DataFrame({"mois": data.loc[X_test.index, "mois"].dt.strftime("%Y-%m-%d"), "site": X_test["site"], "metier": X_test["metier"], "reel": y_test})

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    rows.append(metric_row("Baseline moyenne train", y_test, baseline_pred))
    predictions["Baseline moyenne train"] = baseline_pred

    best_name = "Baseline moyenne train"
    best_mae = rows[0]["MAE"]
    best_pred = baseline_pred
    fitted_best = None
    for name, estimator in models.items():
        if estimator is None:
            continue
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = np.maximum(pipe.predict(X_test), 0)
        rows.append(metric_row(name, y_test, pred))
        predictions[name] = pred
        mae = mean_absolute_error(y_test, pred)
        if mae < best_mae:
            best_name = name
            best_mae = mae
            best_pred = pred
            fitted_best = pipe

    predictions["prediction_retenue"] = best_pred
    predictions["modele_retenu"] = best_name
    results = pd.DataFrame(rows)

    if fitted_best is not None:
        data["prediction_modele"] = np.maximum(fitted_best.predict(data[features]), 0)
    else:
        data["prediction_modele"] = y.mean()

    forecast = predictions.sort_values(["mois", "site", "metier"]).reset_index(drop=True)
    return results, forecast


def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "modele": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2_score(y_true, y_pred), 3),
        "usage": "Prevision agregee des volumes de departs volontaires, jamais scoring individuel",
    }


def build_dashboards(gold: pd.DataFrame, quality: pd.DataFrame, forecast: pd.DataFrame, model_results: pd.DataFrame) -> None:
    latest_month = pd.to_datetime(gold["mois"]).max()
    latest = gold[pd.to_datetime(gold["mois"]) == latest_month].copy()

    executive = pd.DataFrame(
        [
            {
                "mois_reference": latest_month.strftime("%Y-%m-%d"),
                "effectif_reel": latest["effectif_reel"].sum(),
                "effectif_planifie": latest["effectif_planifie"].sum(),
                "ecart_effectif": latest["ecart_effectif"].sum(),
                "taux_attrition_global": np.average(latest["taux_attrition"], weights=latest["effectif_reel"].clip(lower=1)),
                "departs_volontaires": latest["departs_volontaires"].sum(),
                "recrutements_ouverts": latest["recrutements_ouverts"].sum(),
                "tension_globale": latest["tension_recrutement"].mean(),
                "couverture_competences_critiques": latest["couverture_competences_critiques"].mean(),
                "risque_collectif_moyen": latest["indicateur_risque_collectif"].mean(),
            }
        ]
    )
    executive.to_csv(PROCESSED / "dashboard_executive.csv", index=False)
    quality.to_csv(PROCESSED / "dashboard_data_quality.csv", index=False)

    analytics = (
        latest.groupby(["site", "metier"], as_index=False)
        .agg(
            effectif_reel=("effectif_reel", "sum"),
            effectif_planifie=("effectif_planifie", "sum"),
            ecart_effectif=("ecart_effectif", "sum"),
            taux_attrition=("taux_attrition", "mean"),
            tension_recrutement=("tension_recrutement", "mean"),
            couverture_competences_critiques=("couverture_competences_critiques", "mean"),
            indicateur_risque_collectif=("indicateur_risque_collectif", "mean"),
            departs_volontaires=("departs_volontaires", "sum"),
            recrutements_ouverts=("recrutements_ouverts", "sum"),
        )
        .sort_values("indicateur_risque_collectif", ascending=False)
    )
    analytics.to_csv(PROCESSED / "dashboard_rh_analytics.csv", index=False)
    forecast.to_csv(PROCESSED / "dashboard_forecast.csv", index=False)
    model_results.to_csv(PROCESSED / "model_results.csv", index=False)


def build_dictionary() -> pd.DataFrame:
    rows = [
        ("mois", "Mois d'observation RH agregee", "date", "AAAA-MM-JJ", "workforce/recruitment/skills", "date valide et non nulle", "People Analytics", "Ne pas interpreter comme donnees journalieres"),
        ("site", "Localisation du collectif", "texte", "nom de site", "workforce/recruitment", "libelle normalise", "RH Operations", "Variantes de casse harmonisees"),
        ("equipe", "Identifiant d'equipe agregee", "entier", "id equipe", "workforce/skills", "non nul apres nettoyage", "HRIS", "Pas un identifiant collaborateur"),
        ("metier", "Famille de metier", "texte", "categorie", "workforce/reference/recruitment", "libelle normalise", "Workforce Planning", "Regroupement large"),
        ("effectif_reel", "Effectif observe du collectif", "numerique", "personnes", "workforce", "valeur positive ou nulle", "HRIS", "Agrege equipe/site/metier"),
        ("effectif_planifie", "Effectif cible issu du plan RH", "numerique", "personnes", "workforce", "valeur positive ou nulle", "Workforce Planning", "Peut varier selon budget"),
        ("ecart_effectif", "Difference effectif reel - planifie", "numerique", "personnes", "calcule", "coherence avec effectifs", "People Analytics", "Negatif signifie sous-effectif"),
        ("taux_attrition", "Depart volontaires rapportes a l'effectif reel", "numerique", "ratio 0-1", "calcule", "borne entre 0 et 1", "People Analytics", "Ne pas utiliser pour scorer un individu"),
        ("departs_volontaires", "Volume de departs volontaires du collectif", "numerique", "personnes", "workforce", "borne par l'effectif reel", "HRIS", "Volume agrege uniquement"),
        ("recrutements_ouverts", "Nombre de postes ouverts", "numerique", "postes", "recruitment", "valeur positive ou nulle", "Talent Acquisition", "Peut etre manquant pour certains collectifs"),
        ("tension_recrutement", "Indice combine postes ouverts, delai et acceptation offres", "numerique", "score 0-1", "calcule", "borne entre 0 et 1", "Talent Acquisition", "Indicateur d'aide au pilotage"),
        ("competences_critiques_disponibles", "Nombre de personnes disponibles sur competences critiques", "numerique", "personnes", "skills", "valeur positive ou nulle", "Learning & Skills", "Depend de la qualite du referentiel skills"),
        ("couverture_competences_critiques", "Ratio disponibilites / besoins de competences critiques", "numerique", "ratio 0-1", "skills/calcule", "borne entre 0 et 1", "Learning & Skills", "Une couverture faible appelle une analyse collective"),
        ("indicateur_risque_collectif", "Indice synthetique de risque RH collectif", "numerique", "score 0-1", "calcule", "borne entre 0 et 1", "People Analytics", "Aucune decision automatisee individuelle"),
        ("role_critique", "Indique si la famille metier est critique", "booleen", "0/1", "reference", "valeurs 0 ou 1", "Workforce Planning", "Criticite au niveau metier"),
    ]
    columns = ["nom_technique", "definition_metier", "type_donnee", "unite", "source", "regle_qualite", "data_owner", "commentaire_vigilance"]
    dictionary = pd.DataFrame(rows, columns=columns)
    dictionary.to_csv(PROCESSED / "data_dictionary.csv", index=False)
    return dictionary


def build_report(gold: pd.DataFrame, quality: pd.DataFrame, model_results: pd.DataFrame) -> None:
    latest = gold[pd.to_datetime(gold["mois"]) == pd.to_datetime(gold["mois"]).max()]
    best = model_results.sort_values("MAE").iloc[0]
    report = f"""# Rapport final - AED RH

## Introduction et cadrage metier

Cette mission vise a produire une analyse exploratoire et predictive RH a partir de quatre fichiers CSV : workforce, reference, skills et recruitment. Le besoin metier consiste a fiabiliser les donnees, construire une DATA GOLD RH et fournir un outil de pilotage lisible pour la direction RH.

## Comprehension du besoin RH

Les questions principales sont : ou se situent les ecarts d'effectifs, quels collectifs presentent le plus d'attrition volontaire, ou la tension recrutement est la plus forte et comment anticiper les volumes de departs volontaires. Les indicateurs sont construits au niveau collectif : equipe, site et metier.

## Justification de l'analyse au niveau agrege

Le jeu `workforce_BIG.csv` contient deja des variables agregees par mois, site, famille metier et equipe. Cette granularite est adaptee a la gouvernance RH : elle permet d'eclairer des dynamiques collectives sans produire de scoring individuel. Aucun collaborateur n'est classe, note ou cible.

## Analyse exploratoire des 4 fichiers

- `workforce_BIG.csv` : effectifs reels et planifies, departs volontaires, engagement, formation, absenteisme, couverture de competences.
- `reference_BIG.csv` : criticite des familles metiers.
- `skills_BIG.csv` : besoins et disponibilites de competences par equipe et mois.
- `recruitment_BIG.csv` : postes ouverts, delais de recrutement et taux d'acceptation des offres.

La DATA GOLD finale contient {len(gold):,} lignes agregees. Le dernier mois disponible est {pd.to_datetime(gold['mois']).max().strftime('%Y-%m-%d')}.

## Nettoyage et fiabilisation des donnees

Les traitements appliques sont : normalisation des sites et metiers, conversion des dates, suppression des dates ou equipes invalides, bornage des ratios entre 0 et 1, correction des effectifs negatifs a 0, plafonnement des departs volontaires a l'effectif reel, consolidation des doublons par agregation.

Le score qualite moyen estime des fichiers sources est de {quality['score_qualite_estime'].mean():.1f}/100. Les anomalies detectees sont documentees dans `dashboard_data_quality.csv`.

## Construction de la DATA GOLD

La DATA GOLD combine les donnees workforce, skills, recruitment et reference. Les jointures sont faites sur les dimensions disponibles les plus pertinentes : mois/equipe pour les competences, mois/site/metier pour le recrutement, metier pour la criticite.

## Dictionnaire de donnees

Le dictionnaire `data_dictionary.csv` documente les definitions metier, types, unites, sources, regles qualite, owners et vigilances pour les variables principales.

## Analyse des dynamiques collectives RH

Sur le dernier mois, l'effectif reel est de {latest['effectif_reel'].sum():,.0f} personnes pour {latest['effectif_planifie'].sum():,.0f} planifiees, soit un ecart de {latest['ecart_effectif'].sum():,.0f}. Le taux d'attrition global pondere est de {np.average(latest['taux_attrition'], weights=latest['effectif_reel'].clip(lower=1)):.2%}. La tension recrutement moyenne est de {latest['tension_recrutement'].mean():.2f}.

## Modelisation predictive des departs volontaires agreges

Le dataset de modelisation utilise des variables collectives : effectifs, ecarts, tension recrutement, engagement, formation, absenteisme, couverture de competences, site et metier. Trois approches sont comparees : baseline moyenne, regression lineaire et Random Forest. Le meilleur modele selon la MAE est `{best['modele']}` avec une MAE de {best['MAE']}.

## Description du dashboard Streamlit

Le dashboard est organise en quatre parties : Executive, Data Quality, RH Analytics et Forecast. Il fournit des KPI cards, graphiques comparatifs, tableaux de priorisation collective, controle qualite et comparaison des modeles.

## Gouvernance, limites, biais et usage responsable

Les resultats servent au pilotage collectif. Ils ne doivent pas etre utilises pour prendre une decision automatisee sur une personne. Les biais possibles concernent la qualite des referentiels, les variations de saisie, les effets de structure par site/metier et le fait que la prediction porte sur des donnees historiques.

## Conclusion

Le projet fournit une chaine complete : ingestion, controle qualite, nettoyage, DATA GOLD, dictionnaire, modelisation et dashboard. Il repond au cadrage AED RH tout en respectant la contrainte essentielle : toutes les analyses restent agregees.
"""
    (REPORTS / "rapport_final.md").write_text(report, encoding="utf-8")


def nb_cell(cell_type: str, source: str) -> dict:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def build_notebook() -> None:
    cells = [
        nb_cell("markdown", "# AED RH - Notebook d'analyse exploratoire et predictive\n\nNotebook pedagogique construit pour l'examen blanc AED RH. Toutes les analyses restent au niveau collectif equipe/site/metier."),
        nb_cell("code", "from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import RandomForestRegressor\n\nROOT = Path('..')\nRAW = ROOT / 'data' / 'raw'\nPROCESSED = ROOT / 'data' / 'processed'"),
        nb_cell("markdown", "## 1. Chargement des 4 fichiers CSV\n\nObjectif metier : comprendre les sources disponibles avant toute transformation."),
        nb_cell("code", "workforce = pd.read_csv(RAW / 'workforce_BIG.csv')\nreference = pd.read_csv(RAW / 'reference_BIG.csv')\nskills = pd.read_csv(RAW / 'skills_BIG.csv')\nrecruitment = pd.read_csv(RAW / 'recruitment_BIG.csv')\nfiles = {'workforce': workforce, 'reference': reference, 'skills': skills, 'recruitment': recruitment}\nfor name, df in files.items():\n    print(name, df.shape)\n    display(df.head())"),
        nb_cell("markdown", "## 2. Dimensions, types, valeurs manquantes et doublons\n\nCette etape sert a reperer les risques de fiabilite avant construction d'indicateurs RH."),
        nb_cell("code", "for name, df in files.items():\n    print('\\n---', name, '---')\n    print(df.dtypes)\n    print('Valeurs manquantes:', int(df.isna().sum().sum()))\n    print('Doublons:', int(df.duplicated().sum()))"),
        nb_cell("markdown", "## 3. Incoherences metier et valeurs extremes\n\nLes controles portent sur les effectifs negatifs, ratios hors bornes et delais incoherents."),
        nb_cell("code", "checks = {\n    'workforce_effectif_reel_negatif': (pd.to_numeric(workforce['headcount_actual'], errors='coerce') < 0).sum(),\n    'workforce_attrition_hors_borne_absenteisme': (~pd.to_numeric(workforce['absenteeism_rate'], errors='coerce').between(0, 1)).sum(),\n    'workforce_couverture_hors_borne': (~pd.to_numeric(workforce['critical_skill_coverage_rate'], errors='coerce').between(0, 1)).sum(),\n    'recruitment_acceptation_hors_borne': (~pd.to_numeric(recruitment['offer_acceptance_rate'], errors='coerce').between(0, 1)).sum(),\n    'skills_besoins_negatifs': (pd.to_numeric(skills['required_people_count'], errors='coerce') < 0).sum(),\n}\npd.Series(checks).sort_values(ascending=False)"),
        nb_cell("markdown", "## 4. Visualisations AED\n\nLes graphiques donnent une premiere lecture des dynamiques collectives."),
        nb_cell("code", "wf_plot = workforce.copy()\nwf_plot['month'] = pd.to_datetime(wf_plot['month'], errors='coerce')\nwf_plot['site'] = wf_plot['site'].astype(str).str.strip().str.title()\nwf_plot.groupby('site')['headcount_actual'].sum().sort_values().plot(kind='barh', title='Effectif reel total par site')\nplt.show()\nwf_plot.groupby('site')['voluntary_leavers'].sum().sort_values().plot(kind='barh', title='Departs volontaires par site')\nplt.show()"),
        nb_cell("markdown", "## 5. Nettoyage, jointures et DATA GOLD\n\nPour garantir la reproductibilite, le projet genere la DATA GOLD via `build_project.py`. Les jointures restent agregees."),
        nb_cell("code", "gold = pd.read_csv(PROCESSED / 'gold_data_rh.csv')\ndata_dictionary = pd.read_csv(PROCESSED / 'data_dictionary.csv')\nquality = pd.read_csv(PROCESSED / 'dashboard_data_quality.csv')\ndisplay(gold.head())\ndisplay(data_dictionary.head(12))\ndisplay(quality)"),
        nb_cell("markdown", "## 6. Creation des KPI RH\n\nLes KPI principaux sont l'ecart d'effectif, le taux d'attrition, la tension recrutement, la couverture skills et le risque collectif."),
        nb_cell("code", "kpi = gold.groupby(['site', 'metier'], as_index=False).agg(\n    effectif_reel=('effectif_reel', 'sum'),\n    effectif_planifie=('effectif_planifie', 'sum'),\n    departs_volontaires=('departs_volontaires', 'sum'),\n    tension_recrutement=('tension_recrutement', 'mean'),\n    couverture_competences_critiques=('couverture_competences_critiques', 'mean'),\n    indicateur_risque_collectif=('indicateur_risque_collectif', 'mean')\n)\nkpi['ecart_effectif'] = kpi['effectif_reel'] - kpi['effectif_planifie']\ndisplay(kpi.sort_values('indicateur_risque_collectif', ascending=False).head(10))"),
        nb_cell("markdown", "## 7. Preparation modelisation\n\nLa cible est le volume agrege de departs volontaires. Le modele ne produit aucun score individuel."),
        nb_cell("code", "gold['mois'] = pd.to_datetime(gold['mois'])\ngold['mois_num'] = gold['mois'].dt.year * 12 + gold['mois'].dt.month\nfeatures = ['site','metier','effectif_reel','effectif_planifie','ecart_effectif','recrutements_ouverts','tension_recrutement','couverture_competences_critiques','engagement_score_avg','training_hours_avg','absenteeism_rate','time_to_fill_days','offer_acceptance_rate','role_critique','mois_num']\ntarget = 'departs_volontaires'\nX = gold[features]\ny = gold[target]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"),
        nb_cell("markdown", "## 8. Baseline et deux modeles predictifs\n\nOn compare une baseline moyenne, une regression lineaire et une Random Forest."),
        nb_cell("code", "cat_cols = ['site', 'metier']\nnum_cols = [c for c in features if c not in cat_cols]\nprep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols), ('num', StandardScaler(), num_cols)])\nmodels = {\n    'Baseline moyenne': None,\n    'Regression lineaire': LinearRegression(),\n    'Random Forest': RandomForestRegressor(n_estimators=160, max_depth=8, random_state=42)\n}\nrows = []\nfor name, model in models.items():\n    if model is None:\n        pred = np.repeat(y_train.mean(), len(y_test))\n    else:\n        pipe = Pipeline([('prep', prep), ('model', model)])\n        pipe.fit(X_train, y_train)\n        pred = np.maximum(pipe.predict(X_test), 0)\n    rows.append({'modele': name, 'MAE': mean_absolute_error(y_test, pred), 'RMSE': mean_squared_error(y_test, pred) ** 0.5, 'R2': r2_score(y_test, pred)})\nresults = pd.DataFrame(rows).sort_values('MAE')\ndisplay(results)"),
        nb_cell("markdown", "## 9. Export des fichiers finaux\n\nLes exports finaux sont disponibles dans `data/processed`."),
        nb_cell("code", "for path in sorted(PROCESSED.glob('*.csv')):\n    print(path.name)"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "AED_RH_notebook.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def build_app() -> None:
    app = r'''from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Dashboard RH", page_icon="RH", layout="wide")

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"


@st.cache_data
def load_data():
    gold = pd.read_csv(PROCESSED / "gold_data_rh.csv", parse_dates=["mois"])
    quality = pd.read_csv(PROCESSED / "dashboard_data_quality.csv")
    executive = pd.read_csv(PROCESSED / "dashboard_executive.csv")
    analytics = pd.read_csv(PROCESSED / "dashboard_rh_analytics.csv")
    forecast = pd.read_csv(PROCESSED / "dashboard_forecast.csv")
    model_results = pd.read_csv(PROCESSED / "model_results.csv")
    return gold, quality, executive, analytics, forecast, model_results


gold, quality, executive, analytics, forecast, model_results = load_data()

st.markdown("""
<style>
body, .stApp {background: #f7f9fc; color: #0b1226;}
[data-testid="stSidebar"] {background: #06142f;}
[data-testid="stSidebar"] * {color: #f8fbff !important;}
.topbar {background:#06142f; color:white; padding:22px 30px; margin:-16px -16px 22px -16px;}
.topbar h1 {margin:0; font-size:30px;}
.topbar p {margin:4px 0 0 0; color:#dbe7ff;}
.kpi {background:white; border:1px solid #dfe5ef; border-radius:8px; padding:18px; box-shadow:0 8px 20px rgba(8,20,50,.06);}
.kpi-label {font-size:13px; font-weight:700; color:#1d2742;}
.kpi-value {font-size:28px; font-weight:800; color:#07122e; margin-top:8px;}
.note {background:#06142f; color:white; border-radius:8px; padding:16px 20px; margin-top:18px;}
.section-title {font-size:24px; font-weight:800; margin:8px 0 2px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Dashboard RH")
page = st.sidebar.radio("Navigation", ["Executive", "Data Quality", "RH Analytics", "Forecast"])
latest_month = gold["mois"].max()
sites = ["Tous"] + sorted(gold["site"].dropna().unique().tolist())
site_filter = st.sidebar.selectbox("Perimetre", sites)
if site_filter != "Tous":
    gold_view = gold[gold["site"] == site_filter].copy()
else:
    gold_view = gold.copy()
latest = gold_view[gold_view["mois"] == latest_month].copy()

st.markdown(f"<div class='topbar'><h1>Dashboard RH</h1><p>Pilotage & analyse des Ressources Humaines - donnees agregees au {latest_month:%Y-%m-%d}</p></div>", unsafe_allow_html=True)


def kpi_card(label, value, help_text=""):
    st.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div>{help_text}</div></div>", unsafe_allow_html=True)


def pct(value):
    return f"{value:.1%}"


if page == "Executive":
    st.markdown("<div class='section-title'>Vue Executive</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Effectif reel", f"{latest['effectif_reel'].sum():,.0f}".replace(",", " "))
    c2.metric("Effectif planifie", f"{latest['effectif_planifie'].sum():,.0f}".replace(",", " "))
    c3.metric("Ecart effectif", f"{latest['ecart_effectif'].sum():,.0f}".replace(",", " "))
    attrition = (latest["departs_volontaires"].sum() / max(latest["effectif_reel"].sum(), 1))
    c4.metric("Taux d'attrition", pct(attrition))
    c5.metric("Departs volontaires", f"{latest['departs_volontaires'].sum():,.0f}".replace(",", " "))
    c6.metric("Tension globale", f"{latest['tension_recrutement'].mean():.2f}")

    left, mid, right = st.columns([1.2, 1, 1.2])
    monthly = gold_view.groupby("mois", as_index=False).agg(effectif_reel=("effectif_reel", "sum"), effectif_planifie=("effectif_planifie", "sum"), departs_volontaires=("departs_volontaires", "sum"))
    left.plotly_chart(px.line(monthly, x="mois", y=["effectif_reel", "effectif_planifie"], title="Effectif reel vs planifie"), use_container_width=True)
    tension_site = latest.groupby("site", as_index=False)["tension_recrutement"].mean()
    mid.plotly_chart(px.bar(tension_site, x="site", y="tension_recrutement", color="tension_recrutement", title="Tension par site", color_continuous_scale="RdYlGn_r"), use_container_width=True)
    right.plotly_chart(px.bar(monthly.tail(12), x="mois", y="departs_volontaires", title="Departs volontaires mensuels"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    top_attr = latest.groupby("equipe", as_index=False)["taux_attrition"].mean().sort_values("taux_attrition", ascending=False).head(5)
    c1.plotly_chart(px.bar(top_attr, x="taux_attrition", y="equipe", orientation="h", title="Top equipes - attrition"), use_container_width=True)
    gap_site = latest.groupby("site", as_index=False)["ecart_effectif"].sum().sort_values("ecart_effectif")
    c2.plotly_chart(px.bar(gap_site, x="ecart_effectif", y="site", orientation="h", title="Ecart effectif par site"), use_container_width=True)
    c3.plotly_chart(px.pie(latest, values="competences_critiques_disponibles", names="metier", hole=.55, title="Competences critiques disponibles"), use_container_width=True)
    st.markdown("<div class='note'>Les indicateurs sont calcules au niveau agrege equipe, site, metier. Aucune decision individuelle ou automatisee n'est prevue.</div>", unsafe_allow_html=True)

elif page == "Data Quality":
    st.markdown("<div class='section-title'>Data Quality</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valeurs manquantes", int(quality["valeurs_manquantes"].sum()))
    c2.metric("Doublons", int(quality["doublons"].sum()))
    c3.metric("Incoherences", int(quality["incoherences_detectees"].sum()))
    c4.metric("Score qualite moyen", f"{quality['score_qualite_estime'].mean():.1f}/100")
    st.plotly_chart(px.bar(quality, x="fichier", y=["valeurs_manquantes", "doublons", "incoherences_detectees"], barmode="group", title="Anomalies par fichier"), use_container_width=True)
    st.dataframe(quality, use_container_width=True)
    st.info("Regles appliquees : normalisation des libelles, conversion des dates, bornage des ratios, correction des valeurs negatives et consolidation des doublons.")

elif page == "RH Analytics":
    st.markdown("<div class='section-title'>RH Analytics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    top_risk = latest.sort_values("indicateur_risque_collectif", ascending=False).head(12)
    c1.plotly_chart(px.bar(top_risk, x="indicateur_risque_collectif", y="equipe", color="site", orientation="h", title="Equipes les plus instables"), use_container_width=True)
    c2.plotly_chart(px.scatter(latest, x="tension_recrutement", y="taux_attrition", size="effectif_reel", color="metier", hover_data=["site", "equipe"], title="Profils d'equipes distincts"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(analytics.head(15), x="site", y="ecart_effectif", color="metier", title="Ecarts effectif reel vs planifie"), use_container_width=True)
    c4.plotly_chart(px.bar(analytics.head(15), x="site", y="couverture_competences_critiques", color="metier", title="Couverture competences critiques"), use_container_width=True)
    st.dataframe(analytics, use_container_width=True)

else:
    st.markdown("<div class='section-title'>Forecast</div>", unsafe_allow_html=True)
    best = model_results.sort_values("MAE").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Modele retenu", best["modele"])
    c2.metric("MAE", f"{best['MAE']:.2f}")
    c3.metric("RMSE", f"{best['RMSE']:.2f}")
    st.plotly_chart(px.bar(model_results, x="modele", y=["MAE", "RMSE"], barmode="group", title="Comparaison baseline vs modeles"), use_container_width=True)
    fc = forecast.copy()
    fc["mois"] = pd.to_datetime(fc["mois"])
    agg_fc = fc.groupby("mois", as_index=False).agg(reel=("reel", "sum"), prediction_retenue=("prediction_retenue", "sum"))
    st.plotly_chart(px.line(agg_fc, x="mois", y=["reel", "prediction_retenue"], markers=True, title="Departs volontaires - reel vs predit"), use_container_width=True)
    st.dataframe(fc, use_container_width=True)
    st.warning("Limites : modele historique, variables agregees, biais possibles de saisie et de structure. Le modele predit des volumes collectifs, jamais des comportements individuels.")
    st.info("Facteurs explicatifs agreges utilises : effectifs, ecarts, tension recrutement, couverture skills, engagement, formation, absenteisme, site, metier et mois.")
'''
    (APP / "streamlit_app.py").write_text(app, encoding="utf-8")


def build_readme() -> None:
    readme = """# Projet AED RH

Projet complet d'analyse exploratoire RH, DATA GOLD, modelisation agregee des departs volontaires et dashboard Streamlit.

## Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Livrables

- `reports/rapport_final.md`
- `notebooks/AED_RH_notebook.ipynb`
- `data/processed/gold_data_rh.csv`
- `data/processed/data_dictionary.csv`
- `app/streamlit_app.py`

Toutes les analyses sont agregees au niveau equipe, site et metier. Aucun scoring individuel n'est produit.
"""
    (PROJECT / "README.md").write_text(readme, encoding="utf-8")
    (PROJECT / "requirements.txt").write_text("pandas\nnumpy\nscikit-learn\nmatplotlib\nplotly\nstreamlit\n", encoding="utf-8")


def main() -> None:
    mkdirs()
    copy_raw()
    gold, quality, _, _ = clean_and_build()
    gold.to_csv(PROCESSED / "gold_data_rh.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    dictionary = build_dictionary()
    model_results, forecast = build_model(gold)
    build_dashboards(gold, quality, forecast, model_results)
    build_report(gold, quality, model_results)
    build_notebook()
    build_app()
    build_readme()
    print(f"Projet genere: {PROJECT}")
    print(f"DATA GOLD: {len(gold)} lignes, {len(gold.columns)} colonnes")
    print(f"Dictionnaire: {len(dictionary)} variables")
    print(model_results)


if __name__ == "__main__":
    main()

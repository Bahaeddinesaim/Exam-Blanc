# Rapport final - AED RH

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

La DATA GOLD finale contient 7,694 lignes agregees. Le dernier mois disponible est 2025-12-01.

## Nettoyage et fiabilisation des donnees

Les traitements appliques sont : normalisation des sites et metiers, conversion des dates, suppression des dates ou equipes invalides, bornage des ratios entre 0 et 1, correction des effectifs negatifs a 0, plafonnement des departs volontaires a l'effectif reel, consolidation des doublons par agregation.

Le score qualite moyen estime des fichiers sources est de 50.0/100. Les anomalies detectees sont documentees dans `dashboard_data_quality.csv`.

## Construction de la DATA GOLD

La DATA GOLD combine les donnees workforce, skills, recruitment et reference. Les jointures sont faites sur les dimensions disponibles les plus pertinentes : mois/equipe pour les competences, mois/site/metier pour le recrutement, metier pour la criticite.

## Dictionnaire de donnees

Le dictionnaire `data_dictionary.csv` documente les definitions metier, types, unites, sources, regles qualite, owners et vigilances pour les variables principales.

## Analyse des dynamiques collectives RH

Sur le dernier mois, l'effectif reel est de 19,570 personnes pour 21,878 planifiees, soit un ecart de -2,308. Le taux d'attrition global pondere est de 27.44%. La tension recrutement moyenne est de 0.57.

## Modelisation predictive des departs volontaires agreges

Le dataset de modelisation utilise des variables collectives : effectifs, ecarts, tension recrutement, engagement, formation, absenteisme, couverture de competences, site et metier. Trois approches sont comparees : baseline moyenne, regression lineaire et Random Forest. Le meilleur modele selon la MAE est `Random Forest` avec une MAE de 12.804.

## Description du dashboard Streamlit

Le dashboard est organise en quatre parties : Executive, Data Quality, RH Analytics et Forecast. Il fournit des KPI cards, graphiques comparatifs, tableaux de priorisation collective, controle qualite et comparaison des modeles.

## Gouvernance, limites, biais et usage responsable

Les resultats servent au pilotage collectif. Ils ne doivent pas etre utilises pour prendre une decision automatisee sur une personne. Les biais possibles concernent la qualite des referentiels, les variations de saisie, les effets de structure par site/metier et le fait que la prediction porte sur des donnees historiques.

## Conclusion

Le projet fournit une chaine complete : ingestion, controle qualite, nettoyage, DATA GOLD, dictionnaire, modelisation et dashboard. Il repond au cadrage AED RH tout en respectant la contrainte essentielle : toutes les analyses restent agregees.

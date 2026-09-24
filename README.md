# Analyse KPI Jira 2025-2026

Script Python d'analyse de l'activité Jira hebdomadaire (tickets créés vs terminés) : calcul d'indicateurs dérivés (flux net, taux de complétion) et génération de graphiques et exports agrégés (par semaine, mois, trimestre) à partir d'un export CSV.

## Stack

- Python (pandas, numpy, matplotlib)

## Usage

```bash
python analyse_kpi_jira.py
```

Lit `jira_activity.csv` en entrée et produit des CSV et images (`kpi_jira_*`) en sortie.

# -*- coding: utf-8 -*-
"""
Analyse KPI - Activite Jira hebdomadaire (Issues created vs completed)
Source : jira_activity.csv
Auteur : thomas.delos36@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "jira_activity.csv"
OUT_PREFIX = "kpi_jira"

# --------------------------------------------------------------------------
# 1. Chargement & mise en forme
# --------------------------------------------------------------------------
raw = pd.read_csv(CSV)
raw.columns = ["semaine", "type", "valeur"]

# passage en format large : une ligne par semaine, colonnes created / completed
df = (raw.pivot_table(index="semaine", columns="type", values="valeur", aggfunc="sum")
         .rename(columns={"Issues created": "created", "Issues completed": "completed"})
         .sort_index()
         .reset_index())

# decomposition annee / numero de semaine + index temporel (lundi de la semaine ISO)
df["annee"] = df["semaine"].str.slice(0, 4).astype(int)
df["num_sem"] = df["semaine"].str.slice(6).astype(int)
df["date"] = pd.to_datetime(df["semaine"] + "-1", format="%G-W%V-%u")
df["mois"] = df["date"].dt.to_period("M").astype(str)
df["trimestre"] = df["date"].dt.to_period("Q").astype(str)

# --------------------------------------------------------------------------
# 2. Indicateurs derives
# --------------------------------------------------------------------------
df["net"] = df["created"] - df["completed"]          # flux net (+ = backlog qui gonfle)
df["ratio_completion"] = df["completed"] / df["created"]  # throughput ratio
df["backlog_cumule"] = df["net"].cumsum()            # evolution du backlog (base 0 au depart)
df["total_activite"] = df["created"] + df["completed"]

# moyennes mobiles 4 semaines
df["created_ma4"] = df["created"].rolling(4).mean()
df["completed_ma4"] = df["completed"].rolling(4).mean()
df["net_ma4"] = df["net"].rolling(4).mean()

nb_sem = len(df)
periode = f"{df['semaine'].iloc[0]} -> {df['semaine'].iloc[-1]}  ({nb_sem} semaines)"

def bloc(titre):
    print("\n" + "=" * 74)
    print(titre)
    print("=" * 74)

# --------------------------------------------------------------------------
# 3. KPI VOLUMES GLOBAUX
# --------------------------------------------------------------------------
bloc("1. VOLUMES GLOBAUX")
tot_created = df["created"].sum()
tot_completed = df["completed"].sum()
print(f"Periode analysee ............... {periode}")
print(f"Total issues creees ........... {tot_created:>8,}")
print(f"Total issues terminees ....... {tot_completed:>8,}")
print(f"Solde net (creees - terminees)  {tot_created - tot_completed:>+8,}")
print(f"Ratio completion global ...... {tot_completed / tot_created:>8.1%}")
print(f"Volume d'activite total ...... {tot_created + tot_completed:>8,}  (creees + terminees)")

# --------------------------------------------------------------------------
# 4. KPI DEBIT / VELOCITE HEBDO
# --------------------------------------------------------------------------
bloc("2. DEBIT HEBDOMADAIRE (velocite)")
def stats(s):
    return dict(moy=s.mean(), med=s.median(), ecart_type=s.std(),
               mini=s.min(), maxi=s.max(),
               cv=s.std() / s.mean(), p90=s.quantile(0.9), p10=s.quantile(0.1))

for nom, s in [("Creees   ", df["created"]), ("Terminees", df["completed"]), ("Flux net ", df["net"])]:
    st = stats(s)
    print(f"{nom} | moy {st['moy']:7.1f} | med {st['med']:7.1f} | "
          f"e-t {st['ecart_type']:6.1f} | min {st['mini']:5.0f} | max {st['maxi']:5.0f} | "
          f"CV {st['cv']:4.0%} | p10 {st['p10']:6.1f} | p90 {st['p90']:6.1f}")
print("\nCV = coefficient de variation (predictibilite : <25% stable, >50% tres irregulier)")

# --------------------------------------------------------------------------
# 5. KPI BACKLOG / FLUX NET
# --------------------------------------------------------------------------
bloc("3. BACKLOG & FLUX NET")
sem_backlog_up = (df["net"] > 0).sum()
sem_backlog_down = (df["net"] < 0).sum()
sem_equilibre = (df["net"] == 0).sum()
print(f"Semaines ou le backlog augmente (creees > terminees) . {sem_backlog_up:>3}  ({sem_backlog_up/nb_sem:.0%})")
print(f"Semaines ou le backlog diminue  (terminees > creees) . {sem_backlog_down:>3}  ({sem_backlog_down/nb_sem:.0%})")
print(f"Semaines a l'equilibre ............................... {sem_equilibre:>3}")
print(f"Croissance nette du backlog sur la periode ........... {df['backlog_cumule'].iloc[-1]:>+6,}")
print(f"Backlog cumule : min {df['backlog_cumule'].min():+.0f} (fin {df.loc[df['backlog_cumule'].idxmin(),'semaine']}) "
      f"| max {df['backlog_cumule'].max():+.0f} (fin {df.loc[df['backlog_cumule'].idxmax(),'semaine']})")
print(f"Derive backlog moyenne .............................. {df['net'].mean():+.1f} issues / semaine")

# Little's law : delai de traitement estime = WIP / debit
wip_moyen = df["backlog_cumule"].iloc[-1]
debit_moyen = df["completed"].mean()
if wip_moyen > 0:
    print(f"Delai d'ecoulement estime (Little, base backlog final / debit) ~ {wip_moyen/debit_moyen:.1f} semaines")

# --------------------------------------------------------------------------
# 6. KPI TENDANCE
# --------------------------------------------------------------------------
bloc("4. TENDANCE (regression lineaire sur le temps)")
x = np.arange(nb_sem)
for nom, s in [("Creees", df["created"]), ("Terminees", df["completed"]), ("Flux net", df["net"])]:
    pente, ordonnee = np.polyfit(x, s, 1)
    var_periode = pente * (nb_sem - 1)
    print(f"{nom:10} : pente {pente:+6.2f} issues/sem  -> evolution {var_periode:+.0f} sur la periode "
          f"({var_periode / s.iloc[0]:+.0%} vs 1re semaine)")

# comparaison 1er tiers vs dernier tiers
t = nb_sem // 3
for nom, s in [("Creees", df["created"]), ("Terminees", df["completed"])]:
    d = s.iloc[:t].mean()
    f = s.iloc[-t:].mean()
    print(f"{nom:10} : moy 1er tiers {d:6.1f}  ->  moy dernier tiers {f:6.1f}  ({(f-d)/d:+.0%})")

# --------------------------------------------------------------------------
# 7. KPI SAISONNALITE / AGREGATS
# --------------------------------------------------------------------------
bloc("5. AGREGATS MENSUELS")
mens = df.groupby("mois")[["created", "completed", "net"]].sum()
mens["ratio"] = mens["completed"] / mens["created"]
print(mens.to_string(float_format=lambda v: f"{v:,.2f}"))

bloc("6. AGREGATS TRIMESTRIELS")
trim = df.groupby("trimestre")[["created", "completed", "net"]].sum()
trim["ratio_completion"] = trim["completed"] / trim["created"]
print(trim.to_string(float_format=lambda v: f"{v:,.2f}"))

bloc("7. COMPARAISON ANNUELLE")
ann = df.groupby("annee")[["created", "completed", "net"]].agg(["sum", "mean", "count"])
print(ann.to_string(float_format=lambda v: f"{v:,.1f}"))

# --------------------------------------------------------------------------
# 8. KPI EXTREMES / ANOMALIES
# --------------------------------------------------------------------------
bloc("8. SEMAINES EXTREMES")
print("Top 3 creation :")
print(df.nlargest(3, "created")[["semaine", "created", "completed", "net"]].to_string(index=False))
print("\nTop 3 completion :")
print(df.nlargest(3, "completed")[["semaine", "created", "completed", "net"]].to_string(index=False))
print("\nCreux d'activite (total le plus faible) :")
print(df.nsmallest(3, "total_activite")[["semaine", "created", "completed", "total_activite"]].to_string(index=False))

bloc("9. DETECTION D'ANOMALIES (z-score > 2)")
for col in ["created", "completed"]:
    z = (df[col] - df[col].mean()) / df[col].std()
    out = df.loc[z.abs() > 2, ["semaine", col]].copy()
    out["z_score"] = z[z.abs() > 2].round(2).values
    if len(out):
        print(f"\n{col} :")
        print(out.to_string(index=False))
    else:
        print(f"\n{col} : aucune anomalie.")

# --------------------------------------------------------------------------
# 9. KPI QUALITE DU FLUX
# --------------------------------------------------------------------------
bloc("10. SANTE DU FLUX (synthese)")
ratio_moy = df["ratio_completion"].mean()
sante = "SOUS-CAPACITE (backlog gonfle)" if ratio_moy < 0.95 else \
        "EQUILIBRE" if ratio_moy <= 1.05 else "RESORPTION (backlog se vide)"
print(f"Ratio completion moyen hebdo ......... {ratio_moy:.1%}  -> {sante}")
print(f"Semaines avec ratio >= 100% .......... {(df['ratio_completion'] >= 1).sum()} / {nb_sem}")
print(f"Predictibilite creation (CV) ......... {df['created'].std()/df['created'].mean():.0%}")
print(f"Predictibilite completion (CV) ...... {df['completed'].std()/df['completed'].mean():.0%}")
corr = df["created"].corr(df["completed"])
print(f"Correlation creation <-> completion .. {corr:+.2f}")
print(f"Debit stabilise (moy mobile 4 sem, derniere valeur) : "
      f"created {df['created_ma4'].iloc[-1]:.0f} | completed {df['completed_ma4'].iloc[-1]:.0f}")

# --------------------------------------------------------------------------
# 10. Export
# --------------------------------------------------------------------------
cols_export = ["semaine", "date", "annee", "num_sem", "created", "completed", "net",
               "ratio_completion", "backlog_cumule", "total_activite",
               "created_ma4", "completed_ma4", "net_ma4"]
df[cols_export].to_csv(f"{OUT_PREFIX}_detail.csv", index=False)
mens.to_csv(f"{OUT_PREFIX}_mensuel.csv")
trim.to_csv(f"{OUT_PREFIX}_trimestriel.csv")
print(f"\n[export] {OUT_PREFIX}_detail.csv / _mensuel.csv / _trimestriel.csv")

# --------------------------------------------------------------------------
# 11. Graphiques (dashboard commente)
# --------------------------------------------------------------------------
# Quelques chiffres cles injectes directement dans les explications du dashboard
kpi_ratio_global = tot_completed / tot_created
kpi_backlog_final = df["backlog_cumule"].iloc[-1]
kpi_sem_up = int((df["net"] > 0).sum())
kpi_pente_completed = np.polyfit(x, df["completed"], 1)[0]

# Style commun pour les encadres d'explication places sous chaque graphique
BOX = dict(boxstyle="round,pad=0.6", facecolor="#f4f4f4", edgecolor="#cccccc")

def explication(ax, texte):
    """Ajoute un encadre de texte explicatif juste sous le graphique 'ax'."""
    ax.text(0.0, -0.32, texte, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace", bbox=BOX)

fig, axes = plt.subplots(3, 1, figsize=(14, 17))
fig.suptitle("Dashboard KPI - Activite Jira hebdomadaire  (" + periode + ")",
             fontsize=14, fontweight="bold", y=0.995)

# --- Graphique 1 : les deux flux -----------------------------------------
axes[0].plot(df["date"], df["created"], marker="o", ms=3, label="Creees", color="#d1495b")
axes[0].plot(df["date"], df["completed"], marker="o", ms=3, label="Terminees", color="#2e86ab")
axes[0].plot(df["date"], df["created_ma4"], "--", color="#d1495b", alpha=.6, label="Creees - moy. mobile 4 sem.")
axes[0].plot(df["date"], df["completed_ma4"], "--", color="#2e86ab", alpha=.6, label="Terminees - moy. mobile 4 sem.")
axes[0].set_title("1) FLUX HEBDOMADAIRE : tickets ouverts vs tickets fermes", fontweight="bold", loc="left")
axes[0].set_ylabel("nb de tickets / semaine")
axes[0].legend(ncol=2, fontsize=8); axes[0].grid(alpha=.3)
explication(axes[0],
    "CE QUE MONTRE LE GRAPHIQUE\n"
    " - Courbe rouge  : tickets OUVERTS chaque semaine (la demande qui arrive)\n"
    " - Courbe bleue  : tickets FERMES chaque semaine (le travail realise = la capacite)\n"
    " - Pointilles    : version lissee sur 4 semaines, pour voir la tendance sans les a-coups\n"
    "COMMENT LIRE\n"
    " - Rouge AU-DESSUS du bleu = on ouvre plus qu'on ne ferme -> on prend du retard cette semaine-la\n"
    " - Le pic bleu de W42-2025 (2523) est un nettoyage massif de vieux tickets, pas un rythme normal\n"
    f"CONSTAT : sur l'annee, ratio ferme/ouvert = {kpi_ratio_global:.0%}  (on ferme moins vite qu'on ouvre)")

# --- Graphique 2 : le flux net -----------------------------------------
colors = ["#2e86ab" if v <= 0 else "#d1495b" for v in df["net"]]
axes[1].bar(df["date"], df["net"], width=5, color=colors)
axes[1].axhline(0, color="k", lw=.8)
axes[1].set_title("2) FLUX NET PAR SEMAINE = ouverts - fermes", fontweight="bold", loc="left")
axes[1].set_ylabel("solde de la semaine")
axes[1].grid(alpha=.3)
explication(axes[1],
    "CE QUE MONTRE LE GRAPHIQUE\n"
    " - Chaque barre = (tickets ouverts) - (tickets fermes) pour la semaine\n"
    " - Barre ROUGE (au-dessus de 0) : le retard AUGMENTE cette semaine\n"
    " - Barre BLEUE (sous 0)         : le retard DIMINUE cette semaine\n"
    "COMMENT LIRE\n"
    " - Beaucoup de rouge = probleme de capacite recurrent\n"
    " - La grande barre bleue de W42-2025 correspond au nettoyage massif vu au graphique 1\n"
    f"CONSTAT : {kpi_sem_up} semaines sur {nb_sem} font grossir le retard ({kpi_sem_up/nb_sem:.0%} du temps)")

# --- Graphique 3 : le backlog cumule -----------------------------------------
axes[2].fill_between(df["date"], df["backlog_cumule"], 0,
                     where=df["backlog_cumule"] >= 0, color="#d1495b", alpha=.4)
axes[2].fill_between(df["date"], df["backlog_cumule"], 0,
                     where=df["backlog_cumule"] < 0, color="#2e86ab", alpha=.4)
axes[2].plot(df["date"], df["backlog_cumule"], color="k", lw=1)
axes[2].axhline(0, color="k", lw=.8)
axes[2].set_title("3) RETARD CUMULE (backlog) - somme de tous les flux nets depuis le debut",
                  fontweight="bold", loc="left")
axes[2].set_ylabel("tickets en attente (cumul)")
explication(axes[2],
    "CE QUE MONTRE LE GRAPHIQUE\n"
    " - On additionne semaine apres semaine les soldes du graphique 2 : c'est la pile de retard\n"
    " - Depart a 0 : le graphique montre l'EVOLUTION du retard, pas sa valeur absolue\n"
    " - Zone rouge = retard positif (on a accumule) ; zone bleue = on est repasse sous le niveau de depart\n"
    "COMMENT LIRE\n"
    " - Courbe qui monte = la situation se degrade ; qui descend = on rattrape\n"
    " - Le creux de fin 2025 vient du nettoyage W42 ; ensuite la courbe remonte regulierement\n"
    f"CONSTAT : +{kpi_backlog_final:,.0f} tickets accumules sur l'annee, "
    f"et la capacite de traitement baisse ({kpi_pente_completed:+.1f} ticket/sem de tendance)")
axes[2].grid(alpha=.3)

plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=6.0)
plt.savefig(f"{OUT_PREFIX}_dashboard.png", dpi=120, bbox_inches="tight")
print(f"[export] {OUT_PREFIX}_dashboard.png")

# --------------------------------------------------------------------------
# 12. KPI "TEMPS DE SAISIE" (hypothese : 8 a 12 min pour creer un ticket)
# --------------------------------------------------------------------------
MIN_LOW, MIN_HIGH = 8, 12
MIN_MID = (MIN_LOW + MIN_HIGH) / 2          # 10 min
H_JOUR = 7                                  # heures travaillees par jour
J_AN = 220                                  # jours travailles par an et par personne

def h(minutes):   # minutes -> heures
    return minutes / 60

tot_min_low  = tot_created * MIN_LOW
tot_min_mid  = tot_created * MIN_MID
tot_min_high = tot_created * MIN_HIGH
sem_h_mid    = df["created"].mean() * MIN_MID / 60
etp = h(tot_min_mid) / (H_JOUR * J_AN)     # equivalents temps plein

bloc("11. TEMPS PASSE A CREER LES TICKETS (hypothese 8-12 min / ticket)")
print(f"Hypothese ......................... {MIN_LOW}-{MIN_HIGH} min par ticket (moyenne retenue : {MIN_MID:.0f} min)")
print(f"Total tickets crees .............. {tot_created:,}")
print(f"Temps total de saisie ........... {h(tot_min_low):,.0f} h  ->  {h(tot_min_high):,.0f} h "
      f"(central : {h(tot_min_mid):,.0f} h  ~  {h(tot_min_mid)/H_JOUR:,.0f} jours-homme)")
print(f"Charge moyenne par semaine ...... {df['created'].mean()*MIN_LOW/60:,.0f} h  ->  "
      f"{df['created'].mean()*MIN_HIGH/60:,.0f} h  (central : {sem_h_mid:,.0f} h/sem)")
print(f"Semaine la plus lourde (W{df.loc[df['created'].idxmax(),'semaine']}) ... "
      f"{df['created'].max()*MIN_MID/60:,.0f} h de saisie pour {df['created'].max():,} tickets")
print(f"Equivalent temps plein (base {H_JOUR}h x {J_AN}j) ... {etp:.1f} ETP dedies uniquement a la creation")
print(f"Cout mensuel moyen ............... {h(mens['created'].mean()*MIN_MID):,.0f} h/mois")

# --------------------------------------------------------------------------
# 13. SLIDE : tickets crees par semaine (peu de texte, lisible par tous)
# --------------------------------------------------------------------------
moy_created = df["created"].mean()
i_max = df["created"].idxmax()

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.06, right=0.97)

bars = ax.bar(df["date"], df["created"], width=5.5, color="#2e86ab")
bars[i_max].set_color("#d1495b")   # on met en avant la semaine record
ax.set_ylim(0, df["created"].max() * 1.15)

# ligne de moyenne
ax.axhline(moy_created, color="#555555", ls="--", lw=1.5)
ax.text(df["date"].iloc[0], moy_created + df["created"].max()*0.02,
        f"moyenne {moy_created:,.0f} tickets/semaine",
        va="bottom", ha="left", fontsize=15, color="#555555")

# etiquette sur la semaine record
ax.annotate(f"record : {df['created'].max():,}  ({df.loc[i_max,'semaine']})",
            xy=(df["date"].iloc[i_max], df["created"].max()),
            xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=15, fontweight="bold", color="#d1495b")

# titre + sous-titre (au niveau figure, pour ne pas chevaucher les barres)
fig.text(0.06, 0.93, "Tickets Jira créés par semaine", fontsize=30, fontweight="bold")
fig.text(0.06, 0.87, f"{df['semaine'].iloc[0]} → {df['semaine'].iloc[-1]}   ·   "
                     f"{tot_created:,} tickets au total   ·   {nb_sem} semaines",
         fontsize=17, color="#666666")

# note "coût temps" en bas, une seule ligne simple
fig.text(0.06, 0.05,
         f"Créer un ticket prend {MIN_LOW}–{MIN_HIGH} min  →  environ "
         f"{df['created'].mean()*MIN_LOW/60:,.0f}–{df['created'].mean()*MIN_HIGH/60:,.0f} h de saisie par semaine   "
         f"(≈ {etp:.1f} personnes à plein temps sur l'année)",
         fontsize=16, color="#333333")

# habillage minimal
ax.set_ylabel("")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", alpha=.25)
ax.margins(x=0.01)
ax.tick_params(axis="x", labelsize=13)

plt.savefig(f"{OUT_PREFIX}_slide_tickets_crees.png", dpi=120)
print(f"[export] {OUT_PREFIX}_slide_tickets_crees.png")

# ==========================================================================
# 14. KPI COMPLEMENTAIRES
# ==========================================================================

# --- 14.a Cadence / takt time -----------------------------------------------
bloc("12. CADENCE (takt time)")
tpw = df["created"].mean()
print(f"Rythme de creation ............... 1 ticket toutes les "
      f"{7*24*60/tpw:,.0f} min en temps calendaire (24/7)")
print(f"                                   1 ticket toutes les "
      f"{5*7*60/tpw:,.1f} min en temps ouvre (5j x 7h)")
print(f"Rythme de completion ............ 1 ticket toutes les "
      f"{5*7*60/df['completed'].mean():,.1f} min en temps ouvre")
print(f"Debit / jour ouvre .............. {tpw/5:,.0f} crees  |  {df['completed'].mean()/5:,.0f} termines")

# --- 14.b KPI "nettoyes" (hors semaines atypiques) -------------------------
EXCLURE = {"2025-W42",             # nettoyage massif (2523 termines)
           "2025-W52", "2026-W01", # fetes de fin d'annee
           "2026-W23",             # semaine quasi vide (ferie / partielle)
           "2026-W35"}             # derniere semaine, donnees incompletes
dfn = df[~df["semaine"].isin(EXCLURE)]
bloc(f"13. KPI NORMALISES (hors {len(EXCLURE)} semaines atypiques : {', '.join(sorted(EXCLURE))})")
print(f"Semaines retenues ............... {len(dfn)} / {nb_sem}")
print(f"Crees   : moy {dfn['created'].mean():6.1f}  (vs {df['created'].mean():.1f} brut)  | CV {dfn['created'].std()/dfn['created'].mean():.0%}")
print(f"Termines: moy {dfn['completed'].mean():6.1f}  (vs {df['completed'].mean():.1f} brut)  | CV {dfn['completed'].std()/dfn['completed'].mean():.0%}")
print(f"Flux net normalise ............. {dfn['net'].mean():+.1f} / sem   (ratio completion {dfn['completed'].sum()/dfn['created'].sum():.1%})")

# --- 14.c Predictibilite : bande autour de la mediane ----------------------
bloc("14. PREDICTIBILITE (part des semaines proches de la normale)")
for nom, s in [("Crees", df["created"]), ("Termines", df["completed"])]:
    med = s.median()
    for p in (0.10, 0.20, 0.30):
        share = s.between(med*(1-p), med*(1+p)).mean()
        print(f"{nom:9} : {share:4.0%} des semaines a +/- {p:.0%} de la mediane ({med:.0f})")
    print()

# --- 14.d Enveloppe de capacite (moyennes mobiles) -----------------------
bloc("15. ENVELOPPE DE CAPACITE (debit lisse)")
for w in (4, 12):
    r = df["completed"].rolling(w).mean().dropna()
    rc = df["created"].rolling(w).mean().dropna()
    print(f"Fenetre {w:2} sem | termines : mini {r.min():6.0f}  maxi {r.max():6.0f}  "
          f"(actuel {r.iloc[-1]:.0f})   || crees : mini {rc.min():6.0f}  maxi {rc.max():6.0f}  (actuel {rc.iloc[-1]:.0f})")

# --- 14.e Indice de saisonnalite mensuel ----------------------------------
# base 100 = moyenne hebdo toutes semaines ; calcule sur la moyenne PAR SEMAINE
# de chaque mois (sinon les mois a 5 semaines paraissent artificiellement hauts)
bloc("16. INDICE DE SAISONNALITE (base 100 = semaine moyenne)")
g = df.groupby("mois")["created"]
mm_mean, mm_cnt = g.mean(), g.count()
mm_idx = (mm_mean / df["created"].mean() * 100).round(0)
for k in mm_idx.index:
    v, n = mm_idx[k], mm_cnt[k]
    if n < 3:
        print(f"  {k} : {v:5.0f}   (mois partiel : {n} sem., non significatif)")
        continue
    marque = "  <== creux" if v < 80 else ("  <== pic" if v > 120 else "")
    print(f"  {k} : {v:5.0f}{marque}")

# --- 14.f Semaines "speciales" chiffrees ----------------------------------
bloc("17. IMPACT DES SEMAINES SPECIALES vs mediane normale")
med_c, med_k = dfn["completed"].median(), dfn["created"].median()
for sem in ["2025-W42", "2025-W52", "2026-W01", "2026-W23"]:
    row = df[df["semaine"] == sem].iloc[0]
    print(f"  {sem} : crees {row['created']:4.0f} ({row['created']/med_k-1:+.0%})  |  "
          f"termines {row['completed']:4.0f} ({row['completed']/med_c-1:+.0%})")

# --- 14.g Stock & projection backlog -------------------------------------
bloc("18. STOCK DE RETARD & PROJECTION")
derive = df["net"].mean()
inv_sem = kpi_backlog_final / df["completed"].mean()
inv_jour = kpi_backlog_final / (df["completed"].mean()/5)
print(f"Retard accumule sur la periode .. {kpi_backlog_final:+,.0f} tickets")
print(f"Semaines d'inventaire ........... {inv_sem:,.1f} sem  ({inv_jour:,.0f} jours ouvres de travail au debit actuel)")
print(f"Derive moyenne ................. {derive:+.1f} tickets / semaine")
if derive > 0:
    print(f"Au rythme actuel, +1000 tickets de retard supplementaires en ~{1000/derive:,.0f} semaines")
    print(f"Pour resorber le retard en 13 semaines : +{kpi_backlog_final/13:,.0f} completions / sem "
          f"(soit {(df['completed'].mean()+kpi_backlog_final/13)/df['completed'].mean()-1:+.0%} de capacite)")

# --- 14.h Delai de traitement estime (diagramme de flux cumule) -----------
bloc("19. DELAI DE TRAITEMENT ESTIME (Little's Law sur flux cumules)")
cc = df["created"].cumsum().values
kk = df["completed"].cumsum().values
delais = []
for t in range(len(df)):
    j = np.searchsorted(kk, cc[t])          # quand le cumul termine rattrape ce cumul cree
    if j < len(df):
        delais.append(j - t)
delais = np.array(delais)
print(f"Delai moyen estime ............. {delais.mean():.1f} semaines  (median {np.median(delais):.0f}, max {delais.max():.0f})")
print("Interpretation : un ticket cree aujourd'hui est ferme ~{:.0f} semaines plus tard en moyenne.".format(delais.mean()))

# --- 14.i Volatilite & persistance --------------------------------------
bloc("20. VOLATILITE & MOMENTUM")
for nom, s in [("Crees", df["created"]), ("Termines", df["completed"])]:
    wow = s.pct_change().abs().mean()
    ac1 = s.autocorr(1)
    print(f"{nom:9} : variation moyenne d'une semaine a l'autre {wow:.0%}  |  autocorrelation (lag 1) {ac1:+.2f}")
print("autocorrelation > 0.3 : les semaines chargees ont tendance a s'enchainer.")

# --- 14.j Concentration (Pareto) --------------------------------------
bloc("21. CONCENTRATION DE LA CHARGE (Pareto)")
top5 = df.nlargest(5, "created")["created"].sum()
top10 = df.nlargest(10, "created")["created"].sum()
print(f"Top 5  semaines = {top5/tot_created:.0%} des tickets crees ({top5:,})")
print(f"Top 10 semaines = {top10/tot_created:.0%} des tickets crees ({top10:,})")
print(f"10 semaines les plus calmes = {df.nsmallest(10,'created')['created'].sum()/tot_created:.0%} du volume")

# --- 14.k Correlation decalee crees -> termines -----------------------
bloc("22. LES PICS DE CREATION SE PROPAGENT-ILS A LA COMPLETION ?")
for lag in range(0, 5):
    c = df["created"].shift(lag).corr(df["completed"])
    print(f"  decalage {lag} sem : correlation crees(t-{lag}) <-> termines(t) = {c:+.2f}")

# --- 14.l Charge RH par mois (creation) -----------------------------
bloc("23. BESOIN RH POUR LA SAISIE, PAR MOIS (10 min/ticket, 152 h/mois/pers)")
h_mois = mens["created"] * MIN_MID / 60
fte_mois = h_mois / 152
print(f"Charge mensuelle : {h_mois.min():.0f} h -> {h_mois.max():.0f} h   |   "
      f"ETP : {fte_mois.min():.1f} -> {fte_mois.max():.1f}  (moyenne {fte_mois.mean():.1f})")
print(f"Mois le plus lourd : {h_mois.idxmax()} ({h_mois.max():.0f} h ~ {fte_mois.max():.1f} ETP)")

# --- 14.m Valorisation financiere (hypothese de taux) -----------------
RATE_EUR = 45   # cout horaire charge suppose - a ajuster
bloc(f"24. VALORISATION DU TEMPS DE SAISIE (hypothese {RATE_EUR} EUR/h charge)")
print(f"Cout annuel de la creation de tickets : "
      f"{h(tot_min_low)*RATE_EUR:,.0f} EUR -> {h(tot_min_high)*RATE_EUR:,.0f} EUR "
      f"(central {h(tot_min_mid)*RATE_EUR:,.0f} EUR)")
print(f"Economie si -1 min/ticket (outillage) : ~{tot_created*1/60*RATE_EUR:,.0f} EUR / an")

# --------------------------------------------------------------------------
# 15. Graphique : diagramme de flux cumule (CFD)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df["date"], df["created"].cumsum(), color="#d1495b", lw=2, label="Cumul tickets crees")
ax.plot(df["date"], df["completed"].cumsum(), color="#2e86ab", lw=2, label="Cumul tickets termines")
ax.fill_between(df["date"], df["completed"].cumsum(), df["created"].cumsum(),
                color="#d1495b", alpha=.15)
ax.set_title("Diagramme de flux cumule (CFD)\n"
             "ecart vertical = retard en cours  |  ecart horizontal = delai de traitement",
             fontweight="bold", loc="left")
ax.set_ylabel("nombre de tickets (cumule)")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_flux_cumule.png", dpi=120)
print(f"[export] {OUT_PREFIX}_flux_cumule.png")

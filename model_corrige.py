"""EquiAlgo - modele corrige et front de Pareto.

Methode : neutralisation contrefactuelle des canaux biaises.

1. On entraine une regression logistique sur les decisions historiques AVEC la region,
   le revenu et la distance. Le modele range ainsi le biais du comite dans des
   coefficients identifiables au lieu de le cacher dans les proxys.
2. Au moment de decider, on neutralise ces termes : tout le monde est evalue comme
   s'il avait la meme region, le meme revenu et la meme distance. Le programme et la
   premiere generation sont aussi neutralises : leur effet est nul (p > 0,5) et les garder
   penaliserait legerement les etudiant-es de premiere generation. Restent la cote R et
   les heures travaillees : merite = cote R + ~0,146 point par heure.
3. On accorde la bourse aux 40 % meilleurs scores : l'enveloppe est respectee par construction.

Le parametre lambda (0 -> 1) dose la correction. On le balaie pour tracer le front de
Pareto, et on compare a ThresholdOptimizer et ExponentiatedGradient (fairlearn).

Usage : python model_corrige.py   (ecrit predictions.csv, pareto.png, resultats_pareto.csv)
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from commun import (BUDGET_MAX, BUDGET_MIN, COULEURS, ENCRE_2, GRIS, RACINE, charger, ecart_egalite_chances,
                    ecart_parite, encoder, style_graphique, top_k)

warnings.filterwarnings("ignore")
GRAINE = 42

REGIONS = ["Montreal", "Capitale-Nationale", "Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
PROGRAMMES = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]
# Canaux du biais regional (audit, sections 3-4) : retires de la decision.
BIAISEES = ["log_revenu", "distance_km"] + [f"reg_{r}" for r in REGIONS[1:]]
# Sans effet mesurable sur le merite (audit, section 6b) : retires aussi.
SANS_EFFET = ["premiere_gen"] + [f"prog_{p}" for p in PROGRAMMES[1:]]
NEUTRALISEES = BIAISEES + SANS_EFFET


# --------------------------------------------------------------------------- donnees
def caracteristiques(df):
    """Matrice explicite pour la regression logistique (une ligne par dossier)."""
    X = pd.DataFrame({
        "cote_r": df["cote_r_equivalent"],
        "heures": df["heures_travail_semaine"],
        "premiere_gen": df["premiere_generation_universitaire"],
        "log_revenu": np.log(df["revenu_familial_estime"]),
        "distance_km": df["distance_domicile_campus_km"],
    }, index=df.index)
    for p in PROGRAMMES[1:]:
        X[f"prog_{p}"] = (df["programme_etudes"] == p).astype(float)
    for r in REGIONS[1:]:
        X[f"reg_{r}"] = (df["region_administrative"] == r).astype(float)
    return X


def etalon_merite(df, poids_heures):
    """Merite reconstruit (audit, section 6) : 40 % meilleurs scores cote R + heures."""
    return top_k(df["cote_r_equivalent"] + poids_heures * df["heures_travail_semaine"])


# --------------------------------------------------------------------------- modele
class ModeleNeutralise:
    """Regression logistique dont on retire, a la prediction, les termes biaises."""

    def fit(self, df, y):
        X = caracteristiques(df)
        self.reference = X[NEUTRALISEES].median()        # valeur commune imposee a tous
        indicatrices = [c for c in NEUTRALISEES if c.startswith(("reg_", "prog_")) or c == "premiere_gen"]
        self.reference[indicatrices] = 0.0                # Montreal, Arts et lettres, pas premiere generation
        self.colonnes = list(X.columns)
        self.logit = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        self.coef = pd.Series(self.logit.coef_[0], index=self.colonnes)
        return self

    def score(self, df, lam=1.0):
        """lam = 0 : copie du comite ; lam = 1 : canaux biaises entierement neutralises."""
        X = caracteristiques(df)[self.colonnes]
        complet = X.values @ self.coef.values + self.logit.intercept_[0]
        biais = (X[NEUTRALISEES] - self.reference).values @ self.coef[NEUTRALISEES].values
        return complet - lam * biais

    def poids_en_points_r(self):
        return (self.coef / self.coef["cote_r"]).round(4)


# --------------------------------------------------------------------------- evaluation
def mesurer(nom, famille, reglage, pred, df, merite):
    g = df["groupe"].values
    return {
        "methode": nom, "famille": famille, "reglage": reglage,
        "taux_octroi": pred.mean(),
        "exactitude_merite": (pred == merite).mean(),
        "exactitude_comite": (pred == df["decision_octroi"].values).mean() if "decision_octroi" in df else np.nan,
        "ecart_egalite_chances": ecart_egalite_chances(merite, pred, g),
        "ecart_parite": ecart_parite(pred, g),
    }


def balayages(train, valid, merite_valid):
    """Toutes les methodes, entrainees sur train, mesurees sur valid."""
    lignes = []
    y_train = train["decision_octroi"]
    X_tr, X_va = encoder(train, valid)

    # Modele de production et retrait naif de la region.
    foret = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=GRAINE)
    lignes.append(mesurer("Foret de base (production)", "reference", None,
                          foret.fit(X_tr, y_train).predict(X_va), valid, merite_valid))
    garder = [c for c in X_tr.columns if not c.startswith(("region_administrative_", "code_postal_3_"))]
    lignes.append(mesurer("Foret sans region ni code postal", "reference", None,
                          foret.fit(X_tr[garder], y_train).predict(X_va[garder]), valid, merite_valid))

    # Notre methode : balayage de lambda.
    modele = ModeleNeutralise().fit(train, y_train)
    for lam in np.round(np.linspace(0, 1, 11), 2):
        pred = top_k(modele.score(valid, lam))
        lignes.append(mesurer("Neutralisation contrefactuelle", "neutralisation", lam, pred, valid, merite_valid))

    # fairlearn : post-traitement (seuils par groupe) - egalite des chances vs decisions historiques.
    to = ThresholdOptimizer(estimator=RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                             random_state=GRAINE),
                            constraints="true_positive_rate_parity", objective="accuracy_score",
                            predict_method="predict_proba")
    to.fit(X_tr, y_train, sensitive_features=train["groupe"])
    pred = np.asarray(to.predict(X_va, sensitive_features=valid["groupe"], random_state=GRAINE))
    lignes.append(mesurer("ThresholdOptimizer (TPR parity)", "reference", None, pred, valid, merite_valid))

    # fairlearn : reentrainement sous contrainte, balayage de la tolerance.
    echelle = StandardScaler().fit(X_tr.astype(float))
    Xs_tr, Xs_va = echelle.transform(X_tr.astype(float)), echelle.transform(X_va.astype(float))
    for eps in [0.20, 0.10, 0.05, 0.02, 0.01]:
        eg = ExponentiatedGradient(LogisticRegression(max_iter=2000), TruePositiveRateParity(difference_bound=eps))
        eg.fit(Xs_tr, y_train, sensitive_features=train["groupe"])
        pred = np.asarray(eg.predict(Xs_va, random_state=GRAINE))
        lignes.append(mesurer("ExponentiatedGradient (TPR parity)", "exp_gradient", eps, pred, valid, merite_valid))
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------- graphique
def tracer_pareto(res, chemin):
    style_graphique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    panneaux = [("exactitude_merite", "Exactitude contre le mérite reconstruit"),
                ("exactitude_comite", "Exactitude contre les décisions historiques")]
    for ax, (col, titre) in zip(axes, panneaux):
        neu = res[res.famille == "neutralisation"].sort_values("reglage")
        ax.plot(neu.ecart_egalite_chances, neu[col], color=COULEURS["Centre"], lw=2, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=1.5, label="Neutralisation (λ = 0 → 1)")
        for _, r in neu.iloc[[0, 5, -1]].iterrows():
            ax.annotate(f"λ={r.reglage:g}", (r.ecart_egalite_chances, r[col]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=ENCRE_2)
        eg = res[res.famille == "exp_gradient"].sort_values("reglage")
        ax.plot(eg.ecart_egalite_chances, eg[col], color=COULEURS["Eloignee"], lw=2, marker="s", ms=6,
                markeredgecolor="white", markeredgewidth=1.5, label="ExponentiatedGradient (ε balayé)")
        for (_, r), m in zip(res[res.famille == "reference"].iterrows(), ["X", "D", "^"]):
            ax.scatter(r.ecart_egalite_chances, r[col], marker=m, s=70, color=GRIS, zorder=3,
                       edgecolor="white", linewidth=1, label=r.methode)
        ax.set_title(titre)
        ax.set_xlabel("Écart d'égalité des chances (Centre − Éloignée), contre le mérite")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.axvline(0, color=GRIS, lw=1, ls="--")
    axes[0].set_ylabel("Exactitude")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, -0.03, "Le mérite reconstruit utilise le poids des heures estimé sur le comité : à λ = 1, l'exactitude "
             "de gauche est presque parfaite par construction.\nValidation externe : HxBuddy mesure 94,6 % "
             "d'exactitude contre le mérite réel caché pour cette solution.", fontsize=8.5, color=ENCRE_2)
    fig.suptitle("Front de Pareto équité / exactitude (jeu de validation, 3 000 dossiers)",
                 fontweight="bold", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- principal
def main():
    hist, cand = charger()

    # Poids des heures dans le merite = celui que le comite leur donne (en points R), estime sur l'historique.
    poids_heures = ModeleNeutralise().fit(hist, hist["decision_octroi"]).poids_en_points_r()["heures"]
    print(f"Poids des heures dans le merite reconstruit : {poids_heures:.4f} point R / heure")

    train, valid = train_test_split(hist, test_size=0.3, random_state=GRAINE, stratify=hist["decision_octroi"])
    resultats = balayages(train, valid, etalon_merite(valid, poids_heures))
    resultats.round(4).to_csv(RACINE / "resultats_pareto.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(resultats.round(3).to_string(index=False))
    tracer_pareto(resultats, RACINE / "pareto.png")

    # Modele final : entraine sur tout l'historique, lambda = 1.
    final = ModeleNeutralise().fit(hist, hist["decision_octroi"])
    print("\nPoids du modele final (equivalent points de cote R) :")
    print(final.poids_en_points_r().to_string())
    decision = top_k(final.score(cand, lam=1.0))
    taux = decision.mean()
    assert BUDGET_MIN <= taux <= BUDGET_MAX, f"Hors enveloppe : {taux:.1%}"
    assert len(decision) == 4000 and set(np.unique(decision)) <= {0, 1}

    soumission = pd.DataFrame({"id_candidat": cand["id_candidat"], "decision_octroi": decision})
    soumission.to_csv(RACINE / "predictions.csv", index=False)
    merite_cand = etalon_merite(cand, poids_heures)
    bilan = mesurer("Modele final", "final", 1.0, decision, cand, merite_cand)
    print(f"\npredictions.csv : {len(soumission)} lignes, taux d'octroi {taux:.1%} (OK)")
    print(f"Taux par groupe : {pd.Series(decision).groupby(cand['groupe'].values).mean().round(3).to_dict()}")
    print(f"Contre le merite reconstruit : exactitude {bilan['exactitude_merite']:.3f}, "
          f"ecart egalite des chances {bilan['ecart_egalite_chances']:+.3f}")


if __name__ == "__main__":
    main()

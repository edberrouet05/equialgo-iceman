"""Fonctions partagees par audit_rapport.ipynb et model_corrige.py (defi EquiAlgo)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
DONNEES = RACINE / "data"

ELOIGNEES = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
CATEGORIELLES = ["programme_etudes", "region_administrative", "code_postal_3"]
BUDGET_MIN, BUDGET_MAX, BUDGET_CIBLE = 0.36, 0.44, 0.40

# Palette validee (skill dataviz, slots 1 et 2) : identite des deux groupes.
COULEURS = {"Centre": "#2a78d6", "Eloignee": "#eb6834"}
GRIS = "#8a8984"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"


def charger():
    """Retourne (historique, candidats) avec une colonne 'groupe' ajoutee."""
    hist = pd.read_csv(DONNEES / "donnees_demandes.csv")
    cand = pd.read_csv(DONNEES / "candidats_evaluation.csv")
    for df in (hist, cand):
        df["groupe"] = groupe_region(df)
    return hist, cand


def groupe_region(df):
    """Deux blocs : 'Centre' (Montreal, Capitale-Nationale) ou 'Eloignee'."""
    return np.where(df["region_administrative"].isin(ELOIGNEES), "Eloignee", "Centre")


def encoder(df_entrainement, df_cible):
    """Encodage one-hot identique a celui du carnet de depart (colonnes alignees)."""
    retirer = ["id_candidat", "decision_octroi", "groupe"]
    X = pd.get_dummies(df_entrainement.drop(columns=retirer, errors="ignore"), columns=CATEGORIELLES)
    Xc = pd.get_dummies(df_cible.drop(columns=retirer, errors="ignore"), columns=CATEGORIELLES)
    return X, Xc.reindex(columns=X.columns, fill_value=0)


def top_k(score, taux=BUDGET_CIBLE):
    """Accorde la bourse aux `taux` x 100 % meilleurs scores (respecte l'enveloppe par construction)."""
    score = np.asarray(score, dtype=float)
    k = int(round(taux * len(score)))
    ordre = np.argsort(-score, kind="stable")
    decision = np.zeros(len(score), dtype=int)
    decision[ordre[:k]] = 1
    return decision


def ecart_egalite_chances(y_ref, y_pred, groupe):
    """TPR(Centre) - TPR(Eloignee), mesure contre l'etiquette de reference fournie."""
    y_ref, y_pred, groupe = map(np.asarray, (y_ref, y_pred, groupe))
    tpr = {g: y_pred[(groupe == g) & (y_ref == 1)].mean() for g in ("Centre", "Eloignee")}
    return tpr["Centre"] - tpr["Eloignee"]


def ecart_parite(y_pred, groupe):
    """Taux d'octroi(Centre) - taux d'octroi(Eloignee)."""
    y_pred, groupe = np.asarray(y_pred), np.asarray(groupe)
    return y_pred[groupe == "Centre"].mean() - y_pred[groupe == "Eloignee"].mean()


def style_graphique():
    """Style sobre : grille discrete, axes recessifs, texte en encre neutre."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": ENCRE_2,
        "axes.titlecolor": ENCRE,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.8,
        "xtick.color": ENCRE_2,
        "ytick.color": ENCRE_2,
        "legend.frameon": False,
        "font.size": 10,
    })

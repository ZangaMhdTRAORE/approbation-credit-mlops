import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "decision"
POSITIVE_LABEL = "approuvé"

FEATURES = [
    "age","sexe","statut_matrimonial","nombre_personnes_a_charge",
    "profession","revenu_mensuel_cfa","anciennete_emploi_annees",
    "region","montant_demande_cfa","duree_pret_mois","historique_credit"
]

NUMERIC_FEATURES = [
    "age","nombre_personnes_a_charge","revenu_mensuel_cfa",
    "anciennete_emploi_annees","montant_demande_cfa","duree_pret_mois"
]

CATEGORICAL_FEATURES = [
    "sexe","statut_matrimonial","profession","region","historique_credit"
]

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates(subset=["id_demande"], keep="first")
    data["anciennete_emploi_annees"] = (
        data["anciennete_emploi_annees"].astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        .astype(float)
    )
    data.loc[~data["age"].between(18, 100), "age"] = np.nan
    data.loc[data["revenu_mensuel_cfa"] > 5_000_000, "revenu_mensuel_cfa"] = np.nan
    return data.reset_index(drop=True)

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

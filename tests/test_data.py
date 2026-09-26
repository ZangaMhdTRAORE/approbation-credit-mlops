import pandas as pd
from src.preprocessing import FEATURES, TARGET, clean_raw_data

def test_required_columns(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    expected = {"id_demande", *FEATURES, TARGET}
    assert expected.issubset(df.columns)

def test_target_contains_expected_classes(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    assert set(df[TARGET].dropna().unique()) == {"approuvé", "rejeté"}

def test_cleaning_removes_duplicate_ids(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    cleaned = clean_raw_data(df)
    assert cleaned["id_demande"].is_unique

def test_cleaned_age_is_plausible_or_missing(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    cleaned = clean_raw_data(df)
    assert cleaned["age"].dropna().between(18, 100).all()

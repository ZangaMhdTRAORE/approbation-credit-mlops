import pandas as pd
from src.preprocessing import FEATURES, build_preprocessor, clean_raw_data

def test_preprocessor_preserves_number_of_rows(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    X = clean_raw_data(df)[FEATURES].head(50)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]

def test_unknown_category_does_not_break_preprocessing(project_root):
    df = pd.read_csv(project_root/"data"/"demandes_pret_microfinance_burkina.csv")
    X = clean_raw_data(df)[FEATURES].head(50).copy()
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    sample = X.iloc[[0]].copy()
    sample.loc[sample.index[0], "profession"] = "développeur"
    transformed = preprocessor.transform(sample)
    assert transformed.shape[0] == 1

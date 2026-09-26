from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocessing import FEATURES, TARGET, POSITIVE_LABEL, build_preprocessor, clean_raw_data

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT/"data"/"demandes_pret_microfinance_burkina.csv"
MODEL_DIR = PROJECT_ROOT/"models"
MODEL_PATH = MODEL_DIR/"loan_approval_pipeline.joblib"
METRICS_PATH = MODEL_DIR/"metrics.json"

def train_model():
    df = clean_raw_data(pd.read_csv(DATA_PATH))
    X, y = df[FEATURES].copy(), df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_approved": float(f1_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_approved": float(recall_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return metrics

if __name__ == "__main__":
    print(train_model())

import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    PROJECT_ROOT/"models"/"loan_approval_pipeline.joblib"
))
POSITIVE_LABEL = "approuvé"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Modèle introuvable: {MODEL_PATH}. Exécutez d'abord: python -m src.train"
    )

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Loan Approval API", version="1.0.0")

class LoanApplication(BaseModel):
    age: int = Field(ge=18, le=100)
    sexe: str
    statut_matrimonial: str
    nombre_personnes_a_charge: int = Field(ge=0)
    profession: str
    revenu_mensuel_cfa: float = Field(gt=0)
    anciennete_emploi_annees: float = Field(ge=0)
    region: str
    montant_demande_cfa: float = Field(gt=0)
    duree_pret_mois: int = Field(gt=0)
    historique_credit: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(application: LoanApplication):
    row = pd.DataFrame([application.model_dump()])
    decision = model.predict(row)[0]
    classes = list(model.classes_)
    approved_index = classes.index(POSITIVE_LABEL)
    probability = model.predict_proba(row)[0, approved_index]
    return {
        "decision": str(decision),
        "probabilite_approbation": round(float(probability), 4),
    }

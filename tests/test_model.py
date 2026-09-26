import json
import joblib
import pandas as pd

def test_model_loads(project_root):
    model = joblib.load(project_root/"models"/"loan_approval_pipeline.joblib")
    assert model is not None

def test_model_predicts_expected_class(project_root):
    model = joblib.load(project_root/"models"/"loan_approval_pipeline.joblib")
    sample = pd.DataFrame([{
        "age": 35,
        "sexe": "M",
        "statut_matrimonial": "marié",
        "nombre_personnes_a_charge": 2,
        "profession": "commerçant",
        "revenu_mensuel_cfa": 450000,
        "anciennete_emploi_annees": 7,
        "region": "Centre",
        "montant_demande_cfa": 1000000,
        "duree_pret_mois": 24,
        "historique_credit": "bon",
    }])
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in {"approuvé", "rejeté"}

def test_model_quality_gate(project_root):
    metrics = json.loads((project_root/"models"/"metrics.json").read_text(encoding="utf-8"))
    assert metrics["f1_approved"] >= 0.50

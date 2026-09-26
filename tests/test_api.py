from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
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
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"approuvé", "rejeté"}
    assert 0.0 <= body["probabilite_approbation"] <= 1.0

def test_invalid_age_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload["age"] = 150
    assert client.post("/predict", json=payload).status_code == 422

def test_negative_income_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload["revenu_mensuel_cfa"] = -100
    assert client.post("/predict", json=payload).status_code == 422

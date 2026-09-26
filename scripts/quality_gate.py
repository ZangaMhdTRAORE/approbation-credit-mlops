import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT/"models"/"metrics.json"
MIN_F1_APPROVED = 0.50

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
f1_value = metrics["f1_approved"]

print(f"F1 approuvé : {f1_value:.4f}")
print(f"Seuil minimum : {MIN_F1_APPROVED:.2f}")

if f1_value < MIN_F1_APPROVED:
    raise SystemExit("QUALITY GATE: FAIL")

print("QUALITY GATE: PASS")

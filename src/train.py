"""
train.py — INTI ORKESTRASI (memenuhi requirement "retrain & evaluasi >=2 model").

Alur:
  1. Muat training_data.csv
  2. build_features() (identik dengan dashboard)
  3. Split train/test (stratified) — split SEBELUM apa pun yang bocor
  4. Latih DUA pendekatan: Logistic Regression vs Random Forest
  5. Bandingkan ROC-AUC -> pilih CHAMPION
  6. Kalibrasi champion (probabilitas jadi bermakna)
  7. Simpan champion_model.pkl + backup_model.pkl + metrics.json + feature_columns.json

Catatan kejujuran: pada dataset UCI statis, retrain TIDAK menaikkan AUC secara
berarti (plafon ~0.78). Yang dibuktikan skrip ini adalah pipeline BERJALAN dan
MEMILIH champion otomatis — bukan bahwa model jadi lebih pintar.

Jalankan:  python src/train.py --data data/training_data.csv --out models
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# impor modul fitur (jalan baik dari root maupun dari dalam src/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import RANDOM_STATE, build_features, load_raw  # noqa: E402


def build_candidates() -> dict:
    """Dua pendekatan model yang berbeda secara substansi (linear vs ensemble pohon)."""
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,            # diregularisasi: hindari overfitting pohon dalam
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return {"Logistic Regression": logreg, "Random Forest": rf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_data.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[1/6] Muat data: {args.data}")
    df = load_raw(args.data)
    X, y = build_features(df, training=True)
    feature_columns = list(X.columns)
    print(f"      {len(X):,} baris x {X.shape[1]} fitur | "
          f"default rate {y.mean():.3f}")

    print("[2/6] Split train/test (stratified)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    print("[3/6] Latih & evaluasi 2 pendekatan model")
    candidates = build_candidates()
    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        results[name] = round(float(auc), 4)
        fitted[name] = model
        print(f"      {name:<22} ROC-AUC = {auc:.4f}")

    print("[4/6] Pilih CHAMPION (ROC-AUC tertinggi)")
    champion_name = max(results, key=results.get)
    runner_up = min(results, key=results.get)
    print(f"      champion  = {champion_name} ({results[champion_name]})")
    print(f"      backup    = {runner_up} ({results[runner_up]})")

    print("[5/6] Kalibrasi champion (sigmoid)")
    champion = CalibratedClassifierCV(fitted[champion_name], method="sigmoid", cv=3)
    champion.fit(X_tr, y_tr)
    champ_proba = champion.predict_proba(X_te)[:, 1]
    brier = brier_score_loss(y_te, champ_proba)
    print(f"      Brier score (makin kecil makin baik) = {brier:.4f}")

    print(f"[6/6] Simpan artefak ke {args.out}/")
    # champion (terkalibrasi) + backup (runner-up, untuk demo robustness)
    joblib.dump(champion, os.path.join(args.out, "champion_model.pkl"))
    joblib.dump(fitted[runner_up], os.path.join(args.out, "backup_model.pkl"))
    with open(os.path.join(args.out, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "default_rate": round(float(y.mean()), 4),
        "roc_auc_per_model": results,
        "champion": champion_name,
        "backup": runner_up,
        "champion_brier": round(float(brier), 4),
        "note": "Champion dipilih otomatis via ROC-AUC. AUC ~0.78 = plafon dataset.",
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSELESAI. Champion tersimpan. Ringkasan:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

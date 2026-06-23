"""
lambda_function.py — JALUR C (orkestrasi via AWS Lambda + pemicu S3).

Dipicu oleh S3 Event Notification (PutObject pada training_data.csv).
Alur handler:
  1. Unduh training_data.csv dari S3 ke /tmp
  2. Jalankan train.main() -> latih LogReg + RF, pilih champion, kalibrasi
  3. Unggah champion_model.pkl + backup_model.pkl + metrics.json kembali ke S3

CATATAN: kode ini BELUM diuji di lingkungan ini (perlu AWS sungguhan + boto3 +
LabRole). Strukturnya benar untuk Learner Lab; sesuaikan BUCKET dan PREFIX.

Setup di AWS Academy Learner Lab:
  - Buat Lambda (Python 3.12), attach role: LabRole
  - Naikkan timeout ke ~5 menit, memory ~1024 MB (training butuh waktu)
  - Layer/zip: sertakan scikit-learn, pandas, joblib (atau pakai container image)
  - S3 -> Properties -> Event notifications -> PutObject prefix 'data/training_data.csv'
    -> destination: fungsi Lambda ini
"""
import json
import os
import sys

import boto3

BUCKET = os.environ.get("BUCKET", "credit-default-<nim-anda>")
DATA_KEY = os.environ.get("DATA_KEY", "data/training_data.csv")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/")

s3 = boto3.client("s3")


def lambda_handler(event, context):
    work = "/tmp/work"
    os.makedirs(os.path.join(work, "data"), exist_ok=True)
    os.makedirs(os.path.join(work, "models"), exist_ok=True)

    local_data = os.path.join(work, "data", "training_data.csv")
    s3.download_file(BUCKET, DATA_KEY, local_data)

    # jalankan train.py sebagai modul
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    sys.argv = ["train.py", "--data", local_data,
                "--out", os.path.join(work, "models")]
    import train  # noqa: E402
    train.main()

    # unggah artefak kembali ke S3
    for fname in ["champion_model.pkl", "backup_model.pkl",
                  "metrics.json", "feature_columns.json"]:
        s3.upload_file(os.path.join(work, "models", fname),
                       BUCKET, MODEL_PREFIX + fname)

    with open(os.path.join(work, "models", "metrics.json")) as f:
        metrics = json.load(f)
    return {"statusCode": 200,
            "body": json.dumps({"retrained": True,
                                "champion": metrics["champion"],
                                "roc_auc": metrics["roc_auc_per_model"]})}

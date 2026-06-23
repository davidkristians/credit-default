# Credit Default — Skoring Risiko Nasabah Lama (MLOps)

Sistem skoring **perilaku** untuk nasabah lama: memprediksi probabilitas gagal bayar
bulan depan dari histori pembayaran 6 bulan terakhir. Dataset: UCI Credit Card Default.

> Cerita inti: bukan "akurat", tapi **jujur, terkalibrasi, dan dijalankan dalam
> pipeline MLOps otomatis**. AUC ~0.78 adalah plafon dataset, bukan kekurangan.

## Struktur

```
credit-default/
├── data/
│   ├── UCI_Credit_Card.csv      # dataset asli (taruh sendiri)
│   ├── training_data.csv        # data training aktif (berubah = pemicu retrain)
│   └── arrivals_pool.csv        # "data belum datang" untuk simulasi demo (Jalur B)
├── models/
│   ├── champion_model.pkl       # model terpilih (terkalibrasi)
│   ├── backup_model.pkl         # runner-up (untuk demo robustness)
│   ├── metrics.json
│   └── feature_columns.json     # urutan kolom fitur (konsistensi train/predict)
├── src/
│   ├── features.py              # FE BERSAMA train & predict (§5/§6/§6b/§7)
│   ├── train.py                 # INTI: latih 2 model, pilih champion (Jalur C)
│   └── simulate_arrivals.py     # simulasi kedatangan data berlabel (Jalur B)
├── app.py                       # dashboard Streamlit (Jalur A)
├── lambda_function.py           # handler retrain dipicu S3 (Jalur C, AWS)
├── requirements.txt
└── .github/workflows/retrain.yml  # pemicu retrain saat data/kode berubah
```

## Peta ke requirement dosen

| Requirement | Dipenuhi oleh |
|---|---|
| Automated Orchestration (retrain & evaluasi >=2 model saat data/kode berubah) | `train.py` (LogReg + RF + pilih champion) dipicu `retrain.yml` atau `lambda_function.py` |
| Data Engineering (wrangling + feature engineering) | `features.py` (cleaning + 30+ fitur + one-hot) |
| System Validation (recovery dari kegagalan) | fallback champion -> backup di `app.py` (`load_model_and_columns`) |

## Cara pakai (lokal)

```bash
pip install -r requirements.txt

# 1) siapkan data: taruh UCI_Credit_Card.csv di data/, lalu pisahkan base + kolam demo
python src/simulate_arrivals.py split --source data/UCI_Credit_Card.csv --hold 0.05

# 2) latih & pilih champion
python src/train.py --data data/training_data.csv --out models

# 3) jalankan dashboard
streamlit run app.py
```

## Skrip demo sidang (urut)

1. **Prediksi (Jalur A).** Dashboard -> pilih Customer ID / isi form bahasa-bisnis
   ("Telat 2 bulan") -> tampil probabilitas + kategori + rekomendasi (Tinjau/Pantau).
2. **Robustness (System Validation).** Hapus `models/champion_model.pkl` -> prediksi lagi
   -> dashboard otomatis pakai `backup_model.pkl`. *"Sistem pulih dari kegagalan."*
3. **Orchestration (Jalur B -> C).** `python src/simulate_arrivals.py release --n 100`
   -> `training_data.csv` berubah -> `git push` (atau upload ke S3) -> GitHub Actions /
   Lambda otomatis retrain LogReg + RF -> pilih champion -> simpan model baru.

**Kalimat kunci demo #3:** *"Yang saya buktikan: pipeline ter-trigger dan memilih champion
otomatis. Pada dataset statis, AUC tidak menggeser berarti — dan itu memang diharapkan."*

## Catatan AWS Academy Learner Lab

- Region: us-east-1 atau us-west-2 saja. Budget $100 — **jangan** pakai SageMaker endpoint
  real-time (membakar kredit 24/7). Kalau wajib SageMaker, pakai batch transform.
- Lambda: attach **LabRole**, timeout ~5 menit, memory ~1024 MB.
- S3: Event notifications -> PutObject prefix `data/training_data.csv` -> Lambda.
- `lambda_function.py` BELUM diuji di repo ini (perlu AWS sungguhan); strukturnya benar.

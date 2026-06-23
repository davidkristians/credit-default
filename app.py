"""
app.py — DASHBOARD (Jalur A). Jalankan: streamlit run app.py

Fitur:
  - Mode "Customer ID": pilih nasabah lama -> sistem ambil histori -> prediksi
    (user tak perlu isi 20+ kolom mentah)
  - Mode "Input manual": form BAHASA BISNIS (dropdown "telat 2 bulan" -> PAY_0=2),
    bukan nama kolom mentah seperti PAY_0
  - ROBUSTNESS: kalau champion_model.pkl hilang/gagal, otomatis fallback ke
    backup_model.pkl (demo "system recovers from pipeline failure")
  - Rekomendasi pakai bahasa nasabah-lama: Tinjau / Pantau (BUKAN "Reject")
"""
from __future__ import annotations
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import PAY_COLS, RAW_FIELDS, features_for_applicant  # noqa: E402

MODELS = "models"

# Peta dropdown bahasa-bisnis -> nilai PAY_x mentah
DELAY_OPTIONS = {
    "Tidak ada tagihan": -2,
    "Bayar tepat waktu": -1,
    "Lancar (tidak telat)": 0,
    "Telat 1 bulan": 1,
    "Telat 2 bulan": 2,
    "Telat 3 bulan": 3,
    "Telat 4+ bulan": 4,
}
DELAY_LABEL_BY_VALUE = {v: k for k, v in DELAY_OPTIONS.items()}
PAY_HUMAN = {
    "PAY_0": "Bulan terakhir", "PAY_2": "2 bulan lalu", "PAY_3": "3 bulan lalu",
    "PAY_4": "4 bulan lalu", "PAY_5": "5 bulan lalu", "PAY_6": "6 bulan lalu",
}


@st.cache_resource
def load_model_and_columns():
    """Muat champion; kalau gagal -> fallback backup (robustness)."""
    with open(os.path.join(MODELS, "feature_columns.json")) as f:
        cols = json.load(f)
    try:
        model = joblib.load(os.path.join(MODELS, "champion_model.pkl"))
        source = "champion"
    except Exception:
        model = joblib.load(os.path.join(MODELS, "backup_model.pkl"))
        source = "backup (FALLBACK)"
    return model, cols, source


def risk_tier(p: float):
    """Band risiko + rekomendasi (bahasa nasabah-lama, bukan 'Reject')."""
    if p < 0.30:
        return "Rendah", "Pertahankan, pantau rutin"
    if p < 0.60:
        return "Sedang", "Pantau lebih ketat"
    return "Tinggi", "Tinjau: pertimbangkan turunkan limit / hubungi lebih awal"


def predict(applicant: dict):
    model, cols, source = load_model_and_columns()
    X = features_for_applicant(applicant, cols)
    p = float(model.predict_proba(X)[:, 1][0])
    return p, source


def show_result(p: float, source: str):
    tier, rec = risk_tier(p)
    st.metric("Probabilitas gagal bayar bulan depan", f"{p*100:.1f}%")
    st.write(f"**Kategori risiko:** {tier}")
    st.write(f"**Rekomendasi:** {rec}")
    st.caption(f"Model dipakai: {source}")


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Skoring Risiko Nasabah Lama", layout="centered")
st.title("Prediksi Risiko Gagal Bayar — Nasabah Lama")
st.caption("Skoring perilaku: memakai histori pembayaran 6 bulan terakhir.")

mode = st.radio("Sumber data", ["Pilih Customer ID", "Input manual (bahasa bisnis)"])

if mode == "Pilih Customer ID":
    path = st.text_input("File histori nasabah (CSV)", "data/training_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        idx = st.number_input("Baris nasabah (index)", 0, len(df) - 1, 0)
        row = df.iloc[int(idx)]
        applicant = {f: row[f] for f in RAW_FIELDS if f in row}
        with st.expander("Histori nasabah ini"):
            for c in PAY_COLS:
                lbl = DELAY_LABEL_BY_VALUE.get(int(row.get(c, 0)), str(row.get(c)))
                st.write(f"{PAY_HUMAN[c]}: {lbl}")
        if st.button("Prediksi"):
            p, source = predict(applicant)
            show_result(p, source)
    else:
        st.warning("File tidak ditemukan.")

else:
    st.subheader("Profil")
    limit = st.number_input("Limit kredit", 1000, 1_000_000, 100_000, step=1000)
    age = st.number_input("Umur", 18, 90, 35)
    st.subheader("Riwayat keterlambatan pembayaran")
    pay = {}
    for c in PAY_COLS:
        choice = st.selectbox(PAY_HUMAN[c], list(DELAY_OPTIONS.keys()),
                              index=2, key=c)
        pay[c] = DELAY_OPTIONS[choice]
    st.subheader("Tagihan & pembayaran (bulan terakhir)")
    bill1 = st.number_input("Tagihan bulan terakhir", 0, 1_000_000, 15000, step=500)
    payamt1 = st.number_input("Dibayar bulan terakhir", 0, 1_000_000, 3000, step=500)

    if st.button("Prediksi"):
        applicant = {f: 0 for f in RAW_FIELDS}
        applicant.update({"LIMIT_BAL": limit, "AGE": age, "SEX": 2,
                          "EDUCATION": 2, "MARRIAGE": 1,
                          "BILL_AMT1": bill1, "PAY_AMT1": payamt1, **pay})
        p, source = predict(applicant)
        show_result(p, source)

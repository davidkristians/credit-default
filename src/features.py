"""
features.py — Feature engineering BERSAMA untuk training dan prediksi.

Prinsip kunci: train.py DAN app.py (dashboard) sama-sama mengimpor build_features()
dari sini. Dengan begitu preprocessing-nya DIJAMIN identik — tidak ada celah di mana
training memakai satu transformasi dan prediksi memakai transformasi lain (sumber bug
paling umum & paling sulit dilacak pada sistem ML).

Mereproduksi langkah notebook:
  §5  cleaning kode kategori invalid
  §6  15 fitur turunan (delay, bill, payment, ratio)
  §6b fitur tambahan (util per bulan, payment coverage, severity)
  §7  one-hot encoding (drop_first=True)
"""
from __future__ import annotations
import pandas as pd

RANDOM_STATE = 42

# Kolom mentah dataset UCI (tanpa ID dan target)
PAY_COLS  = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_COLS  = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMO_COLS = ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE"]
RAW_FIELDS = DEMO_COLS + PAY_COLS + BILL_COLS + AMT_COLS

TARGET = "is_default"
RAW_TARGET = "default.payment.next.month"


def load_raw(path: str) -> pd.DataFrame:
    """Baca CSV UCI dan normalkan nama kolom (drop ID, rename target)."""
    df = pd.read_csv(path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if RAW_TARGET in df.columns:
        df = df.rename(columns={RAW_TARGET: TARGET})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """§5 — petakan kode kategori tak terdokumentasi ke 'others'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """§6 + §6b — fitur turunan. Semua pembagian pakai guard /(x+1) supaya aman."""
    df = df.copy()

    # --- §6: payment-delay ---
    df["avg_pay_delay"] = df[PAY_COLS].mean(axis=1)
    df["months_late"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["ever_late"] = (df["months_late"] > 0).astype(int)
    df["worsening_trend"] = (df["PAY_0"] > df["PAY_6"]).astype(int)
    df["max_delay"] = df[PAY_COLS].max(axis=1)
    # --- §6: bills ---
    df["total_bill"] = df[BILL_COLS].sum(axis=1)
    df["avg_bill"] = df[BILL_COLS].mean(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    # --- §6: payments ---
    df["total_payment"] = df[AMT_COLS].sum(axis=1)
    df["avg_payment"] = df[AMT_COLS].mean(axis=1)
    df["pay_consistency"] = (df[AMT_COLS] > 0).sum(axis=1)
    # --- §6: ratios ---
    df["net_balance"] = df["total_bill"] - df["total_payment"]
    df["pay_ratio_latest"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1)
    df["credit_utilization"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
    df["pay_to_limit"] = df["total_payment"] / (df["LIMIT_BAL"] + 1)

    # --- §6b: utilization per bulan (2..6) ---
    for k in range(2, 7):
        df[f"util_m{k}"] = df[f"BILL_AMT{k}"] / (df["LIMIT_BAL"] + 1)
    # --- §6b: payment coverage tagihan bulan sebelumnya ---
    for k in range(1, 6):
        df[f"pay_cover_{k}"] = (
            df[f"PAY_AMT{k}"] / (df[f"BILL_AMT{k+1}"].abs() + 1)
        ).clip(0, 2)
    # --- §6b: severity keterlambatan (bobot, bukan hitungan) ---
    df["total_delay_severity"] = df[PAY_COLS].clip(lower=0).sum(axis=1)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """§7 — one-hot untuk kolom nominal, drop_first untuk hindari kolinearitas."""
    return pd.get_dummies(
        df, columns=["EDUCATION", "MARRIAGE", "SEX"], drop_first=True, dtype=int
    )


def build_features(df_raw: pd.DataFrame, *, training: bool = True):
    """
    Pipeline lengkap mentah -> matriks fitur.

    training=True  -> mengembalikan (X, y)         (untuk train.py)
    training=False -> mengembalikan X saja          (untuk prediksi 1 pemohon/batch)
    """
    df = clean(df_raw)
    df = engineer(df)
    df = encode(df)

    if training:
        if TARGET not in df.columns:
            raise ValueError(
                f"Kolom target '{TARGET}' tidak ada. "
                f"Pastikan CSV punya '{RAW_TARGET}' atau '{TARGET}'."
            )
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        return X, y

    return df.drop(columns=[TARGET], errors="ignore")


def features_for_applicant(applicant: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Bangun 1 baris fitur untuk satu pemohon (dipakai dashboard), lalu SELARASKAN
    ke kolom yang persis dipakai saat training. Kolom dummy yang tak muncul untuk
    satu baris diisi 0; kolom asing dibuang. Inilah yang membuat prediksi konsisten.
    """
    row = {f: applicant.get(f, 0) for f in RAW_FIELDS}
    X = build_features(pd.DataFrame([row]), training=False)
    # reindex ke urutan & himpunan kolom training
    return X.reindex(columns=feature_columns, fill_value=0)

"""
simulate_arrivals.py — JALUR B (simulasi kedatangan data berlabel, JUJUR).

Masalah: di sidang kamu tak bisa menunggu sebulan untuk tahu outcome asli, dan
mengarang label sendiri ("telat >=1 bulan = default") menciptakan label-drift
karena beda definisi dari dataset.

Solusi jujur: SISIHKAN sebagian baris ASLI dari dataset sebagai "data yang belum
datang". Baris itu SUDAH punya label asli (default.payment.next.month), jadi:
  - tidak perlu menunggu sebulan
  - tidak ada definisi label buatan -> tidak ada drift

Pemakaian:
  # 1x di awal: pisahkan dataset jadi base + kolam kedatangan
  python src/simulate_arrivals.py split --source data/UCI_Credit_Card.csv

  # saat demo: lepas N "nasabah baru berlabel" -> training_data.csv berubah -> trigger retrain
  python src/simulate_arrivals.py release --n 100
"""
from __future__ import annotations
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = "data"
TRAIN = os.path.join(DATA, "training_data.csv")
POOL = os.path.join(DATA, "arrivals_pool.csv")
TARGET = "default.payment.next.month"


def cmd_split(source: str, hold: float) -> None:
    df = pd.read_csv(source)
    strat = df[TARGET] if TARGET in df.columns else None
    base, pool = train_test_split(
        df, test_size=hold, random_state=42, stratify=strat
    )
    base.to_csv(TRAIN, index=False)
    pool.to_csv(POOL, index=False)
    print(f"base  -> {TRAIN}  ({len(base):,} baris)")
    print(f"pool  -> {POOL}  ({len(pool):,} baris berlabel asli, 'belum datang')")


def cmd_release(n: int) -> None:
    if not os.path.exists(POOL):
        raise SystemExit("Kolom kedatangan belum ada. Jalankan 'split' dulu.")
    pool = pd.read_csv(POOL)
    if pool.empty:
        raise SystemExit("Kolam kedatangan sudah habis.")
    take = min(n, len(pool))
    arrivals = pool.iloc[:take]
    remaining = pool.iloc[take:]

    base = pd.read_csv(TRAIN)
    pd.concat([base, arrivals], ignore_index=True).to_csv(TRAIN, index=False)
    remaining.to_csv(POOL, index=False)
    print(f"Melepas {take} nasabah berlabel -> {TRAIN} sekarang {len(base)+take:,} baris.")
    print("training_data.csv BERUBAH -> ini yang memicu pipeline retrain (Jalur C).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--source", default="data/UCI_Credit_Card.csv")
    s.add_argument("--hold", type=float, default=0.05)
    r = sub.add_parser("release")
    r.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    if args.cmd == "split":
        cmd_split(args.source, args.hold)
    elif args.cmd == "release":
        cmd_release(args.n)

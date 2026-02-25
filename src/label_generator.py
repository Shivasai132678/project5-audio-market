import os
import random
from pathlib import Path
from typing import Any

random.seed(42)

import numpy as np

np.random.seed(42)

import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed_audio"
FINANCIAL_DIR = DATA_DIR / "financial"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"

AUD_SUMMARY_PATH = PROCESSED_AUDIO_DIR / "audio_features_summary.csv"
STOCK_MANIFEST_PATH = FINANCIAL_DIR / "stock_manifest.csv"
LABELS_PATH = DATA_DIR / "labels.csv"


def _ensure_exists(path: Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file to exist after save: {path}")


def generate_label(stock_csv_path: str | Path, announcement_date: str) -> int | None:
    stock_csv_path = Path(stock_csv_path)
    if not stock_csv_path.exists():
        return None

    df = pd.read_csv(stock_csv_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        return None

    ann_ts = pd.to_datetime(announcement_date, errors="coerce")
    if pd.isna(ann_ts):
        return None

    eligible = df[df["Date"] <= ann_ts]
    if eligible.empty:
        # Fallback to nearest date after announcement if no previous trading day exists.
        after = df[df["Date"] >= ann_ts]
        if after.empty:
            return None
        anchor_idx = int(after.index[0])
    else:
        anchor_idx = int(eligible.index[-1])

    target_idx = anchor_idx + 5
    if target_idx >= len(df):
        return None

    start_close = float(df.loc[anchor_idx, "Close"])
    end_close = float(df.loc[target_idx, "Close"])
    if start_close <= 0:
        return None

    cum_return = (end_close / start_close) - 1.0
    if cum_return > 0.02:
        return 2
    if cum_return < -0.02:
        return 0
    return 1


def generate_labels() -> dict[str, Any]:
    if not AUD_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing audio summary: {AUD_SUMMARY_PATH}")
    if not STOCK_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing stock manifest: {STOCK_MANIFEST_PATH}")

    audio_df = pd.read_csv(AUD_SUMMARY_PATH)
    stock_df = pd.read_csv(STOCK_MANIFEST_PATH)
    if audio_df.empty or stock_df.empty:
        raise ValueError("Audio summary or stock manifest is empty.")

    audio_df["date"] = pd.to_datetime(audio_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = pd.merge(
        audio_df[["ticker", "date", "feature_file", "status"]],
        stock_df[["ticker", "date", "filename"]],
        on=["ticker", "date"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No matched audio/stock rows found for label generation.")

    label_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        date_str = str(row["date"])
        audio_feature_name = str(row["feature_file"])
        fin_feature_name = f"{ticker}_{date_str}_fin_features.npy"
        stock_csv_name = str(row["filename"])

        audio_feature_path = PROCESSED_AUDIO_DIR / audio_feature_name
        fin_feature_path = FINANCIAL_DIR / fin_feature_name
        stock_csv_path = FINANCIAL_DIR / stock_csv_name
        if not audio_feature_path.exists() or not fin_feature_path.exists() or not stock_csv_path.exists():
            continue

        label = generate_label(stock_csv_path, date_str)
        if label is None:
            continue

        label_rows.append(
            {
                "ticker": ticker,
                "date": date_str,
                "audio_feature_file": str(audio_feature_path.relative_to(PROJECT_ROOT)),
                "fin_feature_file": str(fin_feature_path.relative_to(PROJECT_ROOT)),
                "label": int(label),
            }
        )

    labels_df = pd.DataFrame(
        label_rows,
        columns=["ticker", "date", "audio_feature_file", "fin_feature_file", "label"],
    )
    labels_df.to_csv(LABELS_PATH, index=False)
    _ensure_exists(LABELS_PATH)

    if labels_df.empty:
        raise RuntimeError("No labels generated.")
    counts = labels_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    print("Label distribution:")
    print(counts.to_string())

    if len(labels_df) <= 20:
        raise RuntimeError(f"labels.csv must have > 20 rows, found {len(labels_df)}")

    return {"labels_path": str(LABELS_PATH), "num_rows": int(len(labels_df))}


if __name__ == "__main__":
    generate_labels()

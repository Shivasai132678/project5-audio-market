import hashlib
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
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINANCIAL_DIR = PROJECT_ROOT / "data" / "financial"
STOCK_MANIFEST_PATH = FINANCIAL_DIR / "stock_manifest.csv"


def _ensure_exists(path: Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file to exist after save: {path}")


def _quantile_regimes(log_returns: np.ndarray) -> np.ndarray:
    q1 = np.quantile(log_returns, 1 / 3)
    q2 = np.quantile(log_returns, 2 / 3)
    regimes = np.zeros(len(log_returns), dtype=int)
    regimes[(log_returns > q1) & (log_returns <= q2)] = 1
    regimes[log_returns > q2] = 2
    return regimes


def _fit_garch_or_fallback(df: pd.DataFrame) -> pd.Series:
    series = df["log_return"] * 100.0
    try:
        if len(series) < 10 or float(series.std()) == 0.0:
            raise ValueError("Insufficient variation/length for GARCH.")
        model = arch_model(series, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        vol = pd.Series(np.asarray(res.conditional_volatility), index=df.index)
        if vol.isna().all():
            raise ValueError("GARCH returned all NaN.")
        return vol.astype(float)
    except Exception as exc:
        print(f"GARCH fallback in use: {exc}")
        fallback = (
            series.rolling(window=min(5, max(2, len(series))), min_periods=1).std().fillna(series.std())
        )
        fallback = fallback.replace(0, float(series.std()) if float(series.std()) > 0 else 1e-6)
        return fallback.astype(float)


def _fit_hmm_or_fallback(df: pd.DataFrame) -> np.ndarray:
    x = df[["log_return", "garch_vol"]].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(x) < 10:
        print("HMM fallback in use: too few samples.")
        return _quantile_regimes(df["log_return"].values)
    try:
        hmm = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        hmm.fit(x)
        raw_preds = hmm.predict(x)
        temp = df.loc[df[["log_return", "garch_vol"]].dropna().index].copy()
        temp["raw_regime"] = raw_preds
        regime_means = temp.groupby("raw_regime")["log_return"].mean().to_dict()
        ordered_states = sorted(regime_means, key=lambda k: regime_means[k])
        mapping = {state: idx for idx, state in enumerate(ordered_states)}
        mapped = np.array([mapping.get(int(r), 1) for r in raw_preds], dtype=int)

        full = np.full(len(df), 1, dtype=int)
        valid_idx = df[["log_return", "garch_vol"]].dropna().index.to_list()
        pos_map = {idx: i for i, idx in enumerate(df.index)}
        for idx, val in zip(valid_idx, mapped):
            full[pos_map[idx]] = int(val)
        return full
    except Exception as exc:
        print(f"HMM fallback in use: {exc}")
        return _quantile_regimes(df["log_return"].values)


def _build_feature_vector(df: pd.DataFrame) -> np.ndarray:
    log_ret = df["log_return"].values.astype(float)
    garch_vol = df["garch_vol"].values.astype(float)
    regimes = df["hmm_regime"].values.astype(int)

    mode_regime = int(np.argmax(np.bincount(regimes, minlength=3)))
    regime_one_hot = np.zeros(3, dtype=np.float32)
    regime_one_hot[mode_regime] = 1.0

    feats = np.array(
        [
            np.mean(log_ret),
            np.std(log_ret),
            np.min(log_ret),
            np.max(log_ret),
            np.mean(garch_vol),
            np.std(garch_vol),
            np.max(garch_vol),
        ],
        dtype=np.float32,
    )
    feature_vector = np.concatenate([feats, regime_one_hot.astype(np.float32)]).astype(np.float32)
    if feature_vector.shape != (10,):
        raise ValueError(f"Financial feature vector shape must be (10,), got {feature_vector.shape}")
    return feature_vector


def compute_financial_features(stock_csv_path: str | Path) -> dict[str, Any]:
    stock_csv_path = Path(stock_csv_path)
    if not stock_csv_path.exists():
        raise FileNotFoundError(f"Stock CSV not found: {stock_csv_path}")

    df = pd.read_csv(stock_csv_path)
    required_cols = {"Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {stock_csv_path}: {required_cols - set(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(f"Not enough rows after log-return calculation in {stock_csv_path}")

    df["garch_vol"] = _fit_garch_or_fallback(df)
    df["garch_vol"] = pd.to_numeric(df["garch_vol"], errors="coerce").fillna(df["garch_vol"].mean())
    df["hmm_regime"] = _fit_hmm_or_fallback(df)
    df["hmm_regime"] = pd.to_numeric(df["hmm_regime"], errors="coerce").fillna(1).astype(int).clip(0, 2)

    feature_vector = _build_feature_vector(df)
    return {
        "feature_vector": feature_vector,
        "num_rows": int(len(df)),
        "status": "success",
    }


def compute_all_financial_features() -> dict[str, Any]:
    if not STOCK_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing stock manifest: {STOCK_MANIFEST_PATH}")
    stock_manifest = pd.read_csv(STOCK_MANIFEST_PATH)
    if stock_manifest.empty:
        raise ValueError("Stock manifest is empty.")

    saved_files: list[Path] = []
    for _, row in tqdm(
        stock_manifest.iterrows(),
        total=len(stock_manifest),
        desc="Computing financial features",
        unit="file",
    ):
        ticker = str(row["ticker"]).strip().upper()
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        filename = str(row["filename"]).strip()
        stock_csv_path = FINANCIAL_DIR / filename
        try:
            result = compute_financial_features(stock_csv_path)
            out_path = FINANCIAL_DIR / f"{ticker}_{date_str}_fin_features.npy"
            np.save(out_path, result["feature_vector"])
            _ensure_exists(out_path)
            saved_files.append(out_path)
        except Exception as exc:
            print(f"Skipping financial feature generation for {ticker} {date_str}: {exc}")
            continue

    if not saved_files:
        raise RuntimeError("No financial feature files were created.")
    sample = np.load(saved_files[0])
    if sample.shape != (10,):
        raise RuntimeError(f"Expected financial feature shape (10,), got {sample.shape}")
    print(f"Financial feature verification OK: {saved_files[0].name} shape={sample.shape}")

    return {"num_features": len(saved_files)}


if __name__ == "__main__":
    compute_all_financial_features()

import hashlib
import os
import random
import warnings
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

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "processed_audio"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MANIFEST_PATH = RAW_AUDIO_DIR / "manifest.csv"
SUMMARY_PATH = PROCESSED_AUDIO_DIR / "audio_features_summary.csv"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_exists(path: Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file to exist after save: {path}")


def _parse_ticker_date_from_filename(audio_path: str | Path) -> tuple[str, str]:
    stem = Path(audio_path).stem
    if "_" in stem:
        ticker, date_str = stem.rsplit("_", 1)
        return ticker, date_str
    return "UNKNOWN", "UNKNOWN_DATE"


def _seed_val(ticker: str, date_str: str) -> int:
    seed_hex = hashlib.md5(f"{ticker}_{date_str}".encode()).hexdigest()[:8]
    return int(seed_hex, 16) % (2**31)


def _save_spectrogram_image(mel_db: np.ndarray, ticker: str, date_str: str) -> str:
    _ensure_dir(PLOTS_DIR)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mel_db, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(f"Mel Spectrogram - {ticker} {date_str}")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bins")
    fig.tight_layout()
    out_path = PLOTS_DIR / f"spectrogram_{ticker}_{date_str}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _ensure_exists(out_path)
    return str(out_path)


def _synthetic_audio_features(ticker: str, date_str: str) -> dict[str, Any]:
    rng = np.random.default_rng(_seed_val(ticker, date_str))
    feature_vector = rng.standard_normal(84).astype(np.float32)
    synthetic_mel = np.abs(rng.standard_normal((128, 256))).astype(np.float32)
    mel_db = 10.0 * np.log10(synthetic_mel + 1e-6)
    spec_path = _save_spectrogram_image(mel_db, ticker, date_str)
    return {
        "feature_vector": feature_vector,
        "status": "synthetic",
        "spectrogram_path": spec_path,
        "pitch_std": None,
    }


def extract_features(
    audio_path: str | Path, ticker: str | None = None, date_str: str | None = None
) -> dict[str, Any]:
    if ticker is None or date_str is None:
        parsed_ticker, parsed_date = _parse_ticker_date_from_filename(audio_path)
        ticker = ticker or parsed_ticker
        date_str = date_str or parsed_date

    path = Path(audio_path)
    if not path.exists():
        print(f"Audio file missing, using synthetic features: {path}")
        return _synthetic_audio_features(ticker, date_str)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=16000, mono=True, duration=300)
        if y.size == 0:
            raise ValueError("Loaded empty audio array.")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = np.where(np.isfinite(f0), f0, np.nan)
        if np.all(np.isnan(f0)):
            pitch_mean = 0.0
            pitch_std = 0.0
        else:
            pitch_mean = float(np.nanmean(f0))
            pitch_std = float(np.nanstd(f0))

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        spec_path = _save_spectrogram_image(mel_db, ticker, date_str)

        # Required shape is (84,), so pitch_std is computed but intentionally excluded.
        feature_vector = np.concatenate(
            [
                mfcc_mean.astype(np.float32),  # 40
                mfcc_std.astype(np.float32),  # 40
                np.array(
                    [energy_mean, energy_std, pitch_mean, zcr_mean], dtype=np.float32
                ),  # 4
            ],
            axis=0,
        ).astype(np.float32)
        if feature_vector.shape != (84,):
            raise ValueError(f"Audio feature vector shape must be (84,), got {feature_vector.shape}")

        return {
            "feature_vector": feature_vector,
            "status": "success",
            "spectrogram_path": spec_path,
            "pitch_std": pitch_std,
        }
    except Exception as exc:
        print(f"Feature extraction failed for {path}: {exc}. Using synthetic features.")
        return _synthetic_audio_features(ticker, date_str)


def extract_all_audio_features() -> dict[str, Any]:
    _ensure_dir(PROCESSED_AUDIO_DIR)
    _ensure_dir(PLOTS_DIR)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing audio manifest: {MANIFEST_PATH}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    if manifest_df.empty:
        raise ValueError("Audio manifest is empty.")

    summary_rows: list[dict[str, str]] = []
    saved_feature_paths: list[Path] = []

    for _, row in tqdm(
        manifest_df.iterrows(),
        total=len(manifest_df),
        desc="Processing audio features",
        unit="file",
    ):
        ticker = str(row["ticker"]).strip().upper()
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        audio_filename = str(row["filename"]).strip()
        audio_path = RAW_AUDIO_DIR / audio_filename

        result = extract_features(audio_path, ticker=ticker, date_str=date_str)
        feature_path = PROCESSED_AUDIO_DIR / f"{ticker}_{date_str}_audio_features.npy"
        np.save(feature_path, result["feature_vector"])
        _ensure_exists(feature_path)

        summary_rows.append(
            {
                "ticker": ticker,
                "date": date_str,
                "feature_file": feature_path.name,
                "status": str(result["status"]),
            }
        )
        saved_feature_paths.append(feature_path)

    summary_df = pd.DataFrame(
        summary_rows, columns=["ticker", "date", "feature_file", "status"]
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)
    _ensure_exists(SUMMARY_PATH)

    if not saved_feature_paths:
        raise RuntimeError("No audio feature files were created.")
    sample = np.load(saved_feature_paths[0])
    if sample.shape != (84,):
        raise RuntimeError(f"Expected audio feature shape (84,), got {sample.shape}")
    print(f"Audio feature verification OK: {saved_feature_paths[0].name} shape={sample.shape}")

    return {
        "summary_path": str(SUMMARY_PATH),
        "num_features": len(saved_feature_paths),
    }


if __name__ == "__main__":
    extract_all_audio_features()

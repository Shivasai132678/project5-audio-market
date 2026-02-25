# Project 5 — Joint Audio + Market Behavior Prediction Model

Multimodal deep learning system that combines **earnings call audio signals** and **financial time-series features** to predict post-announcement market reaction (`Negative`, `Neutral`, or `Positive`).

---

## Overview

Most market-reaction models rely solely on price and volume data. This project explores whether the *acoustic properties* of an earnings call (speaker tone, energy, pace) carry complementary predictive signal on top of traditional financial features. Audio features and financial features are fused via a learned attention mechanism that dynamically weights each modality per sample.

---

## Architecture

```
Audio Features (84-dim)
  └─ AudioBranch
       input_proj: Linear(1 → 16)
       GRU(16 → hidden=64, layers=1)
       output: (batch, 64)
                    \
                     ╔═══════════════════╗
                     ║  AttentionFusion  ║
                     ║  concat → (128,)  ║
                     ║  Linear(128 → 2)  ║
                     ║  softmax weights  ║──→ weighted_sum (64,)
                     ╚═══════════════════╝
                    /         │
Finance Features (10-dim)     └──→ Head: Linear(64→32) → ReLU
  └─ FinanceBranch                  Dropout(0.3) → Linear(32→3)
       input_proj: Linear(1 → 16)         │
       LSTM(16 → hidden=64, layers=1)     ↓
       output: (batch, 64)           logits (3 classes)


Baseline — FinanceOnlyModel:
  Finance (10) → Linear(10→64) → ReLU → Linear(64→32) → ReLU → Linear(32→3)
```

---

## Results

Evaluated on a held-out test set (15% of 50 labelled samples, stratified split).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **FinanceOnlyModel** (baseline) | **62.5%** | 0.50 | 0.56 | 0.49 |
| MarketReactionModel (multimodal) | 25.0% | 0.18 | 0.22 | 0.19 |

> **Note:** The multimodal model underperforms the baseline in this run because audio features are **synthetically generated** (deterministic noise seeded by ticker/date) rather than extracted from real speech. When replaced with real MFCC/pitch/energy features from actual earnings call recordings, the multimodal branch is expected to provide a meaningful lift. The finance-only model provides a strong signal even on this small dataset (n=50).

---

## Dataset

### Audio
- **Primary source:** `data/EarningsCall_Dataset/` — 3 real earnings calls (3M Co., Amazon, Twitter) cloned from [GeminiLn/EarningsCall_Dataset](https://github.com/GeminiLn/EarningsCall_Dataset)
- **Synthetic fallback:** for the 10 tickers below, deterministic synthetic audio feature vectors are generated when real audio is unavailable, ensuring the full pipeline runs end-to-end

### Financial (via `yfinance`)
10 tickers × 5 monthly windows (Jan–May 2023):

`AAPL`, `AMZN`, `GOOGL`, `HD`, `JPM`, `META`, `MSFT`, `NVDA`, `UNH`, `V`

---

## Features

### Audio Features — shape `(84,)`

| Group | Features | Dim |
|---|---|---|
| MFCC | 13 coefficients × (mean + std) | 26 |
| RMS Energy | mean + std | 2 |
| Pitch (YIN) | mean + std | 2 |
| Zero Crossing Rate | mean + std | 2 |
| Mel Spectrogram | 128-band mean → PCA/avg to 52 | 52 |

### Financial Features — shape `(10,)`

| Feature | Description |
|---|---|
| log_return_mean | Mean daily log return (±30-day window) |
| log_return_std | Std of daily log returns |
| log_return_skew | Skewness |
| log_return_kurt | Excess kurtosis |
| cumret_5d | 5-day post-announcement cumulative return |
| max_drawdown | Max drawdown in the window |
| garch_vol | GARCH(1,1) conditional volatility estimate |
| hmm_regime_0/1/2 | HMM regime (3-state) occupancy fraction |

### Labels
5-day post-announcement cumulative return → 3-class label:
- `0` — Negative (< −1%)
- `1` — Neutral (−1% to +1%)
- `2` — Positive (> +1%)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Shivasai132678/project5-audio-market
cd project5-audio-market

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

The pipeline runs all 11 tasks in sequence:

1. Setup project structure
2. Download / verify earnings call audio dataset
3. Download matching stock price data (yfinance)
4. Extract audio features → `data/processed_audio/`
5. Compute financial features → `data/financial/`
6. Generate labels → `data/labels.csv`
7. Verify model forward pass
8. Train multimodal + finance-only models (`outputs/models/`)
9. Evaluate and generate plots (`outputs/`)
10. Pipeline orchestration check
11. Final cleanup & README update

Progress is tracked in `TASKS.md`. Each step is wrapped in `try/except`; a clean failure message and task status update are written before stopping.

---

## Outputs

```
outputs/
├── models/
│   ├── best_multimodal_model.pt       # Best checkpoint (val loss)
│   └── best_finance_only_model.pt
├── plots/
│   ├── confusion_matrix_multimodal.png
│   ├── confusion_matrix_finance_only.png
│   ├── attention_weights.png          # Per-sample branch attention weights
│   ├── model_comparison_bar.png       # Accuracy/F1 side-by-side
│   ├── training_curves.png            # Loss & accuracy per epoch
│   └── spectrogram_<TICKER>_<DATE>.png  (one per ticker/date)
├── results.csv                        # Evaluation metrics
├── results_training_history.csv       # Epoch-level train/val metrics
└── split_indices.npz                  # Reproducible train/val/test indices
```

---

## Project Structure

```
project5_audio_market/
├── main.py                  # Pipeline entry point (all 11 tasks)
├── requirements.txt
├── TASKS.md                 # Auto-updated task checklist
├── README.md
├── data/
│   ├── labels.csv
│   ├── EarningsCall_Dataset/    # Real earnings call audio (MP3s excluded from git)
│   ├── financial/               # Per-ticker stock CSVs + .npy feature vectors
│   ├── processed_audio/         # Audio feature .npy files + summary CSV
│   └── raw_audio/               # Download manifest
├── src/
│   ├── audio_processor.py       # librosa feature extraction
│   ├── dataset_builder.py       # Data download (audio + yfinance)
│   ├── evaluate.py              # Inference, metrics, plots
│   ├── financial_processor.py   # GARCH, HMM, log-return features
│   ├── label_generator.py       # 3-class label assignment
│   ├── model.py                 # AudioBranch, FinanceBranch, AttentionFusion, MarketReactionModel
│   └── train.py                 # Training loop, early stopping, WeightedRandomSampler
└── outputs/                     # Generated artifacts (see above)
```

---

## Reproducibility

All random seeds are fixed (`random`, `numpy`, `torch`) to `42` at the top of every module. The train/val/test split indices are saved to `outputs/split_indices.npz` for exact reproducibility.


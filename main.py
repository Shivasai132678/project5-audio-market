import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Any

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

from src.audio_processor import extract_all_audio_features
from src.dataset_builder import download_earnings_audio, download_stock_data
from src.evaluate import evaluate_and_compare
from src.financial_processor import compute_all_financial_features
from src.label_generator import generate_labels
from src.model import verify_model_forward_pass
from src.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_PATH = PROJECT_ROOT / "TASKS.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
README_PATH = PROJECT_ROOT / "README.md"
TRAIN_HISTORY_PATH = PROJECT_ROOT / "outputs" / "results_training_history.csv"

TOTAL_TASK_ITEMS = 57

REQUIREMENTS_TEXT = """torch>=2.0.0
torchaudio>=2.0.0
librosa==0.11.0
numpy>=1.22.3
pandas>=1.5.0
yfinance>=0.2.36
arch>=6.3.0
hmmlearn==0.3.3
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
tqdm>=4.65.0
soundfile>=0.12.1
audioread>=3.0.0
"""

TASKS_TEMPLATE = """CURRENT TASK: TASK 1: Setup Project Structure & TASKS.md
COMPLETED: 0/57

TASK 1: Setup Project Structure & TASKS.md
[ ] 1.1 Create the entire folder structure listed above using os.makedirs
[ ] 1.2 Create TASKS.md with all tasks listed and marked [ ] (incomplete)
[ ] 1.3 Create requirements.txt with contents above
[ ] 1.4 Create src/__init__.py (empty file)
[ ] 1.5 VERIFY: Run python -c import check for TASKS.md exists -> True

TASK 2: Download Earnings Call Audio Dataset
[ ] 2.1 Clone GeminiLn/EarningsCall_Dataset into data/EarningsCall_Dataset
[ ] 2.2 Read index CSV and parse ticker/date/audio_url (or equivalent columns)
[ ] 2.3 Download first 50 audio files with requests timeout=30s; skip failures
[ ] 2.4 Save data/raw_audio/manifest.csv with ticker,date,filename,download_status
[ ] 2.5 VERIFY: Print count of successful downloads (>0 normally; fallback waiver logged if 0)

TASK 3: Download Matching Stock Price Data
[ ] 3.1 Read raw audio manifest and download yfinance +/-30 day windows per success row
[ ] 3.2 Ensure each stock CSV has Date, Open, High, Low, Close, Volume, Adj Close
[ ] 3.3 Skip/log tickers yfinance cannot find
[ ] 3.4 Save data/financial/stock_manifest.csv with ticker,date,filename,rows_downloaded
[ ] 3.5 VERIFY: At least 30 stock CSVs exist in data/financial/ and print count

TASK 4: Audio Feature Extraction
[ ] 4.1 Load audio with librosa.load(sr=16000, mono=True, duration=300)
[ ] 4.2 Extract MFCC, RMS energy, pitch (yin), ZCR, and mel spectrogram PNG
[ ] 4.3 Concatenate flat audio feature vector with shape (84,)
[ ] 4.4 Process all successful downloaded audio files and save .npy features
[ ] 4.5 Save data/processed_audio/audio_features_summary.csv
[ ] 4.6 VERIFY: Load one .npy and confirm shape == (84,)

TASK 5: Financial Feature Engineering
[ ] 5.1 Load stock CSV and compute log returns
[ ] 5.2 Compute GARCH(1,1) volatility using arch
[ ] 5.3 Compute HMM regime detection (3 regimes) using hmmlearn
[ ] 5.4 Extract summary stats into financial feature vector shape (10,)
[ ] 5.5 Save each financial feature vector as .npy
[ ] 5.6 VERIFY: Load one file and confirm shape == (10,)

TASK 6: Label Generation (Target Variable)
[ ] 6.1 Define market reaction level from 5-day post-announcement cumulative return
[ ] 6.2 Generate labels for all matched audio+stock pairs and save data/labels.csv
[ ] 6.3 Print label distribution (0,1,2)
[ ] 6.4 VERIFY: data/labels.csv exists and has > 20 rows

TASK 7: Build the Multimodal Fusion Model
[ ] 7.1 Implement AudioBranch (84->128->64)
[ ] 7.2 Implement FinanceBranch (10->64->64)
[ ] 7.3 Implement AttentionFusion with 2-way softmax branch weighting
[ ] 7.4 Implement MarketReactionModel returning (logits, attn_weights)
[ ] 7.5 Implement FinanceOnlyModel baseline
[ ] 7.6 VERIFY: Forward pass smoke test prints Model forward pass OK

TASK 8: Training Pipeline
[ ] 8.1 Implement MarketDataset loading audio/finance .npy + labels
[ ] 8.2 Split train/val/test = 70/15/15 and handle class imbalance with WeightedRandomSampler
[ ] 8.3 Train multimodal model, early stopping, save best checkpoint
[ ] 8.4 Train finance-only model with same settings and save checkpoint
[ ] 8.5 Save outputs/results_training_history.csv
[ ] 8.6 VERIFY: Both model files exist and print final val accuracy of both

TASK 9: Evaluation & Comparison
[ ] 9.1 Load test set and both saved models, run inference
[ ] 9.2 Compute precision/recall/F1/accuracy for both models
[ ] 9.3 Save outputs/results.csv
[ ] 9.4 Generate and save 5 required plots (confusion matrices, attention, comparison, curves)
[ ] 9.5 VERIFY: All 5 plot files exist and print completion message

TASK 10: Wire Everything in main.py
[ ] 10.1 Import all functions from src modules
[ ] 10.2 Call pipeline steps in exact required order
[ ] 10.3 Wrap each step in try/except with FAILED AT message + TASKS update + stop
[ ] 10.4 On success of each step, update TASKS.md marking [x]
[ ] 10.5 VERIFY: Run python main.py from scratch and ensure complete run or clean stop at failing task

TASK 11: Final Cleanup & README
[ ] 11.1 Write README.md with setup/run/outputs/ASCII architecture
[ ] 11.2 Confirm final folder structure matches the required structure
[ ] 11.3 Mark all tasks [x] in TASKS.md if everything passed
[ ] 11.4 Print final TASKS.md contents to terminal
"""


def _ensure_exists(path: Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file to exist after save: {path}")


def _read_tasks() -> str:
    if not TASKS_PATH.exists():
        return TASKS_TEMPLATE
    return TASKS_PATH.read_text(encoding="utf-8")


def _write_tasks(text: str) -> None:
    TASKS_PATH.write_text(text, encoding="utf-8")
    _ensure_exists(TASKS_PATH)


def refresh_completed_counter() -> None:
    text = _read_tasks()
    completed = len(
        re.findall(r"^\[x\]\s+\d+\.\d+\s", text, flags=re.MULTILINE)
    )
    text = re.sub(
        r"^COMPLETED:\s+\d+/\d+",
        f"COMPLETED: {completed}/{TOTAL_TASK_ITEMS}",
        text,
        flags=re.MULTILINE,
    )
    _write_tasks(text)


def set_current_task(task_label: str) -> None:
    text = _read_tasks()
    text = re.sub(
        r"^CURRENT TASK:.*$",
        f"CURRENT TASK: {task_label}",
        text,
        flags=re.MULTILINE,
    )
    _write_tasks(text)
    refresh_completed_counter()


def mark_task_block(task_number: int, status: str) -> None:
    if status not in {"x", "FAILED"}:
        raise ValueError("status must be 'x' or 'FAILED'")
    marker = f"[{status}]"
    text = _read_tasks()
    pattern = re.compile(rf"^\[(?: |x|FAILED)\](\s+{task_number}\.\d+\s.*)$", re.MULTILINE)
    text = pattern.sub(lambda m: f"{marker}{m.group(1)}", text)
    _write_tasks(text)
    refresh_completed_counter()


def initialize_tasks_md_if_missing() -> None:
    if not TASKS_PATH.exists():
        _write_tasks(TASKS_TEMPLATE)
    refresh_completed_counter()


def setup_project_structure() -> dict[str, Any]:
    os.chdir(PROJECT_ROOT)
    initialize_tasks_md_if_missing()

    dirs = [
        PROJECT_ROOT / "data" / "raw_audio",
        PROJECT_ROOT / "data" / "processed_audio",
        PROJECT_ROOT / "data" / "financial",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "outputs" / "models",
        PROJECT_ROOT / "outputs" / "plots",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    REQUIREMENTS_PATH.write_text(REQUIREMENTS_TEXT, encoding="utf-8")
    _ensure_exists(REQUIREMENTS_PATH)

    init_path = PROJECT_ROOT / "src" / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")
    _ensure_exists(init_path)

    print(os.path.exists("TASKS.md"))
    return {"status": "ok"}


def _write_readme() -> None:
    readme_text = """# PROJECT 5 — Joint Audio + Market Behavior Prediction Model

This project combines earnings call audio signals and financial time-series signals to predict market reaction levels (`Negative`, `Neutral`, `Positive`).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Outputs

- `outputs/models/best_multimodal_model.pt`: best multimodal checkpoint
- `outputs/models/best_finance_only_model.pt`: best finance-only baseline checkpoint
- `outputs/results_training_history.csv`: epoch-level training/validation history for both models
- `outputs/results.csv`: evaluation metrics (precision, recall, f1, accuracy)
- `outputs/plots/*.png`: spectrograms and evaluation/comparison figures
- `data/labels.csv`: matched feature pairs with target labels

## Notes

- If real earnings-call audio files are unavailable, audio features are generated synthetically (deterministic per ticker/date).
- Replace `data/processed_audio/*_audio_features.npy` with real extracted features when audio becomes available.
- If `yfinance` is unavailable/offline, synthetic stock price series are generated so the pipeline remains runnable end-to-end.

## Architecture (ASCII)

```text
Audio Features (84) ----> [AudioBranch: GRU encoder -> 64] -----\
                                                                  +--> [AttentionFusion] --> [Head 64->32->3] --> logits
Finance Features (10) --> [FinanceBranch: LSTM encoder -> 64] ---/

AttentionFusion:
  concat(audio_emb, finance_emb) -> Linear(128->2) -> softmax
  weighted_sum = w_audio * audio_emb + w_fin * finance_emb

Baseline:
  Finance Features (10) -> [FinanceOnlyModel 10->64->32->3] -> logits
```
"""
    README_PATH.write_text(readme_text, encoding="utf-8")
    _ensure_exists(README_PATH)


def _confirm_folder_structure() -> None:
    required_paths = [
        PROJECT_ROOT / "TASKS.md",
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data" / "raw_audio",
        PROJECT_ROOT / "data" / "processed_audio",
        PROJECT_ROOT / "data" / "financial",
        PROJECT_ROOT / "src" / "__init__.py",
        PROJECT_ROOT / "src" / "dataset_builder.py",
        PROJECT_ROOT / "src" / "audio_processor.py",
        PROJECT_ROOT / "src" / "financial_processor.py",
        PROJECT_ROOT / "src" / "label_generator.py",
        PROJECT_ROOT / "src" / "model.py",
        PROJECT_ROOT / "src" / "train.py",
        PROJECT_ROOT / "src" / "evaluate.py",
        PROJECT_ROOT / "outputs" / "models",
        PROJECT_ROOT / "outputs" / "plots",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def _print_tasks_md() -> None:
    text = _read_tasks()
    print(text)


def _run_task8_training() -> dict[str, Any]:
    if TRAIN_HISTORY_PATH.exists():
        TRAIN_HISTORY_PATH.unlink()
    result_mm = train_model("multimodal")
    result_fin = train_model("finance_only")
    return {"multimodal": result_mm, "finance_only": result_fin}


def _step(task_num: int, task_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    set_current_task(f"TASK {task_num}: {task_name}")
    try:
        result = fn(*args, **kwargs)
        mark_task_block(task_num, "x")
        return result
    except Exception as exc:
        print(f"FAILED AT: {task_name} — {exc}")
        mark_task_block(task_num, "FAILED")
        raise


def run_pipeline() -> int:
    os.chdir(PROJECT_ROOT)

    try:
        _step(1, "Setup Project Structure & TASKS.md", setup_project_structure)
        _step(2, "Download Earnings Call Audio Dataset", download_earnings_audio)
        _step(3, "Download Matching Stock Price Data", download_stock_data)
        _step(4, "Audio Feature Extraction", extract_all_audio_features)
        _step(5, "Financial Feature Engineering", compute_all_financial_features)
        _step(6, "Label Generation (Target Variable)", generate_labels)
        _step(7, "Build the Multimodal Fusion Model", verify_model_forward_pass)
        _step(8, "Training Pipeline", _run_task8_training)
        _step(9, "Evaluation & Comparison", evaluate_and_compare)

        # Task 10 verifies that main.py orchestrated the full run cleanly.
        set_current_task("TASK 10: Wire Everything in main.py")
        print("PROJECT 5 COMPLETE. Check outputs/ folder.")
        mark_task_block(10, "x")

        def _task11_finalize() -> dict[str, Any]:
            _write_readme()
            _confirm_folder_structure()
            mark_task_block(11, "x")
            _print_tasks_md()
            return {"status": "ok"}

        _step(11, "Final Cleanup & README", _task11_finalize)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())

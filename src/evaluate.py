import os
import random
from pathlib import Path
from typing import Any

random.seed(42)

import numpy as np

np.random.seed(42)

import torch
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.model import FinanceOnlyModel, MarketReactionModel
from src.train import MarketDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_PATH = OUTPUTS_DIR / "results.csv"
TRAIN_HISTORY_PATH = OUTPUTS_DIR / "results_training_history.csv"
SPLITS_PATH = OUTPUTS_DIR / "split_indices.npz"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_exists(path: Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file to exist after save: {path}")


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_test_loader() -> DataLoader:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Missing split indices: {SPLITS_PATH}")
    dataset = MarketDataset(PROJECT_ROOT / "data" / "labels.csv")
    split_data = np.load(SPLITS_PATH)
    test_idx = split_data["test_idx"].astype(int)
    test_subset = Subset(dataset, test_idx.tolist())
    return DataLoader(test_subset, batch_size=16, shuffle=False)


def _load_models() -> tuple[MarketReactionModel, FinanceOnlyModel]:
    mm_path = MODELS_DIR / "best_multimodal_model.pt"
    fin_path = MODELS_DIR / "best_finance_only_model.pt"
    _ensure_exists(mm_path)
    _ensure_exists(fin_path)

    multimodal = MarketReactionModel()
    mm_ckpt = _safe_torch_load(mm_path)
    multimodal.load_state_dict(mm_ckpt["state_dict"])
    multimodal.eval()

    finance_only = FinanceOnlyModel()
    fin_ckpt = _safe_torch_load(fin_path)
    finance_only.load_state_dict(fin_ckpt["state_dict"])
    finance_only.eval()
    return multimodal, finance_only


def _run_inference(
    multimodal: MarketReactionModel,
    finance_only: FinanceOnlyModel,
    test_loader: DataLoader,
) -> dict[str, Any]:
    y_true: list[int] = []
    mm_preds: list[int] = []
    fin_preds: list[int] = []
    attn_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for audio_t, finance_t, labels in test_loader:
            mm_logits, mm_attn = multimodal(audio_t, finance_t)
            # audio_t intentionally ignored for finance-only baseline.
            fin_logits = finance_only(finance_t)

            mm_batch_preds = torch.argmax(mm_logits, dim=1).cpu().numpy()
            fin_batch_preds = torch.argmax(fin_logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            attn_np = mm_attn.cpu().numpy()

            y_true.extend(labels_np.tolist())
            mm_preds.extend(mm_batch_preds.tolist())
            fin_preds.extend(fin_batch_preds.tolist())
            for pred_cls, weights in zip(mm_batch_preds.tolist(), attn_np.tolist()):
                attn_rows.append(
                    {
                        "pred_class": int(pred_cls),
                        "audio_weight": float(weights[0]),
                        "finance_weight": float(weights[1]),
                    }
                )

    return {
        "y_true": np.array(y_true, dtype=int),
        "mm_preds": np.array(mm_preds, dtype=int),
        "fin_preds": np.array(fin_preds, dtype=int),
        "attn_df": pd.DataFrame(attn_rows),
    }


def _metrics_row(model_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    print(
        f"{model_type}: Precision={precision:.4f} Recall={recall:.4f} "
        f"F1={f1:.4f} Accuracy={acc:.4f}"
    )
    return {
        "model_type": model_type,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = [0, 1, 2]
    class_names = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _ensure_exists(out_path)


def _save_attention_plot(attn_df: pd.DataFrame, out_path: Path) -> None:
    class_names_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if attn_df.empty:
        grouped = pd.DataFrame(
            {
                "audio_weight": [0.0, 0.0, 0.0],
                "finance_weight": [0.0, 0.0, 0.0],
            },
            index=[0, 1, 2],
        )
    else:
        grouped = (
            attn_df.groupby("pred_class")[["audio_weight", "finance_weight"]]
            .mean()
            .reindex([0, 1, 2], fill_value=0.0)
        )

    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, grouped["audio_weight"].values, width, label="Audio")
    ax.bar(x + width / 2, grouped["finance_weight"].values, width, label="Finance")
    ax.set_xticks(x)
    ax.set_xticklabels([class_names_map[i] for i in [0, 1, 2]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Attention Weight")
    ax.set_title("Average Attention Weights by Predicted Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _ensure_exists(out_path)


def _save_model_comparison_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["precision", "recall", "f1", "accuracy"]
    mm = results_df.set_index("model_type").reindex(["multimodal", "finance_only"])
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, mm.loc["multimodal", metrics].values, width, label="Multimodal")
    ax.bar(x + width / 2, mm.loc["finance_only", metrics].values, width, label="Finance-Only")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _ensure_exists(out_path)


def _save_training_curves(out_path: Path) -> None:
    if not TRAIN_HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing training history CSV: {TRAIN_HISTORY_PATH}")
    hist_df = pd.read_csv(TRAIN_HISTORY_PATH)
    if hist_df.empty:
        raise ValueError("Training history CSV is empty.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type in ["multimodal", "finance_only"]:
        sub = hist_df[hist_df["model_type"] == model_type].copy()
        if sub.empty:
            continue
        ax.plot(sub["epoch"], sub["train_loss"], label=f"{model_type} train_loss")
        ax.plot(sub["epoch"], sub["val_loss"], label=f"{model_type} val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _ensure_exists(out_path)


def evaluate_and_compare() -> dict[str, Any]:
    _ensure_dir(OUTPUTS_DIR)
    _ensure_dir(PLOTS_DIR)
    test_loader = _load_test_loader()
    multimodal, finance_only = _load_models()

    inference = _run_inference(multimodal, finance_only, test_loader)
    y_true = inference["y_true"]
    mm_preds = inference["mm_preds"]
    fin_preds = inference["fin_preds"]
    attn_df = inference["attn_df"]

    results_rows = [
        _metrics_row("multimodal", y_true, mm_preds),
        _metrics_row("finance_only", y_true, fin_preds),
    ]
    results_df = pd.DataFrame(
        results_rows, columns=["model_type", "precision", "recall", "f1", "accuracy"]
    )
    results_df.to_csv(RESULTS_PATH, index=False)
    _ensure_exists(RESULTS_PATH)

    _save_confusion_matrix(
        y_true,
        mm_preds,
        PLOTS_DIR / "confusion_matrix_multimodal.png",
        "Confusion Matrix - Multimodal",
    )
    _save_confusion_matrix(
        y_true,
        fin_preds,
        PLOTS_DIR / "confusion_matrix_finance_only.png",
        "Confusion Matrix - Finance Only",
    )
    _save_attention_plot(attn_df, PLOTS_DIR / "attention_weights.png")
    _save_model_comparison_plot(results_df, PLOTS_DIR / "model_comparison_bar.png")
    _save_training_curves(PLOTS_DIR / "training_curves.png")

    required_plots = [
        PLOTS_DIR / "confusion_matrix_multimodal.png",
        PLOTS_DIR / "confusion_matrix_finance_only.png",
        PLOTS_DIR / "attention_weights.png",
        PLOTS_DIR / "model_comparison_bar.png",
        PLOTS_DIR / "training_curves.png",
    ]
    for plot_path in required_plots:
        _ensure_exists(plot_path)

    print("Evaluation complete. Results saved.")
    return {
        "results_path": str(RESULTS_PATH),
        "plots": [str(p) for p in required_plots],
    }


if __name__ == "__main__":
    evaluate_and_compare()

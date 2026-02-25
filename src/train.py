import os
import random
from pathlib import Path
from typing import Any

random.seed(42)

import numpy as np

np.random.seed(42)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import FinanceOnlyModel, MarketReactionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_ROOT / "data" / "labels.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SPLITS_PATH = OUTPUTS_DIR / "split_indices.npz"
TRAIN_HISTORY_PATH = OUTPUTS_DIR / "results_training_history.csv"


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


class MarketDataset(Dataset):
    def __init__(self, labels_csv_path: str | Path = LABELS_PATH) -> None:
        labels_csv_path = Path(labels_csv_path)
        if not labels_csv_path.exists():
            raise FileNotFoundError(f"labels.csv not found: {labels_csv_path}")
        df = pd.read_csv(labels_csv_path)
        if df.empty:
            raise ValueError("labels.csv is empty.")

        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        for _, row in df.iterrows():
            audio_path = Path(str(row["audio_feature_file"]))
            fin_path = Path(str(row["fin_feature_file"]))
            if not audio_path.is_absolute():
                audio_path = PROJECT_ROOT / audio_path
            if not fin_path.is_absolute():
                fin_path = PROJECT_ROOT / fin_path
            if not audio_path.exists() or not fin_path.exists():
                continue
            try:
                audio_features = np.load(audio_path).astype(np.float32)
                fin_features = np.load(fin_path).astype(np.float32)
            except Exception:
                continue
            if audio_features.shape != (84,) or fin_features.shape != (10,):
                continue
            label = int(row["label"])
            self.samples.append((audio_features, fin_features, label))

        if not self.samples:
            raise RuntimeError("No valid samples found in MarketDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_features, fin_features, label = self.samples[idx]
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
        finance_tensor = torch.tensor(fin_features, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return audio_tensor, finance_tensor, label_tensor

    def labels_array(self) -> np.ndarray:
        return np.array([label for _, _, label in self.samples], dtype=int)


def _can_stratify(labels: np.ndarray) -> bool:
    if labels.size < 6:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) >= 2 and np.all(counts >= 2)


def _create_and_save_splits(dataset: MarketDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(dataset))
    labels = dataset.labels_array()

    stratify_labels = labels if _can_stratify(labels) else None
    if stratify_labels is None:
        print("Using unstratified split (insufficient class counts).")

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=stratify_labels,
    )

    temp_labels = labels[temp_idx]
    stratify_temp = temp_labels if _can_stratify(temp_labels) else None
    if stratify_temp is None:
        print("Using unstratified val/test split (insufficient class counts in temp split).")

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=stratify_temp,
    )

    np.savez(SPLITS_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    _ensure_exists(SPLITS_PATH)
    return train_idx, val_idx, test_idx


def get_or_create_splits(dataset: MarketDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if SPLITS_PATH.exists():
        data = np.load(SPLITS_PATH)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]
        max_idx = len(dataset) - 1
        if all(len(arr) > 0 for arr in (train_idx, val_idx, test_idx)) and all(
            np.all((arr >= 0) & (arr <= max_idx)) for arr in (train_idx, val_idx, test_idx)
        ):
            return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)
        print("Existing split indices invalid for current dataset; regenerating.")
    return _create_and_save_splits(dataset)


def _make_dataloaders(
    dataset: MarketDataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())

    labels = dataset.labels_array()
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=3)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # prevent /0
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_subset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    return train_loader, val_loader, test_loader


def _epoch_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    model_type: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for audio_t, finance_t, labels in loader:
        audio_t = audio_t.to(device)
        finance_t = finance_t.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        if model_type == "multimodal":
            logits, _ = model(audio_t, finance_t)
        else:
            # audio_t intentionally ignored for finance-only baseline.
            logits = model(finance_t)

        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def _checkpoint_path_for_model(model_type: str) -> Path:
    if model_type == "multimodal":
        return MODELS_DIR / "best_multimodal_model.pt"
    if model_type == "finance_only":
        return MODELS_DIR / "best_finance_only_model.pt"
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_model(model_type: str = "multimodal") -> dict[str, Any]:
    if model_type not in {"multimodal", "finance_only"}:
        raise ValueError("model_type must be 'multimodal' or 'finance_only'")

    _ensure_dir(OUTPUTS_DIR)
    _ensure_dir(MODELS_DIR)

    dataset = MarketDataset(LABELS_PATH)
    train_idx, val_idx, test_idx = get_or_create_splits(dataset)
    train_loader, val_loader, _ = _make_dataloaders(dataset, train_idx, val_idx, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "multimodal":
        model: torch.nn.Module = MarketReactionModel()
    else:
        model = FinanceOnlyModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    checkpoint_path = _checkpoint_path_for_model(model_type)
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, 51):
        train_loss, train_acc = _epoch_pass(
            model, train_loader, criterion, optimizer, device, model_type
        )
        with torch.no_grad():
            val_loss, val_acc = _epoch_pass(
                model, val_loader, criterion, None, device, model_type
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "model_type": model_type,
            }
        )

        print(
            f"Epoch {epoch}/50 | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": model_type,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "audio_dim": 84,
                    "fin_dim": 10,
                },
                checkpoint_path,
            )
            _ensure_exists(checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for {model_type} at epoch {epoch}.")
                break

    history_df = pd.DataFrame(
        history_rows,
        columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "model_type"],
    )
    if model_type == "multimodal":
        write_mode = "w"
        header = True
    else:
        write_mode = "a"
        header = not TRAIN_HISTORY_PATH.exists()
    history_df.to_csv(TRAIN_HISTORY_PATH, mode=write_mode, index=False, header=header)
    _ensure_exists(TRAIN_HISTORY_PATH)

    _ensure_exists(checkpoint_path)
    print(f"{model_type} best val accuracy: {best_val_acc * 100:.2f}%")

    if model_type == "finance_only":
        mm_path = MODELS_DIR / "best_multimodal_model.pt"
        fin_path = MODELS_DIR / "best_finance_only_model.pt"
        _ensure_exists(mm_path)
        _ensure_exists(fin_path)
        mm_ckpt = _safe_torch_load(mm_path)
        fin_ckpt = _safe_torch_load(fin_path)
        print(
            "Final val accuracies | multimodal: "
            f"{float(mm_ckpt.get('best_val_acc', 0.0)) * 100:.2f}% | finance_only: "
            f"{float(fin_ckpt.get('best_val_acc', 0.0)) * 100:.2f}%"
        )

    return {
        "model_type": model_type,
        "checkpoint_path": str(checkpoint_path),
        "best_val_acc": best_val_acc,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }


if __name__ == "__main__":
    train_model("multimodal")
    train_model("finance_only")

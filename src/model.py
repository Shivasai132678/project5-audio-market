import random

random.seed(42)

import numpy as np

np.random.seed(42)

import torch
import torch.nn as nn

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AudioBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Treat the 84-dim audio summary vector as a short sequence for GRU encoding.
        self.input_proj = nn.Linear(1, 16)
        self.gru = nn.GRU(
            input_size=16,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)  # (batch, 84, 1)
        seq = torch.relu(self.input_proj(seq))  # (batch, 84, 16)
        _, h_n = self.gru(seq)
        return h_n[-1]  # (batch, 64)


class FinanceBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Treat the 10-dim financial feature vector as a feature sequence for LSTM encoding.
        self.input_proj = nn.Linear(1, 16)
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)  # (batch, 10, 1)
        seq = torch.relu(self.input_proj(seq))  # (batch, 10, 16)
        _, (h_n, _) = self.lstm(seq)
        return h_n[-1]  # (batch, 64)


class AttentionFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(128, 2)

    def forward(
        self, audio_emb: torch.Tensor, finance_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([audio_emb, finance_emb], dim=1)
        attn_logits = self.attn(combined)
        attn_weights = torch.softmax(attn_logits, dim=1)
        fused = (
            attn_weights[:, 0:1] * audio_emb
            + attn_weights[:, 1:2] * finance_emb
        )
        return fused, attn_weights


class MarketReactionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_branch = AudioBranch()
        self.finance_branch = FinanceBranch()
        self.fusion = AttentionFusion()
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

    def forward(
        self, audio_x: torch.Tensor, finance_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_emb = self.audio_branch(audio_x)
        finance_emb = self.finance_branch(finance_x)
        fused, attn_weights = self.fusion(audio_emb, finance_emb)
        logits = self.head(fused)
        return logits, attn_weights


class FinanceOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, finance_x: torch.Tensor) -> torch.Tensor:
        return self.net(finance_x)


def verify_model_forward_pass() -> None:
    audio = torch.randn(4, 84)
    finance = torch.randn(4, 10)
    model = MarketReactionModel()
    logits, attn = model(audio, finance)
    assert logits.shape == (4, 3), f"Unexpected logits shape: {logits.shape}"
    assert attn.shape == (4, 2), f"Unexpected attention shape: {attn.shape}"
    print("Model forward pass OK")


__all__ = [
    "AudioBranch",
    "FinanceBranch",
    "AttentionFusion",
    "MarketReactionModel",
    "FinanceOnlyModel",
    "verify_model_forward_pass",
]


if __name__ == "__main__":
    verify_model_forward_pass()

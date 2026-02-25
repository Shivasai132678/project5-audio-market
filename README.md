# PROJECT 5 — Joint Audio + Market Behavior Prediction Model

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
Audio Features (84) ----> [AudioBranch: GRU encoder -> 64] -----                                                                  +--> [AttentionFusion] --> [Head 64->32->3] --> logits
Finance Features (10) --> [FinanceBranch: LSTM encoder -> 64] ---/

AttentionFusion:
  concat(audio_emb, finance_emb) -> Linear(128->2) -> softmax
  weighted_sum = w_audio * audio_emb + w_fin * finance_emb

Baseline:
  Finance Features (10) -> [FinanceOnlyModel 10->64->32->3] -> logits
```

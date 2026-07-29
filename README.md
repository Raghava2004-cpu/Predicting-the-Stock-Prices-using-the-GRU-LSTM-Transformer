# Predicting-the-Stock-Prices-using-the - GRU_LSTM_Transformer

## Benchmark results — AAPL (2015-01-01 to 2025-01-01)

| Model | RMSE | MAE | MAPE (%) | Directional Accuracy (%) |
|---|---|---|---|---|
| Naive persistence | 2.67 | 1.96 | 1.00 | 0.27 |
| Linear Regression | 2.84 | 2.12 | 1.08 | **52.66** |
| LSTM | 7.03 | 5.00 | 2.35 | 44.95 |
| GRU | 7.04 | 5.27 | 2.50 | 46.54 |
| Transformer | 9.49 | 7.41 | 3.51 | 43.35 |


<img width="2100" height="900" alt="predictions_vs_actual" src="https://github.com/user-attachments/assets/4e768ae0-e7c0-4930-9e2a-0272134bb72f" />
<img width="1500" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/4dc8a4ba-2084-4382-b37f-d18e5c7acf30" />


### Reading these results honestly

The baselines beat every deep model here, on every error metric. Naive persistence
(predict "no change") gets the lowest RMSE/MAE/MAPE by construction — daily closing
prices are close to a random walk, so guessing "tomorrow = today" is a genuinely hard
benchmark to beat. Linear regression is the strongest model overall and the only one
above 50% directional accuracy.

The GRU, LSTM, and Transformer all underperform both baselines, and score *below*
50% directional accuracy — worse than chance at calling the next day's direction. The
predictions plot shows why: all three visibly lag the actual price rather than
tracking it, with LSTM lagging worst. This is a well-documented failure mode for
recurrent/attention models on raw daily OHLCV — with only ~1,180 training rows and a
noisy, near-random-walk target, these architectures have more capacity than the
signal in the data supports, and they partially collapse toward smoothed
autocorrelation with the recent price rather than learning real predictive structure.

**Takeaway:** on this ticker/date range/feature set, model complexity did not help
prediction accuracy — a useful, common, and legitimate finding. It also points at the
main lever for improvement: more data, more regularization, or better features/labels
(e.g. predicting returns instead of price, or a longer prediction horizon) rather than
a bigger model.

### Next steps to close the gap
- Predict **returns** rather than raw price (removes most of the "just copy the last
  value" trivial signal that inflates naive/linear baselines).
- Regularize the deep models more (weight decay, dropout tuning) and/or shrink
  hidden size — they may simply be overparameterized for ~1,180 training rows.
- Multiple seeds per model, reported as mean ± std, since a single run this close to
  the baselines isn't enough to rule out noise.
- Walk-forward validation across several time windows and tickers instead of one
  static split.



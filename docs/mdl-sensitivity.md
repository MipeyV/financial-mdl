# Synthetic MDL sensitivity study

## Question and declared plan

Does the first MDL selector retain the same variables when target coding precision and
training sample size change? In particular, when does a real weak signal repay the fixed
64-bit coefficient cost? This experiment varies target precision, not parameter precision:
all parameters still use binary64 and Ridge alpha stays at 1.

The complete grid is committed in [the experiment plan](../experiments/mdl-sensitivity-v1.json)
before the first full run:

- Training sizes: 150, 600, 2,400 and 9,600 observations.
- Target-grid precisions: 0.0001, 0.01 and 0.1, in synthetic target units.
- Seeds: 0, 1, 2, 3 and 4.
- Independent held-out block: 300 observations per scenario and seed.
- Scenarios: strong signals with an exact duplicate and three noise variables; pure noise;
  and a weak signal with coefficient 0.05. Noise standard deviation is 0.4 in all cases.
- Evaluation: raw continuous-target MAE/RMSE, selected features, training code decomposition
  and paired RMSE ratio to full Ridge. The existing exploratory +2% tolerance is descriptive.

This yields 180 exhaustive subset searches and 360 full/MDL evaluations. Every search tests
all 64 subsets of the same six-feature universe, including the constant model. No cell is
selected because it has the best held-out error. All declared configurations are reported.

## Pairing and separation

For each scenario/seed, generate 9,900 observations once with `controlled_regression`.
The first 9,600 form the training pool; smaller sizes take its prefixes. The final 300 rows
are the common held-out block for every size and precision. Rows between a small prefix and
the held-out block are unused for that size. The data are independent regression observations,
so these are sample-size controls rather than a financial temporal backtest.

Features and noise are shared across scenarios for each seed. Precision comparisons have
identical training and held-out data. Per-block fingerprints are exported. Both scaler and
selector see training rows only, and every full/MDL pair has the same alpha and evaluation
rows. Changing held-out values cannot change the selected subset or its training code.

The earlier 600/300 controls used a different total generator size. Because the noise draws
follow all feature draws, this larger fixed pool produces different target noise and held-out
observations. Their absolute scores are not a direct before/after comparison with the earlier
study; comparisons within this declared grid remain paired.

## Reproduction

```powershell
uv run financial-mdl mdl-sensitivity --config experiments/mdl-sensitivity-v1.json --output artifacts/mdl-sensitivity-v1
```

Use a fresh output directory. Before generating data or fitting, the command writes
`experiment.json` with the complete plan, fingerprint, source hashes, Git/runtime provenance
and lock hash. It exports every subset search as `*-selection.json`, every paired result in
`results.csv`, aggregated cells in `summary.csv`, feature frequencies, and Plotly HTML/JSON
figures. Standard deviations describe variation across seeds, not confidence intervals.
These standalone synthetic outputs do not create financial benchmark or MLflow runs.

## Observed results

The complete declared grid was run from clean commit
`6cb8e7949f33c52749d1b034234965f017d9b25b` with experiment fingerprint
`17e1184b376e343cefa9b46c34a7ebd9059f480885bae3736644aa9847efcf76`.
All 180 searches and 360 full/MDL evaluations completed. The full aggregated table is
[versioned here](results/mdl-sensitivity-v1-summary.csv); detailed artifacts are local under
`artifacts/mdl-sensitivity-v1`.

Strong-signal scenarios always select `signal_a` and `signal_b`: two features rather than six,
for all 60 scenario/size/precision/seed combinations. The exact duplicate and all three noise
columns are omitted. Noise-only scenarios always select zero features, also across all 60
combinations. No noise column is selected anywhere in the grid.

The weak-signal selection exposes both sample-size and precision sensitivity:

| Training observations | Precision 0.0001 | Precision 0.01 | Precision 0.1 |
| ---: | ---: | ---: | ---: |
| 150 | 0/5 seeds retain signal_a | 0/5 | 0/5 |
| 600 | 0/5 | 0/5 | 0/5 |
| 2,400 | 0/5 | 0/5 | 0/5 |
| 9,600 | 5/5 | 5/5 | 2/5 |

Every weak-signal model is either constant or retains only `signal_a`. For 9,600 rows,
adding this coefficient costs **67 extra representation bits**: 64 parameter bits and
3 additional subset-identification bits. At precision 0.0001, the saved target bits are
99 to 137 across seeds, so every seed repays that cost. At precision 0.1, target savings
are 60, 47, 121, 73 and 59 bits for seeds 0 through 4. Only seeds 2 and 3 exceed 67 bits.
This explains the selection directly from the training code, independently of held-out error.

Selected-model RMSE means across five seeds at the largest training size are:

| Scenario | Full Ridge RMSE | MDL RMSE, precision 0.0001 | MDL RMSE, precision 0.01 | MDL RMSE, precision 0.1 |
| --- | ---: | ---: | ---: | ---: |
| Strong signals | 0.394573 | 0.394783 | 0.394783 | 0.394783 |
| Noise only | 0.394584 | 0.394661 | 0.394661 | 0.394661 |
| Weak signal | 0.394584 | 0.394721 | 0.394721 | 0.395554 |

All **180 paired MDL evaluations** remain within the predeclared +2% RMSE tolerance.
The worst individual ratio is 1.010757, about **1.08% worse** than full Ridge. MDL does
not consistently improve prediction error: at 9,600 rows, all three scenario means are
slightly worse than full Ridge. At smaller sizes, simplicity can reduce error, but these
five-seed observations do not establish a general predictive advantage.

The supported finding is that this explicit code reliably removes noise/redundancy in these
controls, while retaining a weak real predictor only when its accumulated coding gain
repays the representation cost. The coarse grid changes that decision in three of five
large-sample weak-signal cases. The experiment leaves the original SPY coding choice intact.

## Interpretation boundaries

Compare description lengths **within** a fixed precision, sample size and dataset. A coarser
grid encodes a different target object, and a larger sample encodes more observations: raw
total lengths across those cells do not measure a compression improvement on the same object.
Prediction metrics always retain the original continuous targets, so paired error comparisons
can examine the effect of selection separately from rounding.

Five seeds provide exploratory controls, not a significance claim. A weak signal can genuinely
exist while failing to repay this code's model cost. Precision may have little influence in
some regimes; that is a result to report. This study does not vary the parameter encoding,
prove optimal coding, or choose a new precision for the SPY experiment. The market protocol,
target, observation dates, temporal boundaries and reserved final test remain unchanged.

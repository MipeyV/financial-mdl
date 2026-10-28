# Initial MDL results

These are exploratory results for `ridge_binary64_geometric_v1`, with target precision
0.0001 and Ridge alpha 1. The code and controls are specified in [mdl-coding.md](mdl-coding.md).
The subset is chosen using training data only; the +2% RMSE tolerance was specified before
the MDL validation comparison. No alternative coding configurations were searched on validation.

## Frozen SPY experiment

The original SPY snapshot contains 2,766 daily sessions from 2015–2025. Dataset fingerprint:
`b5d1607f6df78ee0fcba7d1fef7d09081e703e5671961aa81fbbacbbcd398409`.
The saved `spy-volatility` version 1 protocol predicts annualized five-session RMS volatility.
The training partition has 1,738 eligible origins (2015–2021); validation has 496 (2022–2023).
The 2024–2025 final test period has not been evaluated.

| Candidate | Features used | Validation MAE | Validation RMSE |
| --- | ---: | ---: | ---: |
| Training mean | 0 | 0.068316 | 0.093630 |
| Persistence 5 sessions | 1 | 0.058852 | 0.079566 |
| Persistence 20 sessions | 1 | 0.051482 | 0.069084 |
| Full Ridge | 5 | 0.049341 | 0.066996 |
| Ridge + MDL | 3 | 0.048443 | 0.065489 |

Persistence consumes its one named volatility feature; all candidates retain the same
complete-case eligibility universe of five features. The MDL subset is:

- `log_return_5`
- `realized_volatility_5`
- `realized_volatility_20`

`log_return_1` and `volume_change_1` are omitted under this coding specification. This is
evidence about this dataset/code, not proof that the variables contain no useful information.

## What becomes shorter?

The following costs encode **training** targets at the fixed precision, conditional on X.
They are not validation losses and do not include storing the financial snapshot or X itself.

| Reference | Subset bits | Model bits | Target bits | Total bits |
| --- | ---: | ---: | ---: | ---: |
| Constant | 3 | 128 | 20,737 | 20,868 |
| Full Ridge | 3 | 448 | 19,706 | 20,157 |
| Selected Ridge | 7 | 320 | 19,718 | 20,045 |

The selected model gives up 12 target-coding bits but saves 128 parameter bits, while costing
4 additional subset-identification bits: the net saving is **112 bits (0.56% of the total)**.
The representation alone (subset plus model) falls from 451 to 327 bits, **27.49% shorter**.
The feature count falls from five to three, **40% fewer features**. These are distinct measures.

Validation RMSE improves by approximately **2.25%** relative to full Ridge. This satisfies the
exploratory +2% degradation tolerance on this validation period. The result does not establish
statistical superiority: the labels overlap, only one market period is evaluated, and the
coding specification is a particular, non-optimal choice.

## Controlled independent regression

Five fixed seeds (0–4), 600 training observations and 300 held-out observations per scenario.
All six input features, noise and target definitions are listed in the coding specification.
Values below average five seeds; standard deviations describe variation between seeds, not
confidence intervals. These scores are not comparable to the financial task's RMSE units.

| Scenario | Method | Mean features | Held-out RMSE mean | RMSE std across seeds |
| --- | --- | ---: | ---: | ---: |
| Strong signals + redundancy | Full Ridge | 6 | 0.407826 | 0.008596 |
| Strong signals + redundancy | Ridge + MDL | 2 | 0.406789 | 0.008896 |
| Noise only | Full Ridge | 6 | 0.407890 | 0.008569 |
| Noise only | Ridge + MDL | 0 | 0.406934 | 0.009380 |
| Weak signal | Full Ridge | 6 | 0.407888 | 0.008570 |
| Weak signal | Ridge + MDL | 0 | 0.407029 | 0.010584 |

In the strong-signal case, every seed retains one representative of signal_a and its exact
copy, plus signal_b. The redundant copy and the three noise columns are excluded. Ties favor
input order, so signal_a is chosen rather than its identical copy; their identities are not
statistically distinguishable.

Every noise-only seed selects the constant. Every weak-signal seed also selects the constant,
even though signal_a genuinely enters the generator. Its training coding gain does not repay
the binary64 model cost. Held-out RMSE nevertheless stays within the 2% tolerance in all 15
paired experiments; the largest observed ratio to full Ridge is about 1.00698 in weak signal.
This illustrates both useful simplification and sensitivity to the chosen code and sample size.

## Reproduction and interpretation

Use the commands in [mdl-coding.md](mdl-coding.md) with fresh output directories. Market
artifacts and local MLflow retain per-run Git/source/runtime provenance, the exact dataset
and evaluation fingerprints, predictions, complete candidate tables and parameter bytes.
The synthetic experiment records its own immutable task configuration and dataset hashes.

The supported conclusion is narrow: **this explicit code selected a smaller Ridge model,
with slightly better held-out performance in the tested controls and the SPY validation
period**. A different code, richer feature universe, another market regime or final test may
produce a different outcome. The current evidence does not establish a general MDL advantage.

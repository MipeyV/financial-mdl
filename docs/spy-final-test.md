# Frozen SPY final evaluation

## Plan fixed before test scores

The [candidate plan](../experiments/spy-final-test-v1.json) is committed before requesting
the reserved 2024–2025 test scores. It binds the existing daily SPY snapshot and immutable
`spy-volatility` v1 benchmark; it does not change the protocol or feature universe.

- Candidates: full Ridge, Ridge with the original MDL selector, and 20-session persistence.
- Ridge alpha 1, intercept enabled, training-only standardization, seed 42, one-step strategy.
- MDL code `ridge_binary64_geometric_v1`, target precision 0.0001, original exhaustive search.
- Target: annualized RMS volatility over the next five observed sessions, without demeaning.
- All fitted components use the original 2015–2021 training partition only. Validation is
  not added to training. The same deterministic training search must reproduce the original
  three MDL features and 20,045 selected training bits versus 20,157 full training bits.
- Test labels must end by 2025-12-31. Expected common test origins: 497, from 2024-01-02
  through 2025-12-23. Eligibility remains complete cases across all five protocol features.
- Forecasts are issued after each session close. Features can use prices already observed
  within the test period; fitted parameters never use test targets.
- Primary metric RMSE, secondary MAE. Report all three candidates on identical test rows.
  Check the existing exploratory `RMSE_mdl / RMSE_full <= 1.02` rule. It is a descriptive
  performance tolerance, not a statistically established non-inferiority test.
- No model, subset, precision or hyperparameter will be adjusted from the test scores.
  Validation and test results remain separate comparisons.

No new scientific implementation is required: the existing explicit `--phase test` path
performs the evaluation and records provenance. Runs and their comparison use local MLflow
with the `test` phase; candidate outputs use fresh directories.

## Reproduction

Use the original frozen dataset and choose new output directories for an independently
audited replay. Replaying a frozen evaluation does not create another untouched test set.

```powershell
uv sync --locked --dev --extra tracking
uv run --no-sync financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --selector identity --strategy one-step --seed 42 --phase test --output artifacts/spy-final-test-v1/full --tracking
uv run --no-sync financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --selector mdl --mdl-target-precision 0.0001 --strategy one-step --seed 42 --phase test --output artifacts/spy-final-test-v1/mdl --tracking
uv run --no-sync financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator persistence-20 --selector identity --strategy one-step --seed 42 --phase test --output artifacts/spy-final-test-v1/persistence-20 --tracking
uv run --no-sync financial-mdl compare artifacts/spy-final-test-v1/full artifacts/spy-final-test-v1/mdl artifacts/spy-final-test-v1/persistence-20 --output artifacts/spy-final-test-v1/comparison --tracking
```

## Verification and interpretation

Verify common protocol, dataset and test evaluation fingerprints; identical observations
and dates; unchanged training split, candidate parameters and MDL diagnostics relative to
validation; and matching MAE/RMSE recomputed from exported predictions. The earlier evaluation
fingerprint identifies validation rows and is expected to differ from the final test fingerprint.

Training code lengths remain training quantities; no test code search is performed. A result
within +2% supports only the declared tolerance on this period. A result outside it means the
selected representation fails that tolerance on this period, even if it remains shorter.
Both outcomes must be reported. Overlapping labels and serial dependence remain, so the
497 forecasts do not represent 497 independent observations. No significance claim is planned.

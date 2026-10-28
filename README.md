# Financial-MDL

A financial ML **research framework**, investigating whether compact representations
selected using Minimum Description Length preserve out-of-sample predictive performance.
The implementation includes reproducible forecasting contracts and a first exhaustive MDL
selector with an explicit two-part reference code. It does not measure Kolmogorov complexity.

## Setup and verification

Requires Python 3.11+ and uv; the default development interpreter is Python 3.12.

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv.lock` pins dependencies. Core tests use synthetic data and require neither a network
connection nor MLflow. Optional integrations: `uv sync --extra market --extra tracking`.

Changes use dedicated branches (`feature/...`, `fix/...`, `chore/...`, etc.) and PRs into
`main`. GitHub Actions checks lint, formatting, package builds, offline tests on Linux
(Python 3.11–3.13) and Windows (3.12), and local MLflow integration. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and local commands.

## Primary experiment: real SPY daily market data

Download once, then freeze the Parquet snapshot and its provenance sidecar. The vendor's
end date is exclusive. Every protocol binds the exact snapshot using SHA-256.

```powershell
uv sync --extra market --extra tracking
uv run financial-mdl fetch --asset SPY --start 2015-01-01 --end 2026-01-01 --dataset-version spy-daily-v1 --output data/spy.parquet
uv run financial-mdl protocol --data data/spy.parquet --output protocols/spy-local-volatility-v1.json --id spy-volatility --train-start 2015-01-01 --train-end 2021-12-31 --validation-end 2023-12-31 --test-end 2025-12-31
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-local-volatility-v1.json --estimator naive --output artifacts/spy-naive --tracking
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-local-volatility-v1.json --estimator ridge --output artifacts/spy-ridge --tracking
uv run financial-mdl compare artifacts/spy-naive artifacts/spy-ridge --output artifacts/spy-comparison --tracking
```

Both estimators use **the same saved protocol**, origin dates, complete-case feature universe,
target, splits, metric implementation and training-only standardization. The naive baseline
predicts the training target mean. `--estimator tree` uses a seeded random forest. No implicit
clipping of negative volatility forecasts is performed. Hyperparameters belong to estimator
factories, not to the benchmark definition. Python usage is in [architecture.md](docs/architecture.md).

`protocols/spy-volatility-v1.json` is a versioned reference bound to the original snapshot.
A fresh download has its own provenance and fingerprint: the commands above create a new
local protocol file. Use new snapshot, protocol and output paths when repeating a study.

### Persistence references

Two stronger references carry forward trailing realized volatility at each forecast origin:
`persistence-5` uses the last 5 observed sessions and `persistence-20` the last 20. They learn
no relationship from the training targets. Run them under exactly the same saved protocol:

```powershell
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-local-volatility-v1.json --estimator persistence-5 --output artifacts/spy-persistence-5 --tracking
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-local-volatility-v1.json --estimator persistence-20 --output artifacts/spy-persistence-20 --tracking
uv run financial-mdl compare artifacts/spy-naive artifacts/spy-persistence-5 artifacts/spy-persistence-20 artifacts/spy-ridge --output artifacts/spy-reference-comparison --tracking
```

Persistence uses an identity transform to retain volatility units; the existing learned
estimators keep training-only standardization. All protocol features still determine the
common eligible origins. The source feature is recorded in estimator parameters, and its
252-session annualization is converted explicitly if the target uses another scale.
For direct multi-horizon forecasting, every horizon receives the same current volatility
estimate. Python and Kedro node entry points share the same candidate factories; custom
Python calls should use `make_candidate` or supply `IdentityFeatures` for persistence.

The default phase is **validation**. Select a candidate using validation results, then evaluate
that frozen candidate once with `--phase test` in a new output directory. The runner fits only
on the training partition in both phases; it does not refit on validation or tune on test.
Test results cannot be compared with validation results. The framework prevents accidental
test scoring by default; it cannot prevent a researcher from repeatedly requesting test runs.

`--horizons 1 5 10` when creating a protocol and `--strategy direct` when running it evaluate
separate scalar models for each horizon. A single five-session volatility forecast has shape
`(N, 1, 1)`, while three requested horizons have shape `(N, 3, 1)`.

For an offline smoke example only:

```powershell
uv run financial-mdl demo --output artifacts/demo
```

## MDL feature selection

The scalar Ridge reference selector evaluates every feature subset, including the constant
model, using an explicit subset code, binary64 model parameters and an integer residual code.
It fits entirely on training data. The default target precision is fixed at 0.0001; the five
market features produce 32 candidates. Full Ridge and Ridge with MDL use the same saved
benchmark protocol and unrounded evaluation targets:

```powershell
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --output artifacts/spy-mdl-v1/full --tracking
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --selector mdl --output artifacts/spy-mdl-v1/mdl --tracking
uv run financial-mdl compare artifacts/spy-mdl-v1/full artifacts/spy-mdl-v1/mdl --output artifacts/spy-mdl-v1/comparison --tracking
uv run financial-mdl mdl-synthetic --output artifacts/mdl-synthetic-v1 --seeds 0 1 2 3 4
```

These commands use the original frozen snapshot. Substitute your own saved protocol for a
fresh download. See [the coding specification](docs/mdl-coding.md) for assumptions, exact cost
terms, limitations and synthetic controls, and [the first results](docs/mdl-results.md).
Code lengths describe the **training reference code**, conditional on the shared design X;
they are separate from held-out prediction metrics. Full and selected code decompositions,
all candidate costs and Plotly figures are exported. MLflow logs these under `training/mdl/`.

## Scientific rules

- Forecast origins are **after the session close**. Features use that close and earlier data;
  target returns start in the next observed session. Horizons count observed rows, not calendar
  days. Validate missing exchange sessions upstream; this version does not supply a calendar.
- The target is `sqrt(252 * mean(next H squared log returns))`: an annualized RMS realized
  volatility, without demeaning. It is distinct from a sample standard deviation.
- Training/validation labels must end within their partition. `--gap-sessions G` additionally
  moves each pre-evaluation label cutoff back G observed sessions. Tail labels are discarded,
  never padded. Test labels must also end by the protocol's test end date.
- Every model sees identical eligible origins, determined by all protocol features before
  selection. Selectors and fitted transformations receive **training rows only**. No filling,
  shuffling, target reordering, or inferred prediction reshaping is allowed.
- Protocols are frozen dataclasses, include implementation versions, and save without
  overwriting existing files. Changing a scientific rule creates a different fingerprint.
  Comparisons also verify the actual evaluation rows, timestamps and labels.
- One asset per run is supported. Panel data, publication-time alignment, historical asset
  universes and survivorship controls require additional explicit protocols.

The yfinance adapter is a convenience source, subject to vendor access and data terms.
Unadjusted OHLC snapshots may still contain vendor restatements/corporate-action handling;
they are not certified point-in-time data. Preserve snapshots rather than silently refreshing
a benchmark. Fundamental or current-ratio data are never injected into SPY experiments.

## Artifacts and local tracking

Each run exports `protocol.json`, `run.json` (provenance and candidate settings), `metrics.json`,
`predictions.parquet` (long form with origin, label window, horizon, target and asset), and
reproducible Plotly HTML/JSON figures. Comparison exports a leaderboard only when protocol,
phase and evaluation fingerprints match. Runtime versions, source hashes, Git commit/dirty
state, seed, feature definitions, selected features and split bounds are recorded.
Fitted feature diagnostics are exported when available, with transformed-space coefficients
distinguished from tree impurity importances; neither is claimed to be causal importance.

`--tracking` uses optional open-source MLflow with local SQLite and local artifact files in
`artifacts/mlflow`. No tracking code appears inside estimators. Without the flag a no-op
tracker is used. The same flag on `compare` logs a comparison run with candidate metadata,
matched metrics and leaderboard artifacts. To inspect local runs from the repository root:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
```

Datasets, benchmark outputs, MLflow files and databases are ignored by Git. Saved protocol
definitions can be versioned, but redistribute data only when the provider's terms permit.

## Architecture and roadmap

See [architecture.md](docs/architecture.md) for contracts, module boundaries, Kedro integration,
point-in-time fundamental requirements, experiment stages and intentionally deferred work.
Core algorithms remain callable without Kedro; this empty repository did not warrant a full
Kedro migration. New estimators and strategies compose with the existing benchmark rather
than redefining its rules.

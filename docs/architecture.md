# Architecture and research protocol

## Repository assessment and first-iteration choices

The starting checkout contained only `.git`, no commits, source files, dependency manifests,
tests or runnable commands. No existing implementation was replaced. The initial plan was:
define contracts, implement schemas/protocols and causal preparation, compose scalar
estimators with forecasting strategies, implement a single benchmark execution path, add
optional tracking/artifacts, and validate with contract/leakage tests and CLI smoke runs.

The package uses composition and structural Python Protocols, not a deep class hierarchy.
NumPy/pandas carry research data; scikit-learn provides estimators and fitted scaling.
Parquet stores snapshots/predictions. Plotly produces exportable artifacts. MLflow is optional.
Kedro is deliberately not a runtime dependency in this initial empty repository.

## Module boundaries and contracts

| Module | Responsibility / contract |
| --- | --- |
| `data` | `MarketDataProvider.load() -> MarketDataset`; strict daily OHLCV and provenance; Parquet and optional yfinance adapters |
| `features` | Versioned definitions for market/statistical features; family labels reserve fundamental and algorithmic features |
| `targets` | Future volatility from complete future log-return windows, independently of models |
| `schemas` | Frozen target specification; validated forecast coordinates, label windows, and `(N,H,K)` values |
| `contracts` | `BaseModel`, `FeatureSelector`, `FeatureTransformer`, `ForecastStrategy`, `Tracker` Protocols |
| `models` | Scalar scikit-learn adapter; training-mean baseline, Ridge, seeded random forest |
| `forecasting` | `ForecastingModel(estimator=factory, strategy=...)`; single-output and direct strategies |
| `protocol` | Frozen benchmark/target/split rules, canonical JSON and SHA-256 |
| `splitting` | Deterministic chronological holdout, label-window purging, optional extra session gap |
| `benchmark` | Authorized dataset verification, eligibility, training-only fitting, aligned predictions, shared metrics, artifacts, comparison |
| `tracking` | No-op tracker and lazily imported local MLflow client |
| `reporting` | Plotly JSON and standalone HTML figures, no custom dashboard |
| `pipelines.nodes` | Thin functions for later Kedro orchestration |

`BaseModel`: `fit(X, y)`, `predict(X)`, `get_params()`. The initial adapter accepts scalar
`y.shape == (N,)` and requires scalar predictions exactly `(N,)`. Strategy code explicitly
maps these to `(N,1,1)`; it never guesses from total element count.

`FeatureSelector`: `fit(X_train, y_train)`, `transform(X)`, `get_selected_features()`,
`get_params()`. The target supplied to selectors is explicitly `(N,H,K)`.
`FeatureTransformer`: `fit(X_train)`, `transform(X)`, `get_params()`; it preserves labeled rows
and columns. Identity selection and standardization are implemented. No fit is allowed on
evaluation observations. Custom components must return new instances through factories.

`ForecastStrategy`: `fit(estimator_factory, X, Targets)`,
`predict(X, ForecastCoordinates) -> Forecast`, `get_params()`. The strategy never receives
evaluation labels. A direct strategy creates independent models per horizon and target;
it is not a genuinely joint multi-output model. A future joint strategy must adapt an estimator
with native vector outputs. Recursive forecasting needs an explicit lag-update and exogenous
availability policy; the common strategy interface accommodates it, but it is not implemented.

Target dimensions are separate concepts: prediction horizon is the session count; H is the
number of requested horizon labels; K is the number of variables. Five-session volatility
has horizon 5, H=1, K=1, and one output. Target coordinates include origin dates, target end
dates, asset identities and ordered horizon/variable labels; Targets additionally retain label
start dates. The forecast schema can represent multiple variables; the initial target builder
intentionally supports only realized volatility. Probabilistic outputs need a separate contract.

## Python example: reuse the same protocol

```python
from pathlib import Path

from financial_mdl.benchmark import BenchmarkRunner
from financial_mdl.data import ParquetMarketDataProvider
from financial_mdl.models import make_estimator
from financial_mdl.protocol import BenchmarkConfig

dataset = ParquetMarketDataProvider(Path("data/spy.parquet")).load()
protocol = BenchmarkConfig.load(Path("protocols/spy-volatility-v1.json"))
for name in ("naive", "ridge"):
    BenchmarkRunner().run(
        dataset,
        protocol,
        estimator_factory=lambda name=name: make_estimator(name, seed=42),
        output=Path("artifacts") / f"python-{name}",
        estimator_name=name,
    )
```

Swap the factory for a custom `BaseModel`, choose a strategy factory, or add a training-only
selector. The protocol is unchanged. An alternative feature universe, target, data snapshot,
observation policy or evaluation period is a **new protocol**, not a model hyperparameter.
To compare feature subsets under matched origins, authorize a common complete-case feature
universe and use selectors within that universe. Incompatible protocols are never pooled.

## Temporal correctness and practical limits

Features are trailing and available only after the origin's close. Future volatility uses
returns from t+1 through t+h, whose last timestamp is retained explicitly. Label windows must
finish on/before their partition cutoff. Additional gaps move the training/validation label
cutoff earlier by the specified number of raw observed sessions. Test labels must finish by
test end; no partial-tail labels are constructed. Internal overlap within a partition remains;
these point metrics do not claim independent observations or confidence intervals.

The complete-case origin mask uses all authorized features, preventing candidate-specific
missingness from changing scores. Trailing features may legitimately use historical market
observations from earlier partitions. No selector, scaler, estimator, or hyperparameter search
can see evaluation targets through the runner's candidate API. The runner is not a security
sandbox for arbitrary researcher code, nor a server enforcing a one-time test access policy.

The initial split is a fixed training/validation/test holdout, not walk-forward CV. Both validation
and explicit final-test phases train on training rows only. The seed is logged and used by the
built-in tree; custom factories must set their own stochastic components and log their settings.
Runtime/source/lock hashes improve auditability; bitwise numerical equality across platforms
is not promised. Freeze source revisions and data snapshots for published experiments.

## Kedro integration path

Conceptual pipeline stages remain `data_ingestion`, `data_validation`, `feature_engineering`,
`target_construction`, `dataset_preparation`, `benchmark`, and `reporting`.
`pipelines/nodes.py` already wraps the same provider, immutable config loader, and runner.
A future Kedro registry can connect these nodes through DataCatalog entries for raw/versioned
Parquet, protocol JSON, predictions and figures, with candidate configuration under parameters.
It must reuse the core feature/target/split functions, not introduce a second benchmark path.
Do not expose freely editable split parameters to estimator nodes. Core contracts accept no
Kedro objects. See the official [pipeline registry documentation](https://docs.kedro.org/en/stable/build/pipeline_registry/).

## Point-in-time fundamentals (design only)

Company statements must remain separate from market features. A future record contract must
retain asset/company identifier, concept, value, unit, currency, fiscal period start/end, filing
date/time, actual public availability timestamp, source/accession and revision identity.
Availability is the first forecast origin after public release, accounting for timezone, exchange
calendar and session availability. Unknown publication time requires a documented conservative
rule (for example next session), not an assumed same-day close. Restatements enter only after
their own release dates. Do not backfill new filings into earlier dates or substitute current
ratios for historical values. ETFs do not inherit company financial statements.

Revenue, net income, operating/free cash flow, debt, cash and equity can support documented
growth/margin/leverage KPIs. EBITDA requires an explicit reconstruction and accounting-concept
mapping; missing EBITDA is not EBIT. Missing concepts, currency conversions and period
comparability must be explicit. Fundamental-only, market-only, combined, market/complexity
and all-feature experiments require matched dates and eligible asset/task combinations.
No SEC ingestion or fundamental alignment is implemented in this iteration.

## Research stages and deferred work

1. **A: controlled synthetic data.** Predictive, redundant and random features to test whether
   future selectors recover information. The current synthetic market generator is only an
   offline smoke fixture; it is not evidence of MDL feature recovery.
2. **B: real market forecasting.** Frozen SPY daily OHLCV, causal indicators and future five-day
   volatility. Compare naive, Ridge and trees under the same saved benchmark.
3. **C: company fundamentals and combined data.** Only after publication-time alignment,
   accounting mappings, historical universes and panel leakage controls are implemented.

Later selectors may use correlation, mutual information, or a documented MDL coding scheme.
An MDL objective must separately account for model code length, feature-subset code length,
and conditional target code length, for example `L(subset) + L(model | subset) + L(y | model,X)`.
An arbitrary compression score is not an exact description length or Kolmogorov complexity.
Any coding parameters learned from data must respect the same training-only protocol.

CatBoost/XGBoost, joint outputs, recursion, rolling CV, SEC ingestion, MDL implementation,
feature-selection frequencies, complexity/error plots and code-length decomposition await
their actual underlying algorithms. Fitted coefficients (in transformed feature space) and
tree impurity importances are exported when available, labeled by meaning and fitted model
index. No importance is fabricated for the mean baseline. Deep learning, options valuation,
registries, tuning platforms and production
inference are outside the current scope.

MLflow uses the official [client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
for local experiments, run metadata, metrics and artifacts without global estimator-side tracking.
The adapter clones external estimators according to [scikit-learn's clone contract](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html).

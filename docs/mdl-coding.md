# First MDL feature-selection experiment

## Research question and scope

Can a feature subset selected by an explicit description-length criterion retain or improve
held-out volatility prediction while shortening the representation? The initial reference is
Ridge with alpha 1, an intercept, and training-only feature standardization. Every subset of
the authorized feature universe is evaluated, including the empty subset (training mean).
The default search limit is 12 features; five market features produce 32 candidates.

This is a deliberately simple **two-part code**, with design choices stated below. It is not
normalized maximum likelihood, exact Kolmogorov complexity, or a measured ZIP file size.
The [MDL tutorial by Grünwald](https://arxiv.org/html/math/0406077v1) explains why specifying
the model and data codes is necessary. The code here is our explicit experimental design.

## Shared information

The decoder knows the ordered training matrix X, its column names, n, and this versioned
coding specification. We evaluate a **conditional** target code, given X; the data-source,
feature-engineering and X-storage costs are not included. Costs therefore support comparisons
of subsets within one fixed universe and dataset, not arbitrary feature-generation programs.

The target-grid precision is fixed before evaluation: delta = 0.0001 in target units. For our
annualized-volatility target this is 0.01 percentage point. Other task units may need a different
predeclared precision. Alpha and precision are not chosen using validation/test performance.

## Subset code

For p available features and a subset S of size k:

```text
L(S) = ceil(log2(p+1)) + ceil(log2(comb(p,k)))
```

The first field is a fixed-width integer identifying k. The second identifies S's lexicographic
rank among the combinations of size k, in original feature order. Unused bit patterns are
allowed. The empty and full subsets need no rank bits. This is a prefix code conditional on p;
tests verify its Kraft sum over every subset is at most one.

## Model code

Fit Ridge on the selected training columns after standardization. Their means and population
standard deviations are deterministic functions of the shared X, so the decoder can derive
them. Constant columns follow StandardScaler's scale=1 convention. The model code contains:

1. One binary64 intercept.
2. k binary64 coefficients, in subset order.
3. One binary64 residual-distribution parameter q.

All words use big-endian IEEE-754 representation. The implementation actually serializes them
with `struct.pack`, retains the bytes as `parameter_hex`, and counts their size:

```text
L(model | S, X) = 64 * (k + 2)
```

The empty subset has an intercept equal to the training target mean, zero coefficients and
the same q field. It requires no placeholder input feature. Model fit and target coding use
training observations only. Full binary64 parameters are intentionally conservative and can
penalize weak predictors more strongly than alternative valid parameter codes.

## Conditional target code

For every training observation, define integer symbols using NumPy rounding (nearest, ties
to even): `z = rint(y/delta)`, `m = rint(prediction/delta)`, and `r = z-m`.
Only the target at this declared precision is reconstructed, not every original floating-point
bit. Values outside the safe symbol range of +/-2^50 are rejected.

Residuals use a normalized two-sided geometric distribution on the integers:

```text
P(r | q) = ((1-q)/(1+q)) * q^abs(r),  0 <= q < 1
a = mean(abs(r))
q = a / (sqrt(a*a + 1) + 1)
```

The q estimate is computed from training residuals and transmitted in the model code. If all
residuals are zero, q=0 assigns probability one to the zero sequence. Otherwise:

```text
ideal_target_bits = -sum(log2(P(r_i | q)))
L(target | model, S, X) = ceil(ideal_target_bits)
```

The ceiling is a Shannon prefix-code length for the joint integer sequence under the stated
distribution. The implementation computes lengths, not the residual bitstream itself. All
reported lengths are conditional on shared configuration/header information. Factorizing the
residual probabilities is a modeling simplification; it does not establish residual independence.

## Selection and downstream evaluation

Minimize `subset_bits + model_bits + target_bits` over all subsets. Ties favor fewer features,
then the original column order. No validation scores enter the search. The reference code is
for a scalar Ridge model; CLI v1 restricts `--selector mdl` to `--estimator ridge`. Multiple
target outputs are rejected explicitly. The Python selector remains an independent component;
applying its subset to another estimator does not make the reference code that estimator's cost.

The runner refits the same Ridge/scaler specification on the selected training columns, using
unrounded target values, as does the reference fitter. The target code's precision applies to
encoding, not to evaluation: benchmark MAE/RMSE still use the original continuous target.
With no selected features, Ridge reduces explicitly to its intercept-only training mean.
Unsupported estimator adapters reject empty subsets rather than silently switching models.

The original benchmark's data fingerprint, complete-case feature universe, eligibility mask,
chronological splits, purging and metric versions remain unchanged. Adding a selector is a
candidate change. The final SPY test period remains reserved for a later explicit evaluation.

The exploratory performance tolerance is **+2% RMSE relative to full Ridge**, declared before
the MDL validation comparison. This is an evaluation rule, not part of the MDL objective or
a statistically established non-inferiority margin. Description length and prediction error
are reported separately; a smaller training code does not guarantee better held-out error.

## Commands and artifacts

Using the existing original snapshot and protocol:

```powershell
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --output artifacts/spy-mdl-v1/full --tracking
uv run financial-mdl run --data data/spy.parquet --protocol protocols/spy-volatility-v1.json --estimator ridge --selector mdl --output artifacts/spy-mdl-v1/mdl --tracking
uv run financial-mdl compare artifacts/spy-mdl-v1/full artifacts/spy-mdl-v1/mdl --output artifacts/spy-mdl-v1/comparison --tracking
uv run financial-mdl mdl-synthetic --output artifacts/mdl-synthetic-v1 --seeds 0 1 2 3 4
```

Choose new output paths when repeating runs. For a fresh vendor snapshot, first create a new
protocol file bound to its fingerprint, as described in the README.

Each MDL market run exports `selection.json`, the complete `mdl_candidates.parquet`, and
Plotly code-decomposition and training-complexity/error figures. The table includes the
selected, full and constant models, actual parameter bytes, residual q, and every cost term.
MLflow keeps the normal validation metrics and logs code costs under `training/mdl/...`.
Run names distinguish the selector; comparison tables include an explicit selector column.

## Synthetic controls

The standalone synthetic experiment is independent regression, not a financial forecasting
benchmark. Each seed creates six features: signal_a, signal_b, an exact copy of signal_a,
and three independent noise features. Independent Normal(0,1) feature values and Normal(0,0.4)
target noise are shared across three scenarios:

| Scenario | Target |
| --- | --- |
| signal | 1 + 2*signal_a - 1.5*signal_b + noise |
| noise_only | 1 + noise, independent of all features |
| weak_signal | 1 + 0.05*signal_a + noise |

The first 600 rows train the selector and models; the remaining 300 are held out. Five fixed
seeds yield 15 paired full/selected comparisons, with no tuning on the held-out rows. Exact
copies are interchangeable, so success means recovering the informative groups without
redundancy, not a uniquely identifiable feature name. The weak-signal case examines the code's
capacity to drop a truly predictive variable when its benefit does not repay 64 parameter bits.

`experiment.json` records the generator version, configuration, fingerprints and code/runtime
provenance. Outputs include per-seed code searches, held-out metrics, selection frequencies
and Plotly figures. Synthetic results are exported as standalone artifacts; the market runs
and their comparison use MLflow. Scores from these two different tasks are never pooled.

## Limits

- The coding choice and target precision affect selection. Comparisons with alternative
  codes need predeclared configurations and new experiments, not retrospective tuning.
- Binary64 is a valid but expensive parameter representation. This first experiment does
  not claim optimal parameter precision or a universal model-complexity penalty.
- Overlapping volatility labels and serial dependence remain in each partition. The residual
  code is valid as a specified distribution, but may compress dependent data inefficiently.
- Five seeds and one SPY validation period support an exploratory result, not a significance
  claim, multi-regime robustness, or a conclusion about the final test period.

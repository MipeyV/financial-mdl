# MDL and held-out error for every feature subset

## Fixed comparison

This diagnostic completes the original independent synthetic controls: 600 training rows,
300 held-out rows, target precision 0.0001, Ridge alpha 1, seeds 0 through 4. The same generator
and split reproduce the original 600/300 experiment, rather than the larger sensitivity pool.
The [versioned configuration](../experiments/mdl-frontier-v1.json) is fixed before this run.

For each of three scenarios and each seed, fit all 64 subsets on training rows only.
Choose the MDL minimum before measuring held-out errors, then replay every candidate's
serialized binary64 parameters on the same held-out block. Standardization is derived from
training X only. The empty subset predicts the training target mean.

This yields 960 candidate evaluations. It does not introduce another selector, tune Ridge,
change the financial protocol, or evaluate the reserved 2024–2025 SPY test.

## Reading the figures

- Horizontal axis: **mean total training code length**, in bits, over the five seeds.
  It includes the subset identifier, model parameters and conditional target code.
- Vertical axis: **mean held-out RMSE**, measured on continuous targets.
- One point: one named subset, evaluated on every seed. Color: number of selected features.
- Highlighted markers: constant model, full Ridge, and subsets selected by training MDL on
  at least one seed. `selection_count` records how often each subset is selected.
- The HTML hover shows subset names, RMSE standard deviation and selection count.
  Standard deviations across five seeds are not confidence intervals.

Every scenario has a complete view and an explicitly labeled predictive zoom. The zoom frames
points with error at most 110% of full Ridge's mean RMSE, with 20% error-span padding
(at least 0.0005 times full Ridge RMSE) and 8% bit-span padding (at least one bit).
This enlarges small differences: read the numeric axes rather than interpreting visual
distance as a large relative gain. This is a display rule, not a selection threshold. The complete view
always retains all 64 points, including poor predictors outside the zoom.

Mean points do not represent one fitted model: they aggregate the same named subset over
different generated datasets. MDL selection is performed independently within each seed,
not by minimizing the displayed mean or choosing the lowest held-out RMSE. If selections
vary, several points can carry the MDL marker. Exact duplicate inputs can create overlapping
points; ties are resolved by the selector's declared feature order.

## Reproduction and presentation exports

```powershell
uv sync --locked --dev
uv run financial-mdl mdl-frontier --config experiments/mdl-frontier-v1.json --output artifacts/mdl-frontier-v1
```

Open `artifacts/mdl-frontier-v1/index.html` for the local overview. Each scenario exports
`*-all` and `*-zoom` in self-contained Plotly HTML, JSON, PNG (200 dpi) and editable SVG.
Static exports use Matplotlib's headless canvas and need no browser or graphics server.
`subsets.csv` retains all per-seed results; `summary.csv` contains one row per scenario/subset.
Selection diagnostics retain all model bytes and training costs. `experiment.json` records
the complete configuration, source/runtime provenance and fingerprint before fitting.

The synthetic held-out block is now inspected for all candidates. These are exploratory
diagnostics, not a fresh test set for selecting a new model. Keep any ensuing model changes
separate from claims about the reserved financial final test.

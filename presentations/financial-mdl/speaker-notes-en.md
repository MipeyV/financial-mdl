# English speaker notes

Six main slides, with a target duration of seven minutes. The four appendices are hidden in the normal slideshow. Rehearse aloud and adapt the wording to your own voice.

The French [research walkthrough](../../docs/research-walkthrough-fr.md) explains the method and includes likely questions. The text below matches the speaker notes embedded in the PowerPoint. Source paths refer to the repository.

## 1. MDL feature selection

Suggested time: 30 seconds.

Can we shorten a model description while preserving acceptable predictive performance? We investigate this question through Minimum Description Length, or MDL. We implemented feature selection, checked its behaviour on synthetic data with known signals, and then evaluated a financial forecasting task. The final result supports a simplification trade-off, with no consistent predictive advantage.

Sources:

- [docs/mdl-coding.md](../../docs/mdl-coding.md)
- [docs/spy-final-test.md](../../docs/spy-final-test.md)

## 2. The MDL selection criterion

Suggested time: 60 seconds.

The course presents MDL as the cost of describing a model plus the cost of describing the data given that model. We made this principle computable. For each feature combination, we fit Ridge on the training data. We count the bits needed to identify the features, describe the parameters, and encode the remaining target information at a fixed precision. We assume X is known. The table shows the actual trade-off: using three rather than five features slightly increases the residual cost, but saves more on parameters. The total falls from 20,157 to 20,045 bits, saving 112 bits. This training total selects the features. Held-out RMSE plays no part in that decision.

Sources:

- [docs/mdl-coding.md](../../docs/mdl-coding.md)
- [docs/mdl-results.md](../../docs/mdl-results.md)
- 5AI25 course, chapter 4, slides 20–21 and 24–28

## 3. A common temporal protocol

Suggested time: 60 seconds.

The financial task uses a frozen SPY snapshot with daily observations from 2015 to 2025. After each session close, we predict volatility over the next five trading sessions. Annualisation sets the units, while the forecast horizon remains five sessions. All candidates share the same eligible dates and temporal partitions. Fitting, standardisation and MDL selection use only 2015 to 2021. Validation in 2022 and 2023 supports comparison. We then froze the candidates before evaluating 2024 and 2025. Each target window must end inside its partition. Our exploratory rule allows at most two percent higher RMSE than full Ridge. It evaluates the MDL choice, without selecting the features.

Sources:

- [protocols/spy-volatility-v1.json](../../protocols/spy-volatility-v1.json)
- [experiments/spy-final-test-v1.json](../../experiments/spy-final-test-v1.json)
- [docs/spy-final-test.md](../../docs/spy-final-test.md)

## 4. Synthetic controls

Suggested time: 120 seconds.

Before studying the market, we test the mechanism where we know which variables matter. The strong-signal scenario has two true signals, an exact copy, and three noise features. Six features give 64 subsets, including the constant model. Each experiment uses 600 training observations and 300 held-out observations, across five seeds. Each point represents one named combination, averaged over those seeds. The horizontal axis shows training code length. The vertical axis shows held-out RMSE. Colour indicates the number of features. The magenta square marks the MDL choice, made independently for each seed. Here it retains both signals and removes redundancy and noise. The outlined square represents full Ridge. The zoom exposes very small RMSE differences, and the appendix shows all combinations. A subset containing noise can even achieve slightly lower held-out RMSE than the magenta square. We do not replace the MDL choice after inspecting that score. In the noise-only control, MDL selects a constant. It also selects a constant for the weak signal at this sample size. This does not establish that the signal is absent. Its coding benefit does not yet repay its representation cost. The sensitivity study in the appendix examines when it becomes worth retaining.

Sources:

- [experiments/mdl-frontier-v1.json](../../experiments/mdl-frontier-v1.json)
- [docs/results/mdl-frontier-v1-summary.csv](../../docs/results/mdl-frontier-v1-summary.csv)
- [docs/mdl-frontier.md](../../docs/mdl-frontier.md)
- [docs/mdl-sensitivity.md](../../docs/mdl-sensitivity.md)

## 5. Final test results

Suggested time: 90 seconds.

We compare three frozen candidates: full Ridge, Ridge with MDL selection, and persistence based on the previous twenty sessions. All three use the same 497 test origins. Full Ridge achieves RMSE 0.078215, compared with 0.078818 for MDL. MDL retains three of five features: the five-session return, and volatility over five and twenty sessions. Its test RMSE is therefore 0.77 percent higher than full Ridge, within the two percent tolerance declared before scoring the test. Its MAE is also slightly higher. Both Ridge models outperform persistence on this period. On validation, MDL had reduced RMSE by 2.25 percent. That improvement does not repeat on the test. The final finding is a smaller model with a small predictive cost. We changed no candidate after inspecting the test scores.

Sources:

- [docs/results/spy-final-test-v1-summary.csv](../../docs/results/spy-final-test-v1-summary.csv)
- [docs/results/spy-final-test-v1-verification.json](../../docs/results/spy-final-test-v1-verification.json)
- [docs/spy-final-test.md](../../docs/spy-final-test.md)

## 6. Conclusion and limitations

Suggested time: 60 seconds.

Our contribution is a reproducible experiment with an explicit description code, controls with known ground truth, and a separate final financial evaluation. In this case, MDL removes forty percent of the features and saves 112 training bits, while test RMSE stays within our exploratory tolerance. These figures describe different quantities. A shorter description does not guarantee better prediction. Coding parameters on 64 bits is an expensive choice that affects selection. We do not measure exact Kolmogorov complexity. We also study one financial series, with overlapping five-session targets. We therefore make no claim of statistical superiority or universal performance. A useful next study would compare alternative codes under a new evaluation plan.

Sources:

- [docs/mdl-coding.md](../../docs/mdl-coding.md)
- [docs/mdl-sensitivity.md](../../docs/mdl-sensitivity.md)
- [docs/spy-final-test.md](../../docs/spy-final-test.md)

## 7. Appendix A: the code in bits

Hidden appendix for questions.

For p available features and a subset of size k, we reserve ceil(log2(p+1)) bits for k, then ceil(log2(C(p,k))) bits for the combination rank. With p=5 and k=3, this gives 3+4=7 bits. Keeping all five requires 3+0=3 bits. Parameters cost 64(k+2) bits: k coefficients, an intercept, and the residual parameter q. We quantise targets at delta=0.0001 and model integer residuals with a two-sided geometric distribution. Their length is the ceiling of the joint negative log probability. We actually serialise the parameter words. We calculate residual code lengths without producing a complete residual bitstream. Reconstruction concerns the quantised symbols, conditional on known X.

Sources:

- [docs/mdl-coding.md](../../docs/mdl-coding.md)
- [src/financial_mdl/mdl.py](../../src/financial_mdl/mdl.py)

## 8. Appendix B: the weak signal

Hidden appendix for questions.

The weak-signal generator has coefficient 0.05 and noise standard deviation 0.4. At smaller sample sizes, its benefit cannot repay the coefficient cost. Adding the feature costs 67 representation bits: 64 for its coefficient and three extra subset-identification bits. At 9,600 training observations and precision 0.0001, it saves between 99 and 137 target bits and appears in all five selected subsets. At precision 0.1, only two seeds repay that cost. Sample size and target precision therefore affect selection. These configurations did not change the code used for SPY.

Sources:

- [docs/results/mdl-sensitivity-v1-summary.csv](../../docs/results/mdl-sensitivity-v1-summary.csv)
- [docs/mdl-sensitivity.md](../../docs/mdl-sensitivity.md)
- [experiments/mdl-sensitivity-v1.json](../../experiments/mdl-sensitivity-v1.json)

## 9. Appendix C: all combinations

Hidden appendix for questions.

This view includes all 64 combinations in the strong-signal scenario. Models that omit a strong signal have much higher errors. Some points overlap because one feature is an exact copy. Noise features can improve an observed held-out score by chance, including an average across a few seeds. MDL selection remains based on training data. Held-out scores are diagnostics. The small differences in the main-slide zoom should not be presented as large predictive gains.

Sources:

- [docs/results/mdl-frontier-v1-summary.csv](../../docs/results/mdl-frontier-v1-summary.csv)
- [docs/mdl-frontier.md](../../docs/mdl-frontier.md)

## 10. Appendix D: scope of the result

Hidden appendix for questions.

Our lengths describe the targets conditional on X, the feature universe and shared configuration. We do not count snapshot storage or feature-generation programs. This is an explicit computable code, with no claim to exact uncomputable Kolmogorov complexity or universal optimality. The two percent tolerance is separate from MDL and establishes no statistical non-inferiority. Ridge fits coefficients through regularised squared error, then MDL selects among those fitted feature combinations. We have now inspected the final test. Later changes informed by its scores require a new evaluation plan. The saved protocol, data fingerprints and prediction exports make the comparison auditable.

Sources:

- [docs/mdl-coding.md](../../docs/mdl-coding.md)
- [docs/spy-final-test.md](../../docs/spy-final-test.md)
- [docs/results/spy-final-test-v1-verification.json](../../docs/results/spy-final-test-v1-verification.json)

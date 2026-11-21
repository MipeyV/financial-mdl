# Financial MDL project report

The English report follows the supplied course template: Abstract, Problem, Method,
Results, Discussion and Bibliography, followed by an appendix of PowerShell reproduction
commands. It retains the existing Abstract and Problem text and the institutional banner.

- [Editable report](mdl_report.docx): preferred Word version, with nine native equations,
  seven result/protocol tables and a two-panel diagnostic figure. The institutional banner
  is an additional table. The document renders to eleven pages in Microsoft Word.
- [Word 97 compatibility copy](mdl_report.doc): completed version of the original file.
  Equations are images in this legacy format; use the DOCX to edit their mathematical content.

The figures and numeric tables use the frozen experiment results. Report preparation does
not refit models or reevaluate the final test. The final finding remains 40% fewer features
with 0.77% higher test RMSE than full Ridge, within the declared exploratory 2% tolerance.

The appendix distinguishes independently reproducible synthetic controls from exact
financial replay, which requires the original `data/spy.parquet` snapshot and its metadata
sidecar. These data are local and ignored by Git. A fresh vendor download has a different
provenance and may require another protocol.

Scientific specifications and result tables:

- [MDL code](../mdl-coding.md)
- [Synthetic sensitivity](../mdl-sensitivity.md)
- [All-subset diagnostics](../mdl-frontier.md)
- [Final financial evaluation](../spy-final-test.md)
- [Validation and test metrics](../results/spy-final-test-v1-summary.csv)

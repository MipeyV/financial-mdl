# Repository workflow

- Before editing, work on a dedicated branch based on an up-to-date `main`.
  Use `<type>/<short-kebab-case-description>` with `feature`, `fix`, `chore`,
  `docs`, `refactor`, `test`, or `ci` as the type.
- Never commit or push changes directly to `main`. Keep each branch focused on one change.
- Run the relevant tests and both Ruff checks before committing. CI uses the locked uv
  environment; keep `uv.lock` consistent with `pyproject.toml`.
- Push the working branch and open a pull request targeting `main`. Describe the resulting
  behavior and the validation performed. Address failed CI checks on that same branch.
- Merge a pull request only when the user explicitly asks. A request to commit, push, or
  open a PR does not authorize merging it.
- Preserve the benchmark's scientific rules: candidate configuration stays separate from
  protocol configuration, fitted components use training data only, and final test evaluation
  remains explicit. Use synthetic/offline fixtures in CI, including local MLflow integration.

See `CONTRIBUTING.md` for commands and the GitHub Actions checks.

# Contributing

## Branches and pull requests

All changes go through a dedicated branch and a pull request into `main`.
Branch names use `<type>/<short-kebab-case-description>`:

| Prefix | Purpose | Example |
| --- | --- | --- |
| `feature/` | New functionality | `feature/correlation-selector` |
| `fix/` | Bug fixes | `fix/target-window-alignment` |
| `chore/` | Maintenance and tooling | `chore/github-actions-ci` |
| `docs/` | Documentation | `docs/benchmark-protocol` |
| `refactor/` | Internal restructuring | `refactor/forecast-coordinates` |
| `test/` | Tests and fixtures | `test/selector-leakage` |
| `ci/` | CI changes | `ci/python-matrix` |

Start from a clean working tree and an up-to-date `main`:

```powershell
git switch main
git pull --ff-only
git switch -c feature/correlation-selector
```

Keep commits focused, push the branch, and open a PR targeting `main`. Describe the problem,
resulting behavior and checks performed. CI checks the branch naming convention for PRs.
Do not push directly to `main`. Merge after review and successful CI; an agent needs an
explicit user instruction to merge.

## Local checks

```powershell
uv sync --locked --dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -m "not tracking"
uv build
```

To run the optional local MLflow integration:

```powershell
uv sync --locked --dev --extra tracking
uv run --no-sync pytest -m tracking
```

Use `uv run ruff format .` to apply formatting. When changing dependencies, regenerate
`uv.lock` with uv and commit both the manifest and lockfile. Tests use synthetic data and
local artifacts; downloading vendor prices is not part of CI.

## GitHub Actions

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs for PRs into `main`, pushes to
`main`, and manual dispatches. It contains:

- **Quality**: PR branch naming, locked dependency installation, Ruff lint and formatting,
  and source/wheel package builds.
- **Tests**: the complete non-tracking suite, including schemas, data validation, temporal
  leakage, estimators, forecasting, CLI and report generation. Linux covers Python 3.11,
  3.12 and 3.13; Windows covers Python 3.12.
- **MLflow integration**: Python 3.12 on Linux with the tracking extra, exercising local
  SQLite, parameters, metrics, predictions and comparison artifacts.
- **CI**: a stable aggregate check that succeeds only when every job succeeds. It also
  fails when an upstream job is cancelled or skipped.

Dependencies and tool versions are pinned; superseded runs are cancelled per PR/ref.
GitHub repository rules can require the **CI** status check and pull requests for `main`.
The workflow files do not configure repository-level branch protection themselves.

The uv setup follows the [official GitHub Actions integration guide](https://docs.astral.sh/uv/guides/integration/github/).

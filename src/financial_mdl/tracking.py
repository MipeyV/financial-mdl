"""Optional tracking adapters; importing this module never imports MLflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NoOpTracker:
    def record(self, metadata: dict[str, Any], metrics: dict[str, float], directory: str) -> None:
        pass


@dataclass(frozen=True)
class LocalMLflowTracker:
    root: Path = Path("artifacts/mlflow")
    experiment_name: str = "financial-mdl"

    def record(self, metadata: dict[str, Any], metrics: dict[str, float], directory: str) -> None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as error:
            raise ImportError("Install tracking: uv sync --extra tracking") from error

        root = self.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
        experiment = client.get_experiment_by_name(self.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                self.experiment_name,
                artifact_location=(root / "artifacts").as_uri(),
            )
        )
        run = client.create_run(
            experiment_id,
            tags={
                "mlflow.runName": metadata["run_name"],
                "benchmark_fingerprint": metadata["benchmark_fingerprint"],
                "git_commit": metadata["git_commit"] or "uncommitted",
                "phase": metadata["phase"],
            },
        )
        run_id = run.info.run_id
        try:
            params = {
                key: metadata[key]
                for key in (
                    "benchmark_id",
                    "benchmark_version",
                    "dataset_fingerprint",
                    "seed",
                    "strategy",
                    "estimator",
                    "selected_features",
                    "feature_families",
                    "selector",
                    "transformer",
                    "horizons",
                    "target_names",
                    "splits",
                )
            }
            for key, value in params.items():
                # Full metadata is always retained in run.json, including large parameter maps.
                encoded = json.dumps(value, sort_keys=True)
                client.log_param(run_id, key, encoded[:5000])
            for key, value in metrics.items():
                client.log_metric(run_id, f"{metadata['phase']}/{key}", value)
            client.log_artifacts(run_id, directory)
            client.set_terminated(run_id, "FINISHED")
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import typing as t

import warnings

try:  # pragma: no cover - optional dep
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional dep
    joblib = None  # type: ignore[assignment]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


@dataclass(frozen=True)
class GeminiGrader:
    root: Path
    feature_names: list[str]
    models: dict[str, t.Any]

    @staticmethod
    def load(root: Path) -> "GeminiGrader":
        if joblib is None:  # pragma: no cover - optional dep
            raise ModuleNotFoundError("joblib is required to load learned graders (pip install joblib)")
        root = root.expanduser().resolve()
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace") or "{}")
        feature_names = list(meta.get("feature_names") or [])
        if not feature_names:
            raise ValueError("meta.json missing feature_names")

        models: dict[str, t.Any] = {}
        # Avoid noisy warnings when sklearn versions differ across machines. Models remain
        # best-effort; training should align versions, but prediction shouldn't spam logs.
        try:  # pragma: no cover - optional dep
            from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except Exception:
            pass
        for item in meta.get("models") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            p = str(item.get("path") or "").strip()
            if not label or not p:
                continue
            mpath = root / p
            if not mpath.exists():
                continue
            models[label] = joblib.load(mpath)

        if not models:
            raise ValueError("No models loaded (meta.json models list is empty or missing files).")
        return GeminiGrader(root=root, feature_names=feature_names, models=models)

    def predict(self, features: dict[str, float | int | None]) -> dict[str, float]:
        # Build row in the exact feature order.
        row: list[float | None] = []
        for k in self.feature_names:
            v = features.get(k) if isinstance(features, dict) else None
            if isinstance(v, bool):
                row.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                row.append(float(v))
            else:
                row.append(None)

        # sklearn pipelines accept None? Safer to use NaN for missing.
        import math
        import warnings

        # Avoid spamming stdout for known-safe imputer behavior.
        warnings.filterwarnings(
            "ignore",
            message=r"Skipping features without any observed values:.*",
            category=UserWarning,
            module=r"sklearn\.impute\._base",
        )

        x = [[float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else float("nan") for v in row]]
        out: dict[str, float] = {}
        for label, model in self.models.items():
            try:
                pred = float(model.predict(x)[0])
            except Exception:
                continue
            # Range-clip for sanity.
            if label in {"overall_score", "match_score"}:
                pred = _clip(pred, 0.0, 10.0)
            else:
                pred = _clip(pred, 0.0, 5.0)
            out[label] = pred
        return out

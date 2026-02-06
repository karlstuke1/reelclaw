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


@dataclass(frozen=True)
class PairwiseRanker:
    """
    Lightweight learning-to-rank model.

    Training uses pairwise preferences within a project:
    - Fit imputer+scaler on absolute feature rows X
    - Train a linear classifier on (X_i - X_j) in scaled space, fit_intercept=False
    - At inference, we score a variant by s(x) = w · x_scaled
    """

    root: Path
    feature_names: list[str]
    imputer: t.Any
    scaler: t.Any
    coef: t.Any  # numpy array shape (n_features,)

    @staticmethod
    def load(root: Path) -> "PairwiseRanker":
        if joblib is None:  # pragma: no cover - optional dep
            raise ModuleNotFoundError("joblib is required to load rankers (pip install joblib)")
        root = root.expanduser().resolve()
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace") or "{}")
        feature_names = list(meta.get("feature_names") or [])
        if not feature_names:
            raise ValueError("meta.json missing feature_names")
        model_rel = str(meta.get("model_path") or meta.get("model") or "").strip() or "ranker.joblib"
        model_path = (root / model_rel).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing ranker model: {model_path}")

        # Avoid noisy warnings when sklearn versions differ across machines.
        try:  # pragma: no cover - optional dep
            from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except Exception:
            pass

        blob = joblib.load(model_path)
        if not isinstance(blob, dict):
            raise ValueError("ranker.joblib must be a dict with imputer/scaler/coef")
        imputer = blob.get("imputer")
        scaler = blob.get("scaler")
        coef = blob.get("coef")
        if imputer is None or scaler is None or coef is None:
            raise ValueError("ranker.joblib missing imputer/scaler/coef")
        return PairwiseRanker(root=root, feature_names=feature_names, imputer=imputer, scaler=scaler, coef=coef)

    def score(self, features: dict[str, float | int | None]) -> float:
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

        import math
        import numpy as np  # type: ignore

        x = np.array(
            [[float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else float("nan") for v in row]],
            dtype=np.float32,
        )
        # sklearn transformers accept NaN for missing.
        x2 = self.imputer.transform(x)
        x3 = self.scaler.transform(x2)

        # coef can be shape (1,n) or (n,)
        w = self.coef
        try:
            w = w.reshape(-1)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return float((x3.reshape(-1) * w).sum())
        except Exception:
            # Last-resort: attempt python-level dot.
            vals = list(t.cast(t.Iterable[float], x3.reshape(-1)))  # type: ignore[attr-defined]
            ww = list(t.cast(t.Iterable[float], w))  # type: ignore[arg-type]
            n = min(len(vals), len(ww))
            return float(sum(float(vals[i]) * float(ww[i]) for i in range(n)))


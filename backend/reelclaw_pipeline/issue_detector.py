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
class IssueDetector:
    """
    Multi-label issue predictor trained from graded videos (top_issues/actionable_fixes).

    The model is meant to be "cheap steering":
    - take fast, no-render features
    - predict likely issues (pacing/hook/clarity/etc.)
    - apply deterministic fix heuristics before final render/polish
    """

    root: Path
    feature_names: list[str]
    issue_names: list[str]
    pipeline: t.Any

    @staticmethod
    def load(root: Path) -> "IssueDetector":
        if joblib is None:  # pragma: no cover - optional dep
            raise ModuleNotFoundError("joblib is required to load issue detectors (pip install joblib)")
        root = root.expanduser().resolve()
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace") or "{}")
        feature_names = list(meta.get("feature_names") or [])
        issue_names = list(meta.get("issue_names") or [])
        if not feature_names:
            raise ValueError("meta.json missing feature_names")
        if not issue_names:
            raise ValueError("meta.json missing issue_names")
        model_rel = str(meta.get("model_path") or meta.get("model") or "").strip() or "issue_detector.joblib"
        model_path = (root / model_rel).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing issue detector model: {model_path}")

        # Avoid noisy warnings when sklearn versions differ across machines.
        try:  # pragma: no cover - optional dep
            from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except Exception:
            pass

        pipe = joblib.load(model_path)
        return IssueDetector(root=root, feature_names=feature_names, issue_names=issue_names, pipeline=pipe)

    def predict_proba(self, features: dict[str, float | int | None]) -> dict[str, float]:
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

        x = [[float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else float("nan") for v in row]]

        # OneVsRestClassifier(LogReg) returns shape (n, k).
        probs: list[float] = []
        try:
            p = self.pipeline.predict_proba(x)
            if hasattr(p, "tolist"):
                p = p.tolist()
            if isinstance(p, list) and p and isinstance(p[0], list):
                probs = [float(v) for v in p[0]]
        except Exception:
            probs = []

        out: dict[str, float] = {}
        for name, pr in zip(self.issue_names, probs, strict=False):
            # Clip for sanity.
            v = float(pr)
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            out[str(name)] = v
        return out


from __future__ import annotations

from dataclasses import dataclass


class CancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgressUpdate:
    stage: str
    current: int
    total: int
    message: str


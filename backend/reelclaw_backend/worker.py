from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reelclaw_backend.config import Settings
from reelclaw_backend.storage import JobStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _apply_pro_defaults(env: dict[str, str]) -> None:
    """
    Keep iOS/prod pipeline in line with the legacy pro-mode scripts by turning on the
    higher-quality "editor brain" knobs, but without overriding explicit env.
    """
    env.setdefault("FOLDER_EDIT_BEAT_SYNC", "1")
    env.setdefault("FOLDER_EDIT_STORY_PLANNER", "1")
    env.setdefault("SHOT_INDEX_MODE", "scene")
    env.setdefault("SHOT_INDEX_WORKERS", "4")
    # Shot-level tagging is a key part of the pro pipeline "editor brain".
    # Cap with SHOT_TAG_MAX to keep local runs bounded on huge libraries.
    env.setdefault("SHOT_TAGGING", "1")
    env.setdefault("SHOT_TAG_MAX", "250")
    env.setdefault("REF_SEGMENT_FRAME_COUNT", "3")
    env.setdefault("REASONING_EFFORT", "high")
    # Directed quality lift: generate more internal candidates, fix worst segments on finalists.
    env.setdefault("VARIANT_FIX_ITERS", "1")
    env.setdefault("VARIANT_PRO_MACRO", "beam")


def _count_variants(finals_dir: Path) -> int:
    try:
        return len(list(finals_dir.glob("v*.mov"))) + len(list(finals_dir.glob("v*.mp4")))
    except Exception:
        return 0


def _count_candidates(variants_dir: Path) -> int:
    """
    Count candidate renders in variants/ (independent of finals selection/copying).
    """
    try:
        return len(list(variants_dir.glob("v*/final_video.mov")))
    except Exception:
        return 0


def _internal_variant_count(*, finals: int, pro_mode: bool) -> int:
    raw = os.getenv("REELCLAW_INTERNAL_VARIANTS", "").strip()
    try:
        configured = int(float(raw)) if raw else 0
    except Exception:
        configured = 0
    if configured > 0:
        return max(int(finals), int(configured))
    # Default: keep old behavior unless pro mode is enabled.
    return max(int(finals), 16) if pro_mode else int(finals)


@dataclass
class JobRunner:
    settings: Settings
    jobs: JobStore
    _threads: dict[str, threading.Thread]

    @classmethod
    def create(cls, *, settings: Settings, jobs: JobStore) -> "JobRunner":
        return cls(settings=settings, jobs=jobs, _threads={})

    def start(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        status = str(job.get("status") or "")
        if status == "succeeded":
            return

        existing = self._threads.get(job_id)
        if existing and existing.is_alive():
            return

        t = threading.Thread(target=self._run_job, name=f"job-{job_id}", args=(job_id,), daemon=True)
        self._threads[job_id] = t
        t.start()

    def _run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        variations = int(job.get("variations") or 3)
        burn_overlays = bool(job.get("burn_overlays") or False)
        director = str(job.get("director") or "").strip().lower() or None
        if director not in {"code", "gemini", "auto"}:
            director = None

        pro_mode = _truthy_env("REELCLAW_PRO_MODE", "1")
        # Gemini/auto director requires pro mode (shot candidates); force it per-job if requested.
        if director in {"gemini", "auto"}:
            pro_mode = True

        internal_variants = _internal_variant_count(finals=variations, pro_mode=pro_mode)
        # Cost/latency guardrail: Gemini director does one LLM call per variant. Cap by default.
        if director in {"gemini", "auto"} and internal_variants > 12 and not _truthy_env("ALLOW_GEMINI_DIRECTOR_MANY", "0"):
            internal_variants = max(int(variations), 12)

        job_dir = self.jobs.job_dir(job_id)
        uploads_dir = job_dir / "uploads"
        pipeline_root = job_dir / "pipeline"
        pipeline_root.mkdir(parents=True, exist_ok=True)

        log_path = pipeline_root / f"job_{job_id}_{_now_ms()}.log"
        finals_dir = pipeline_root / "finals"
        variants_dir = pipeline_root / "variants"
        backend_root = Path(__file__).resolve().parents[1]

        try:
            if not uploads_dir.exists() or not any(uploads_dir.iterdir()):
                self.jobs.update(
                    job_id,
                    status="failed",
                    stage="Missing clips",
                    message="Upload at least 1 clip before starting.",
                    error="no_uploads",
                    error_code="no_uploads",
                    error_detail=None,
                )
                return

            self.jobs.update(
                job_id,
                status="running",
                stage="Starting",
                message="Booting up the editors…",
                progress_current=0,
                progress_total=max(1, variations),
                pipeline_root=str(pipeline_root),
                error=None,
                error_code=None,
                error_detail=None,
            )

            cmd = [
                sys.executable,
                "-m",
                "reelclaw_pipeline.run_folder_edit_variants",
                "--reel",
                str(job.get("reference_reel_url") or ""),
                "--folder",
                str(uploads_dir),
                "--variants",
                str(internal_variants),
                "--finals",
                str(variations),
                "--out",
                str(pipeline_root),
                "--seed",
                "1337",
                "--model",
                self.settings.director_model,
            ]
            if pro_mode:
                cmd.append("--pro")
            if director:
                cmd.extend(["--director", str(director)])
            if burn_overlays:
                cmd.append("--burn-overlays")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(backend_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            if pro_mode:
                _apply_pro_defaults(env)
            # Allow using the full reel length (no -t) by setting this to 0.
            if self.settings.reference_analysis_max_seconds is None:
                env["REEL_ANALYSIS_MAX_SECONDS"] = "0"
            else:
                env["REEL_ANALYSIS_MAX_SECONDS"] = str(int(self.settings.reference_analysis_max_seconds))

            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ {' '.join(cmd)}\n\n")
                log.flush()

                proc = subprocess.Popen(
                    cmd,
                    cwd=backend_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

                last_progress = -1
                while proc.poll() is None:
                    cur = _count_candidates(variants_dir)
                    if cur != last_progress:
                        last_progress = cur
                        # Scale candidate progress to the number of finals the user requested.
                        scaled = 0
                        try:
                            scaled = int(round((float(cur) / max(1.0, float(internal_variants))) * float(variations)))
                        except Exception:
                            scaled = 0
                        self.jobs.update(
                            job_id,
                            status="running",
                            stage=f"Exploring candidates ({min(cur, internal_variants)}/{internal_variants})",
                            message="This can take a few minutes depending on clip length.",
                            progress_current=min(max(0, scaled), variations),
                            progress_total=max(1, variations),
                        )
                    time.sleep(1.0)

                rc = int(proc.returncode or 0)
                produced = _count_variants(finals_dir)
                if rc != 0:
                    self.jobs.update(
                        job_id,
                        status="failed",
                        stage="Failed",
                        message=f"Pipeline exited with code {rc}. Check logs: {log_path.name}",
                        error=f"exit_{rc}",
                        error_code=f"exit_{rc}",
                        error_detail=None,
                    )
                    return

                if produced <= 0:
                    self.jobs.update(
                        job_id,
                        status="failed",
                        stage="No outputs",
                        message="Pipeline finished but produced no variants.",
                        error="no_outputs",
                        error_code="no_outputs",
                        error_detail=None,
                    )
                    return

                self.jobs.update(
                    job_id,
                    status="succeeded",
                    stage="Done",
                    message="Your variations are ready.",
                    progress_current=min(produced, variations),
                    progress_total=max(1, variations),
                    error=None,
                    error_code=None,
                    error_detail=None,
                )
        except Exception as exc:
            self.jobs.update(
                job_id,
                status="failed",
                stage="Failed",
                message=str(exc),
                error="exception",
                error_code="exception",
                error_detail=str(exc),
            )

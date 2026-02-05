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


def _count_variants(finals_dir: Path) -> int:
    try:
        return len(list(finals_dir.glob("v*.mov"))) + len(list(finals_dir.glob("v*.mp4")))
    except Exception:
        return 0


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

        job_dir = self.jobs.job_dir(job_id)
        uploads_dir = job_dir / "uploads"
        pipeline_root = job_dir / "pipeline"
        pipeline_root.mkdir(parents=True, exist_ok=True)

        log_path = pipeline_root / f"job_{job_id}_{_now_ms()}.log"
        finals_dir = pipeline_root / "finals"
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
                str(variations),
                "--out",
                str(pipeline_root),
                "--seed",
                "1337",
                "--model",
                self.settings.director_model,
            ]
            if burn_overlays:
                cmd.append("--burn-overlays")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(backend_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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
                    cur = _count_variants(finals_dir)
                    if cur != last_progress:
                        last_progress = cur
                        self.jobs.update(
                            job_id,
                            status="running",
                            stage=f"Rendering variations ({min(cur, variations)}/{variations})",
                            message="This can take a few minutes depending on clip length.",
                            progress_current=min(cur, variations),
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

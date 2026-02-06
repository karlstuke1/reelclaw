from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import typing as t


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_api_key() -> str | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = Path("openrouter")
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def _ensure_backend_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = (repo_root / "backend").resolve()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout_s: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=float(timeout_s))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}). Log: {log_path}")


def _discover_reference_files(refs_dir: Path, *, limit: int) -> list[Path]:
    """
    Prefer refs_dir/manifest.json (stable ordering), else fall back to sorted mp4/mov.
    """
    refs_dir = refs_dir.expanduser().resolve()
    manifest = refs_dir / "manifest.json"
    out: list[Path] = []
    if manifest.exists():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8", errors="replace") or "{}")
            reels = doc.get("reels") if isinstance(doc, dict) else None
            if isinstance(reels, list):
                for r in reels:
                    if not isinstance(r, dict):
                        continue
                    lp = str(r.get("local_path") or "").strip()
                    if not lp:
                        continue
                    p = Path(lp).expanduser()
                    if not p.is_absolute():
                        p = (refs_dir / lp).resolve()
                    if p.exists():
                        out.append(p.resolve())
        except Exception:
            out = []

    if not out:
        vids = sorted([p for p in refs_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}])
        out = [p.resolve() for p in vids]

    if limit > 0:
        out = out[:limit]
    return out


def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v", ".webm"}


def _sample_media_folder(src: Path, *, dst: Path, limit: int) -> tuple[Path, list[Path]]:
    """
    Create a deterministic symlink sample of a large media folder to speed indexing.
    """
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    vids = sorted([p for p in src.iterdir() if _is_video_file(p)])
    picked = vids[: max(1, int(limit))]
    for p in picked:
        link = dst / p.name
        if link.exists():
            continue
        try:
            link.symlink_to(p)
        except Exception:
            # Fallback: keep the original path in the manifest even if symlink fails.
            pass
    # Also write a manifest for reproducibility.
    manifest = {
        "created_at": _now_ts(),
        "source_folder": str(src),
        "count": int(len(picked)),
        "picked": [str(p) for p in picked],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dst, picked


def _build_refs_library(*, refs: list[Path], exclude: Path | None, dst: Path, limit: int | None = None) -> Path:
    """
    Build a deterministic symlink library from downloaded reference reels.
    Useful when your local footage folder is style-mismatched (e.g., no drift clips).
    """
    dst = dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    exclude_resolved = exclude.resolve() if isinstance(exclude, Path) else None

    picked: list[Path] = []
    for p in refs:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if exclude_resolved is not None and rp == exclude_resolved:
            continue
        if not _is_video_file(rp):
            continue
        picked.append(rp)
        if isinstance(limit, int) and limit > 0 and len(picked) >= int(limit):
            break

    for i, p in enumerate(picked, start=1):
        # Prefer stable basenames so callers can exclude by filename without needing to know
        # symlink renames. On collision, prefix deterministically.
        name = p.name
        link = dst / name
        if link.exists():
            if link.is_symlink():
                try:
                    if link.resolve() == p:
                        continue
                except Exception:
                    pass
            name = f"{i:03d}_{p.name}"
        link = dst / name
        if link.exists():
            continue
        try:
            link.symlink_to(p)
        except Exception:
            pass

    manifest = {
        "created_at": _now_ts(),
        "exclude": str(exclude_resolved) if exclude_resolved is not None else None,
        "count": int(len(picked)),
        "picked": [str(p) for p in picked],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dst


def _find_best_final(project_root: Path) -> Path:
    finals = project_root / "finals"
    for cand in [finals / "v001.mov", finals / "v001.mp4"]:
        if cand.exists():
            return cand.resolve()
    vids = sorted([p for p in finals.glob("*") if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v", ".webm"}])
    if vids:
        return vids[0].resolve()
    raise FileNotFoundError(f"No finals found under: {finals}")


def _find_reference_clip(project_root: Path) -> Path:
    cand = project_root / "reference" / "analysis_clip.mp4"
    if cand.exists():
        return cand.resolve()
    cand2 = project_root / "analysis_clip.mp4"
    if cand2.exists():
        return cand2.resolve()
    raise FileNotFoundError(f"Missing reference analysis clip under: {project_root}")


def _extract_thumb(video_path: Path, *, out_path: Path, at_s: float) -> None:
    _ensure_backend_on_syspath()
    from reelclaw_pipeline.folder_edit_pipeline import _extract_frame  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _extract_frame(video_path=video_path, at_s=float(at_s), out_path=out_path, timeout_s=120.0)
    except Exception:
        # Best-effort thumbnails only.
        pass


def _score_compare(
    *,
    api_key: str,
    judge_model: str,
    output_video: Path,
    reference_video: Path,
    max_mb: float,
    tmp_dir: Path,
    criteria: str,
) -> dict[str, t.Any]:
    _ensure_backend_on_syspath()
    from reelclaw_pipeline.folder_edit_evaluator import evaluate_edit_full_video_compare  # type: ignore
    from reelclaw_pipeline.video_proxy import ensure_inlineable_video  # type: ignore

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_proxy, out_meta = ensure_inlineable_video(output_video, max_mb=float(max_mb), tmp_dir=tmp_dir, allow_proxy=True)
    ref_proxy, ref_meta = ensure_inlineable_video(reference_video, max_mb=float(max_mb), tmp_dir=tmp_dir, allow_proxy=True)

    ev = evaluate_edit_full_video_compare(
        api_key=api_key,
        model=str(judge_model),
        output_video_path=Path(out_proxy),
        reference_video_path=Path(ref_proxy),
        criteria=str(criteria),
        prompt_variant="compare_general",
        timeout_s=240.0,
    )
    # Normalize to stable keys where possible.
    res = dict(ev.result or {})
    return {
        "ok": True,
        "result": res,
        "usage": ev.usage,
        "model_requested": ev.model_requested,
        "model_used": ev.model_used,
        "proxies": {
            "output": {"path": str(out_proxy), "meta": out_meta},
            "reference": {"path": str(ref_proxy), "meta": ref_meta},
        },
    }


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class ScoredRun:
    director: str
    project_root: Path
    final_video: Path | None
    reference_clip: Path | None
    judge: dict[str, t.Any]
    ok: bool = True
    error: str | None = None
    log_path: Path | None = None

    def overall(self) -> float | None:
        return _safe_float(((self.judge.get("result") or {}) if isinstance(self.judge, dict) else {}).get("overall_score"))

    def stability(self) -> float | None:
        return _safe_float(((self.judge.get("result") or {}) if isinstance(self.judge, dict) else {}).get("stability"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed benchmark: compare --director code vs gemini using Gemini Pro compare-mode scoring.")
    ap.add_argument("--refs-dir", default="Outputs/final_edit/refs_sexinporsche_1770009006", help="Folder with reference reels + manifest.json")
    ap.add_argument("--refs-limit", type=int, default=8, help="How many reference reels to benchmark (0=all in manifest)")
    ap.add_argument("--folder", default="/Users/work/Downloads/VideosJanuary26", help="Footage folder (library clips)")
    ap.add_argument(
        "--footage-from-refs",
        action="store_true",
        help="Use the downloaded reference reels (excluding the current ref) as the footage library. Helps when your local folder is style-mismatched.",
    )
    ap.add_argument(
        "--footage-sample",
        type=int,
        default=0,
        help="Sample N videos from --folder via symlinks for speed (0=use full folder). NOTE: sampling changes asset ids and can bypass tag caches.",
    )
    ap.add_argument(
        "--shot-index-max-videos",
        type=int,
        default=80,
        help="Cap the pro shot index to N videos deterministically (0=all). Speeds up local benchmarks on huge libraries.",
    )
    ap.add_argument("--variants", type=int, default=6, help="Internal variants to generate per director")
    ap.add_argument("--finals", type=int, default=1, help="Finals to write per director")
    ap.add_argument("--seed", type=int, default=1337, help="Base seed (per-ref seeds are derived deterministically)")
    ap.add_argument("--analysis-model", default=os.getenv("DIRECTOR_MODEL", "google/gemini-3-pro-preview"))
    ap.add_argument("--judge-model", default=os.getenv("CRITIC_MODEL", "google/gemini-3-pro-preview"))
    ap.add_argument("--max-mb", type=float, default=8.0, help="Max proxy size (MB) to inline to judge")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out", help="Output root folder (default: Outputs/BenchDirectors_<ts>)")
    args = ap.parse_args()

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (env) or ./openrouter file")

    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out).expanduser().resolve() if args.out else (repo_root / "Outputs" / f"BenchDirectors_{int(time.time())}")
    out_root.mkdir(parents=True, exist_ok=True)

    refs_dir = Path(args.refs_dir).expanduser().resolve()
    ref_paths = _discover_reference_files(refs_dir, limit=int(args.refs_limit))
    if not ref_paths:
        raise SystemExit(f"No reference reels found under: {refs_dir}")

    use_refs_library = bool(getattr(args, "footage_from_refs", False))
    # For fast, cache-friendly runs we use the downloaded refs folder directly as the
    # footage library (and exclude the current ref via FOLDER_EDIT_EXCLUDE_BASENAMES).
    # This keeps media_index + shot_index caches stable across repeated benchmarks.
    shared_refs_library: Path | None = refs_dir if use_refs_library else None

    # Media folder: optionally sample for speed (symlinks). When --footage-from-refs is enabled,
    # a per-case media folder is created inside each case_root.
    media_folder = Path(args.folder).expanduser().resolve()
    if not use_refs_library:
        if not media_folder.exists():
            raise SystemExit(f"Footage folder not found: {media_folder}")
        sample_n = int(args.footage_sample or 0)
        if sample_n > 0:
            sample_dir = out_root / "media_sample"
            media_folder, picked = _sample_media_folder(media_folder, dst=sample_dir, limit=sample_n)
            _ = picked

    # Shared env: pro defaults + caches.
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = api_key
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("FOLDER_EDIT_BEAT_SYNC", "1")
    env.setdefault("FOLDER_EDIT_STORY_PLANNER", "1")
    env.setdefault("SHOT_INDEX_MODE", "scene")
    env.setdefault("SHOT_INDEX_WORKERS", "4")
    env.setdefault("SHOT_TAGGING", "1")
    env.setdefault("SHOT_TAG_MAX", "250")
    env.setdefault("REF_SEGMENT_FRAME_COUNT", "3")
    env.setdefault("REASONING_EFFORT", "high")
    env.setdefault("VARIANT_PRO_MACRO", "beam")
    env.setdefault("VARIANT_FIX_ITERS", "1")
    try:
        mv = int(args.shot_index_max_videos or 0)
    except Exception:
        mv = 0
    if mv > 0:
        env.setdefault("SHOT_INDEX_MAX_VIDEOS", str(int(mv)))
    # Reuse global asset tag cache (huge cost saver).
    env.setdefault("FOLDER_EDIT_TAG_CACHE_PATH", str((repo_root / "Outputs" / "_asset_tag_cache.json").resolve()))
    env["PYTHONPATH"] = str((repo_root / "backend").resolve()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    criteria = (
        "Be extremely strict about stability, transitions, rhythm, and emotional arc. "
        "Stability is ABSOLUTE viewer comfort; do not excuse shake even if the reference is shaky. "
        "If stability <= 2 then overall must be <= 5."
    )

    runs: list[dict[str, t.Any]] = []
    for idx, ref in enumerate(ref_paths, start=1):
        code = ref.stem.split("_", 2)[1] if "_" in ref.stem else ref.stem
        case_root = out_root / f"{idx:02d}_{code}"
        case_root.mkdir(parents=True, exist_ok=True)

        seed = int(args.seed) + (idx * 101)

        def run_one(director: str) -> ScoredRun:
            proj = (case_root / f"director_{director}").resolve()
            # Per-case library selection (optional).
            media_folder_case = shared_refs_library if use_refs_library and shared_refs_library is not None else media_folder
            env_run = dict(env)
            if use_refs_library:
                # Prevent trivial "copy the reference" cheating while keeping cache keys stable.
                env_run["FOLDER_EDIT_EXCLUDE_BASENAMES"] = str(ref.name)
            if str(director) == "gemini":
                # Benchmarks must fail-fast if the director silently falls back to top picks.
                env_run.setdefault("VARIANT_GEMINI_STRICT", "1")
            cmd = [
                sys.executable,
                "-m",
                "reelclaw_pipeline.run_folder_edit_variants",
                "--pro",
                "--director",
                str(director),
                "--reel",
                str(ref),
                "--folder",
                str(media_folder_case),
                "--niche",
                "cars",
                "--vibe",
                "aesthetic luxury",
                "--model",
                str(args.analysis_model),
                "--variants",
                str(int(args.variants)),
                "--finals",
                str(int(args.finals)),
                "--seed",
                str(int(seed)),
                "--out",
                str(proj),
                "--timeout",
                str(float(args.timeout)),
            ]
            log_path = (case_root / "logs" / f"{director}.log").resolve()
            try:
                _run(cmd, cwd=repo_root, env=env_run, log_path=log_path, timeout_s=float(args.timeout) * max(1, int(args.variants)))
            except Exception as e:
                return ScoredRun(
                    director=director,
                    project_root=proj,
                    final_video=None,
                    reference_clip=None,
                    judge={"ok": False, "error": f"{type(e).__name__}: {e}", "log_path": str(log_path)},
                    ok=False,
                    error=f"{type(e).__name__}: {e}",
                    log_path=log_path,
                )

            final: Path | None = None
            ref_clip: Path | None = None
            try:
                final = _find_best_final(proj)
                ref_clip = _find_reference_clip(proj)
            except Exception as e:
                return ScoredRun(
                    director=director,
                    project_root=proj,
                    final_video=final,
                    reference_clip=ref_clip,
                    judge={"ok": False, "error": f"{type(e).__name__}: {e}", "log_path": str(log_path)},
                    ok=False,
                    error=f"{type(e).__name__}: {e}",
                    log_path=log_path,
                )

            judge: dict[str, t.Any]
            try:
                judge = _score_compare(
                    api_key=api_key,
                    judge_model=str(args.judge_model),
                    output_video=t.cast(Path, final),
                    reference_video=t.cast(Path, ref_clip),
                    max_mb=float(args.max_mb),
                    tmp_dir=(proj / "judge_tmp"),
                    criteria=criteria,
                )
            except Exception as e:
                judge = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            return ScoredRun(director=director, project_root=proj, final_video=final, reference_clip=ref_clip, judge=judge, ok=True, log_path=log_path)

        code_run = run_one("code")
        gem_run = run_one("gemini")

        # Thumbs (best-effort).
        thumbs_dir = case_root / "thumbs"
        if isinstance(code_run.reference_clip, Path) and code_run.reference_clip.exists():
            _extract_thumb(code_run.reference_clip, out_path=(thumbs_dir / "ref.jpg"), at_s=0.5)
        if isinstance(code_run.final_video, Path) and code_run.final_video.exists():
            _extract_thumb(code_run.final_video, out_path=(thumbs_dir / "code.jpg"), at_s=0.5)
        if isinstance(gem_run.final_video, Path) and gem_run.final_video.exists():
            _extract_thumb(gem_run.final_video, out_path=(thumbs_dir / "gemini.jpg"), at_s=0.5)

        runs.append(
            {
                "ref": str(ref),
                "case_root": str(case_root),
                "seed": int(seed),
                "code": {
                    "ok": bool(code_run.ok),
                    "error": code_run.error,
                    "log_path": str(code_run.log_path) if code_run.log_path is not None else None,
                    "project_root": str(code_run.project_root),
                    "final_video": str(code_run.final_video) if code_run.final_video is not None else None,
                    "reference_clip": str(code_run.reference_clip) if code_run.reference_clip is not None else None,
                    "judge": code_run.judge,
                },
                "gemini": {
                    "ok": bool(gem_run.ok),
                    "error": gem_run.error,
                    "log_path": str(gem_run.log_path) if gem_run.log_path is not None else None,
                    "project_root": str(gem_run.project_root),
                    "final_video": str(gem_run.final_video) if gem_run.final_video is not None else None,
                    "reference_clip": str(gem_run.reference_clip) if gem_run.reference_clip is not None else None,
                    "judge": gem_run.judge,
                },
                "thumbs": {
                    "ref": str((thumbs_dir / "ref.jpg").resolve()),
                    "code": str((thumbs_dir / "code.jpg").resolve()),
                    "gemini": str((thumbs_dir / "gemini.jpg").resolve()),
                },
            }
        )

    # Gate: Gemini director must not drop absolute overall or stability versus code director.
    gate_rows: list[dict[str, t.Any]] = []
    tol_overall = float(os.getenv("BENCH_GATE_OVERALL_TOL", "0.0") or 0.0)
    tol_stab = float(os.getenv("BENCH_GATE_STABILITY_TOL", "0.0") or 0.0)
    for r in runs:
        code_judge = ((r.get("code") or {}).get("judge") or {}) if isinstance(r.get("code"), dict) else {}
        gem_judge = ((r.get("gemini") or {}).get("judge") or {}) if isinstance(r.get("gemini"), dict) else {}
        code_res = code_judge.get("result") if isinstance(code_judge, dict) else None
        gem_res = gem_judge.get("result") if isinstance(gem_judge, dict) else None
        code_over = _safe_float(code_res.get("overall_score")) if isinstance(code_res, dict) else None
        gem_over = _safe_float(gem_res.get("overall_score")) if isinstance(gem_res, dict) else None
        code_stab0 = _safe_float(code_res.get("stability")) if isinstance(code_res, dict) else None
        gem_stab0 = _safe_float(gem_res.get("stability")) if isinstance(gem_res, dict) else None
        # Judge failures should fail the gate: otherwise we "pass" benchmarks with missing data.
        scores_present = (code_over is not None) and (gem_over is not None) and (code_stab0 is not None) and (gem_stab0 is not None)
        over_ok = bool(scores_present and (float(gem_over) + float(tol_overall) >= float(code_over)))
        stab_ok = bool(scores_present and (float(gem_stab0) + float(tol_stab) >= float(code_stab0)))
        gate_rows.append(
            {
                "ref": str(r.get("ref") or ""),
                "scores_present": bool(scores_present),
                "code_judge_ok": bool(code_judge.get("ok")) if isinstance(code_judge, dict) else False,
                "gemini_judge_ok": bool(gem_judge.get("ok")) if isinstance(gem_judge, dict) else False,
                "code_judge_error": code_judge.get("error") if isinstance(code_judge, dict) else None,
                "gemini_judge_error": gem_judge.get("error") if isinstance(gem_judge, dict) else None,
                "overall_ok": bool(over_ok),
                "stability_ok": bool(stab_ok),
                "delta_overall": (float(gem_over) - float(code_over)) if (code_over is not None and gem_over is not None) else None,
                "delta_stability": (float(gem_stab0) - float(code_stab0)) if (code_stab0 is not None and gem_stab0 is not None) else None,
            }
        )

    def _mean(xs: list[float]) -> float | None:
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    code_overall = [_safe_float(((r.get("code") or {}).get("judge") or {}).get("result", {}).get("overall_score")) for r in runs]
    gem_overall = [_safe_float(((r.get("gemini") or {}).get("judge") or {}).get("result", {}).get("overall_score")) for r in runs]
    code_stab = [_safe_float(((r.get("code") or {}).get("judge") or {}).get("result", {}).get("stability")) for r in runs]
    gem_stab = [_safe_float(((r.get("gemini") or {}).get("judge") or {}).get("result", {}).get("stability")) for r in runs]
    code_overall_f = [x for x in code_overall if isinstance(x, (int, float))]
    gem_overall_f = [x for x in gem_overall if isinstance(x, (int, float))]
    code_stab_f = [x for x in code_stab if isinstance(x, (int, float))]
    gem_stab_f = [x for x in gem_stab if isinstance(x, (int, float))]

    summary = {
        "created_at": _now_ts(),
        "refs_dir": str(refs_dir),
        "refs_used": [str(p) for p in ref_paths],
        "footage_from_refs": bool(getattr(args, "footage_from_refs", False)),
        "footage_folder": str(media_folder) if not bool(getattr(args, "footage_from_refs", False)) else str(refs_dir),
        "variants": int(args.variants),
        "finals": int(args.finals),
        "shot_index_max_videos": int(mv) if "mv" in locals() else None,
        "analysis_model": str(args.analysis_model),
        "judge_model": str(args.judge_model),
        "max_mb": float(args.max_mb),
        "means": {
            "code_overall": _mean(code_overall_f),
            "gemini_overall": _mean(gem_overall_f),
            "code_stability": _mean(code_stab_f),
            "gemini_stability": _mean(gem_stab_f),
        },
        "gate": {
            "rule": "gemini must not drop absolute overall_score or stability vs code (tolerances configurable via BENCH_GATE_*_TOL)",
            "tolerance": {"overall": float(tol_overall), "stability": float(tol_stab)},
            "per_ref": gate_rows,
            "pass_all": bool(all(bool(row.get("overall_ok")) and bool(row.get("stability_ok")) for row in gate_rows)),
        },
        "runs": runs,
    }
    out_path = out_root / "bench_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

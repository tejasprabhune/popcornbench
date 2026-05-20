"""Publish PopcornBench traces and the leaderboard to gh-pages.

The publisher does three jobs on each invocation. First, it walks
``traces/`` for every ``*.jsonl`` trace (single-run files at the top
level plus per-cell traces under ``traces/<sweep>/<cell_slug>/``) and
materialises a per-run viewer under ``<run_slug>/viewer.html`` on the
gh-pages branch, copying the ensemble static viewer next to each
trace so the polling viewer works offline. Second, it parses every
published trace into a summary record (timestamp, scenario, model,
persona, level, problem, outcome, runtime, speedup, gpu_seconds) and
writes ``runs.json`` at the gh-pages root, which is what the
leaderboard and the run index page fetch on load. Third, it copies
the top-level site assets (``style.css``, ``index.html``,
``runs.html``) from ``site/`` to the gh-pages root.

The old gh-pages tree is wiped before publishing so the site reflects
exactly what the current ``traces/`` and ``site/`` directories say.
Earlier versions of this script kept the pre-existing gh-pages
content (a hand-authored experiments site) underneath the new
viewer; the rewrite drops that content. If it is wanted again, pull
it from git history on the gh-pages branch.

Usage
-----
    # one-shot
    uv run python scripts/publish_traces.py --ensemble-root ~/Documents/ensemble

    # loop every five minutes from inside tmux
    uv run python scripts/publish_traces.py \\
        --ensemble-root ~/Documents/ensemble --watch 300

Summary-record extraction
-------------------------
For each trace the script extracts:

- ``timestamp``: first event's ``ts_ms`` converted to UTC ISO 8601.
- ``scenario``: parsed from a ``grader:`` system event when present;
  otherwise taken from the trace file stem.
- ``model`` and ``persona``: read from a co-located ``runs.jsonl``
  (the sweep runner records the matrix point per cell); ``null`` for
  ad-hoc traces with no co-located record.
- ``level``, ``problem_id``, ``problem_name``: parsed from the
  ``fetch_problem`` tool_call args and tool_result effect.
- ``runtime_us``, ``ref_runtime_us``, ``speedup``: parsed from the
  most recent ``state_diff`` whose ``field == "kernel_submissions"``
  (``submit_kernel`` populates this).
- ``outcome``: derived from the grader scores. ``passed`` when
  ``submission_passed >= 1`` (or ``submit_passed`` for TOML
  scenarios), ``incomplete`` when the agent never submitted,
  ``failed`` otherwise.
- ``cost_gpu_seconds``: sum of every event's ``costs.gpu_seconds``.
- ``viewer_path``: ``<run_slug>/viewer.html`` relative to gh-pages
  root.

GitHub Pages side (one-time)
----------------------------
    Settings -> Pages -> Source: Deploy from a branch
                         Branch: gh-pages, Folder: / (root)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_TOP = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_TOP / "traces"
SITE_DIR = REPO_TOP / "site"


def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _git(args: List[str], cwd: Path = REPO_TOP, check: bool = True) -> str:
    proc = _run(["git", *args], cwd=cwd, check=check)
    return proc.stdout.strip()


def _ensure_gh_pages_branch_exists() -> None:
    branches = _git(["branch", "-a"]).splitlines()
    flat = [b.strip().lstrip("* ").replace("remotes/origin/", "") for b in branches]
    if "gh-pages" in flat:
        return
    print("creating gh-pages orphan branch", file=sys.stderr)
    _git(["checkout", "--orphan", "gh-pages"])
    _git(["rm", "-rf", "--quiet", "."], check=False)
    seed = REPO_TOP / "README.md"
    seed.write_text("# PopcornBench leaderboard\n\nPublished by scripts/publish_traces.py.\n")
    _git(["add", "README.md"])
    _git(["commit", "-m", "seed gh-pages branch"])
    _git(["checkout", "-"])


def _worktree(scratch: Path, *, fetch_remote: bool = True) -> Path:
    worktree = scratch / "gh-pages-worktree"
    if worktree.exists():
        _git(["worktree", "remove", "--force", str(worktree)], check=False)
        shutil.rmtree(worktree, ignore_errors=True)
    if fetch_remote:
        _git(["fetch", "origin", "gh-pages"], check=False)
        _git(["worktree", "add", "-B", "gh-pages", str(worktree), "origin/gh-pages"], check=False)
    if not worktree.exists():
        # Fall back to a local branch (creates an empty one when no
        # remote was fetched). The wipe step then leaves an empty
        # worktree that gets populated from site/ and traces/ below.
        _git(["worktree", "add", "-b", "gh-pages-local", str(worktree)], check=False)
        if not worktree.exists():
            _git(["worktree", "add", str(worktree)], check=False)
    return worktree


def _wipe_worktree(worktree: Path) -> None:
    """Empty the gh-pages worktree (preserving .git) so the next publish
    is the full state of site/ + traces/, not an accumulation."""
    for entry in worktree.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink()


def _copy_ensemble_viewer(ensemble_root: Path, dest: Path) -> None:
    src = ensemble_root / "site"
    if not src.exists():
        raise FileNotFoundError(
            f"could not find ensemble site at {src}; pass --ensemble-root"
        )
    dest.mkdir(parents=True, exist_ok=True)
    # Only copy the per-run viewer assets, not the ensemble landing
    # page (we have our own at gh-pages root).
    wanted = {"viewer.html", "viewer.js", "style.css"}
    for entry in src.iterdir():
        if entry.name in wanted:
            shutil.copy2(entry, dest / entry.name)


def _copy_local_site(dest: Path) -> None:
    """Copy site/ (top-level leaderboard + run index) to gh-pages root."""
    if not SITE_DIR.exists():
        raise FileNotFoundError(f"local site dir not found at {SITE_DIR}")
    for entry in SITE_DIR.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest / entry.name)


def _all_traces() -> List[Path]:
    if not TRACES_DIR.exists():
        return []
    return sorted(TRACES_DIR.rglob("*.jsonl"))


def _slug_for(trace: Path) -> str:
    """The per-run viewer slug. Top-level traces use the file stem;
    nested traces (sweep cells) flatten directory path with hyphens."""
    rel = trace.relative_to(TRACES_DIR)
    parts = list(rel.with_suffix("").parts)
    # Sweep cells live at <sweep>/<cell_slug>/<scenario>.jsonl. Drop
    # the trailing scenario component because it is redundant with the
    # cell slug; we keep <sweep>__<cell_slug>.
    if len(parts) >= 3 and parts[-1] in {"popcorn.single_problem", "popcorn.judge_review"}:
        parts = parts[:-1]
    return "__".join(parts)


@dataclass
class Summary:
    slug: str
    trace_relpath: str
    timestamp: Optional[str] = None
    scenario: Optional[str] = None
    model: Optional[str] = None
    persona: Optional[str] = None
    level: Optional[int] = None
    problem_id: Optional[int] = None
    problem_name: Optional[str] = None
    outcome: str = "incomplete"
    runtime_us: Optional[float] = None
    ref_runtime_us: Optional[float] = None
    speedup: Optional[float] = None
    cost_gpu_seconds: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    viewer_path: str = ""


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _load_runs_jsonl_index(trace: Path) -> Optional[Dict[str, Any]]:
    """For sweep cells, the runner writes traces/<sweep>/runs.jsonl
    with one entry per cell. Look up this trace's entry by trace_path
    match. Returns the matching record or None."""
    sweep_dir = trace.parent.parent
    runs_jsonl = sweep_dir / "runs.jsonl"
    if not runs_jsonl.exists():
        return None
    trace_str = str(trace)
    last = None
    for rec in _read_jsonl(runs_jsonl):
        if rec.get("trace_path") == trace_str or rec.get("trace_path", "").endswith(str(trace.relative_to(REPO_TOP))):
            last = rec  # latest record wins (resume appends new ones)
    return last


_GRADER_NOTE = re.compile(r"^grader:\s*(\{.*\})$")


def _parse_trace(trace: Path) -> Summary:
    summary = Summary(slug=_slug_for(trace), trace_relpath=str(trace.relative_to(REPO_TOP)))
    last_submission: Optional[Dict[str, Any]] = None
    grader_payload: Optional[Dict[str, Any]] = None
    first_ts_ms: Optional[int] = None

    for ev in _read_jsonl(trace):
        ts_ms = ev.get("ts_ms")
        if first_ts_ms is None and isinstance(ts_ms, int):
            first_ts_ms = ts_ms
        payload = ev.get("payload") or {}
        kind = payload.get("kind")

        if kind == "tool_call" and payload.get("name") == "fetch_problem":
            args = payload.get("args") or {}
            if isinstance(args.get("level"), int):
                summary.level = args["level"]
            if isinstance(args.get("problem_id"), int):
                summary.problem_id = args["problem_id"]
        elif kind == "tool_result" and payload.get("name") == "fetch_problem":
            result = payload.get("result") or {}
            effect = result.get("effect") if isinstance(result.get("effect"), dict) else result
            if isinstance(effect, dict):
                if isinstance(effect.get("name"), str):
                    summary.problem_name = effect["name"]
                if isinstance(effect.get("level"), int) and summary.level is None:
                    summary.level = effect["level"]
                if isinstance(effect.get("problem_id"), int) and summary.problem_id is None:
                    summary.problem_id = effect["problem_id"]
        elif kind == "state_diff":
            diffs = payload.get("diff") or []
            if isinstance(diffs, list):
                items = diffs
            else:
                items = [diffs]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("field") == "kernel_submissions" and isinstance(item.get("new"), dict):
                    last_submission = item["new"]
                if item.get("field") == "problem" and isinstance(item.get("new"), dict):
                    new = item["new"]
                    if summary.level is None and isinstance(new.get("level"), int):
                        summary.level = new["level"]
                    if summary.problem_id is None and isinstance(new.get("problem_id"), int):
                        summary.problem_id = new["problem_id"]
                    if summary.problem_name is None and isinstance(new.get("name"), str):
                        summary.problem_name = new["name"]
        elif kind == "system":
            note = payload.get("note") or ""
            m = _GRADER_NOTE.match(note)
            if m:
                try:
                    grader_payload = json.loads(m.group(1))
                except json.JSONDecodeError:
                    grader_payload = None

        costs = ev.get("costs") or payload.get("costs") or {}
        if isinstance(costs, dict):
            gpu = costs.get("gpu_seconds")
            if isinstance(gpu, (int, float)):
                summary.cost_gpu_seconds += float(gpu)

    if first_ts_ms is not None:
        summary.timestamp = _dt.datetime.utcfromtimestamp(first_ts_ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        mtime = trace.stat().st_mtime
        summary.timestamp = _dt.datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%SZ")

    if last_submission:
        rt = last_submission.get("runtime_us")
        if isinstance(rt, (int, float)):
            summary.runtime_us = float(rt)
        ref = last_submission.get("ref_runtime_us")
        if isinstance(ref, (int, float)):
            summary.ref_runtime_us = float(ref)
        sp = last_submission.get("speedup")
        if isinstance(sp, (int, float)):
            summary.speedup = float(sp)
        elif summary.runtime_us and summary.ref_runtime_us:
            summary.speedup = round(summary.ref_runtime_us / summary.runtime_us, 4)

    if grader_payload:
        if isinstance(grader_payload.get("scenario"), str):
            summary.scenario = grader_payload["scenario"]
        scores = grader_payload.get("scores") or {}
        if isinstance(scores, dict):
            summary.scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}

    if summary.scenario is None:
        summary.scenario = trace.stem

    # Outcome from grader scores; the scenarios expose both naming
    # conventions (`submission_passed` from python scenarios,
    # `submit_passed` from the TOML grader expression `correct =
    # "submit_passed"`).
    passed = summary.scores.get("submission_passed") or summary.scores.get("submit_passed") or summary.scores.get("correct")
    submitted = summary.scores.get("submitted")
    if passed is not None and passed >= 1.0:
        summary.outcome = "passed"
    elif submitted is not None and submitted < 1.0:
        summary.outcome = "incomplete"
    elif passed is not None:
        summary.outcome = "failed"
    elif last_submission and last_submission.get("correctness"):
        summary.outcome = "passed"
    elif last_submission:
        summary.outcome = "failed"

    # Sweep runner records the matrix point in runs.jsonl. Prefer that
    # over best-effort trace parsing for fields it knows authoritatively.
    runs_rec = _load_runs_jsonl_index(trace)
    if runs_rec:
        if isinstance(runs_rec.get("model"), str):
            summary.model = runs_rec["model"]
        if isinstance(runs_rec.get("persona"), str):
            summary.persona = runs_rec["persona"]
        if isinstance(runs_rec.get("level"), int):
            summary.level = runs_rec["level"]
        if isinstance(runs_rec.get("problem_id"), int):
            summary.problem_id = runs_rec["problem_id"]
        if isinstance(runs_rec.get("scenario"), str):
            summary.scenario = runs_rec["scenario"]
        if isinstance(runs_rec.get("scores"), dict):
            for k, v in runs_rec["scores"].items():
                if isinstance(v, (int, float)):
                    summary.scores.setdefault(k, float(v))

    summary.viewer_path = f"{summary.slug}/viewer.html"
    return summary


def _summarize_all() -> List[Summary]:
    summaries: List[Summary] = []
    for trace in _all_traces():
        try:
            summaries.append(_parse_trace(trace))
        except Exception as e:
            print(f"parse failed for {trace}: {e}", file=sys.stderr)
    # Most recent first.
    summaries.sort(key=lambda s: s.timestamp or "", reverse=True)
    return summaries


def _write_runs_json(worktree: Path, summaries: List[Summary]) -> None:
    payload = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": [
            {
                "slug": s.slug,
                "timestamp": s.timestamp,
                "scenario": s.scenario,
                "model": s.model,
                "persona": s.persona,
                "level": s.level,
                "problem_id": s.problem_id,
                "problem_name": s.problem_name,
                "outcome": s.outcome,
                "runtime_us": s.runtime_us,
                "ref_runtime_us": s.ref_runtime_us,
                "speedup": s.speedup,
                "cost_gpu_seconds": round(s.cost_gpu_seconds, 3) if s.cost_gpu_seconds else 0.0,
                "scores": s.scores,
                "viewer_path": s.viewer_path,
            }
            for s in summaries
        ],
    }
    (worktree / "runs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def publish(ensemble_root: Path, scratch: Path, dry_run: bool = False) -> None:
    _ensure_gh_pages_branch_exists()
    # Dry-run does not need the remote gh-pages tree; skipping the
    # fetch keeps the verification fast and works offline.
    worktree = _worktree(scratch, fetch_remote=not dry_run)
    _wipe_worktree(worktree)

    _copy_local_site(worktree)

    summaries = _summarize_all()
    for trace in _all_traces():
        slug = _slug_for(trace)
        target = worktree / slug
        target.mkdir(parents=True, exist_ok=True)
        _copy_ensemble_viewer(ensemble_root, target)
        shutil.copy2(trace, target / "trace.jsonl")

    _write_runs_json(worktree, summaries)

    if dry_run:
        print(f"dry-run: built worktree at {worktree} with {len(summaries)} runs", file=sys.stderr)
        return

    _git(["add", "."], cwd=worktree)
    diff = _git(["status", "--porcelain"], cwd=worktree)
    if not diff:
        print("nothing to publish", file=sys.stderr)
        return
    stamp = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _git(["commit", "-m", f"publish {len(summaries)} runs {stamp}"], cwd=worktree)
    push = _run(["git", "push", "origin", "gh-pages"], cwd=worktree, check=False)
    if push.returncode != 0:
        print(push.stderr, file=sys.stderr)
        print("publish: push failed (see stderr above)", file=sys.stderr)
        return
    print(f"published {len(summaries)} runs to gh-pages", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="publish_traces")
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        default=os.environ.get("ENSEMBLE_ROOT", str(Path.home() / "Documents" / "ensemble")),
        help="Path to the ensemble checkout that holds site/.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=REPO_TOP.parent / ".popcorn-publish",
        help="Scratch directory for the gh-pages worktree.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, repeat the publish every N seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the worktree and runs.json but skip the commit and push.",
    )
    args = parser.parse_args(argv)

    ensemble_root = Path(args.ensemble_root).expanduser().resolve()
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            publish(ensemble_root, scratch, dry_run=args.dry_run)
        except Exception as e:
            print(f"publish failed: {e}", file=sys.stderr)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())

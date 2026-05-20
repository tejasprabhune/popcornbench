"""Publish ensemble traces to the repo's gh-pages branch.

The new ensemble-driven setup writes traces to ``traces/<scenario>.jsonl``.
The viewer lives in the ensemble checkout at ``site/``: static HTML +
``viewer.js`` polling ``trace.jsonl`` next to it. To make traces
readable through GitHub Pages, we copy that ``site/`` directory into a
worktree on the ``gh-pages`` branch, drop each trace under
``<run-id>/trace.jsonl`` next to a copy of the viewer, and rewrite
the top-level ``index.html`` to link to every run.

Use one-shot for a single push or ``--watch SECONDS`` to loop while a
sweep is running. The worktree lives outside the working tree so it
does not disturb your checkout.

Examples
--------
    # one-shot: publish everything currently under traces/
    uv run python scripts/publish_traces.py \\
        --ensemble-root ~/Documents/ensemble

    # loop every 5 minutes from inside tmux
    uv run python scripts/publish_traces.py \\
        --ensemble-root ~/Documents/ensemble --watch 300

GitHub Pages side (one-time):
    Settings -> Pages -> Source: Deploy from a branch
                         Branch: gh-pages, Folder: / (root)
    URL:   https://<user>.github.io/<repo>/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_TOP = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _git(args: list[str], cwd: Path = REPO_TOP, check: bool = True) -> str:
    proc = _run(["git", *args], cwd=cwd, check=check)
    return proc.stdout.strip()


def _ensure_gh_pages_branch_exists() -> None:
    """Create the gh-pages branch if it does not exist (orphan)."""
    branches = _git(["branch", "-a"]).splitlines()
    flat = [b.strip().lstrip("* ").replace("remotes/origin/", "") for b in branches]
    if "gh-pages" in flat:
        return
    # Create a brand-new orphan branch with an empty README.
    print("creating gh-pages orphan branch", file=sys.stderr)
    _git(["checkout", "--orphan", "gh-pages"])
    _git(["rm", "-rf", "--quiet", "."], check=False)
    readme = REPO_TOP / "README.md"
    readme.write_text("# ensemble traces\n\nPublished by `scripts/publish_traces.py`.\n")
    _git(["add", "README.md"])
    _git(["commit", "-m", "seed gh-pages branch"])
    _git(["checkout", "-"])


def _worktree(scratch: Path) -> Path:
    """Materialize a worktree of the gh-pages branch outside the checkout."""
    worktree = scratch / "gh-pages-worktree"
    if worktree.exists():
        _git(["worktree", "remove", "--force", str(worktree)], check=False)
        shutil.rmtree(worktree, ignore_errors=True)
    _git(["fetch", "origin", "gh-pages"], check=False)
    _git(["worktree", "add", "-B", "gh-pages", str(worktree), "origin/gh-pages"], check=False)
    if not worktree.exists():
        # Fallback: branch never existed remotely.
        _git(["worktree", "add", "-b", "gh-pages", str(worktree)])
    return worktree


def _copy_site(ensemble_root: Path, dest: Path) -> None:
    """Copy ensemble's static viewer into dest."""
    src = ensemble_root / "site"
    if not src.exists():
        raise FileNotFoundError(
            f"could not find ensemble site at {src}; pass --ensemble-root"
        )
    dest.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest / entry.name)


def _runs() -> list[Path]:
    traces = REPO_TOP / "traces"
    if not traces.exists():
        return []
    return sorted([p for p in traces.glob("*.jsonl")])


def _build_index(worktree: Path, runs: list[tuple[str, str]]) -> None:
    """Write a top-level index.html linking to each published run."""
    rows = []
    for slug, label in runs:
        rows.append(
            f'<li><a href="{_html.escape(slug)}/viewer.html">{_html.escape(label)}</a></li>'
        )
    rendered_at = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    body = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>ensemble traces</title>"
        "<link rel='stylesheet' href='style.css'>"
        "<main><h1>ensemble traces</h1>"
        f"<p>updated {rendered_at}</p>"
        f"<ul>{''.join(rows) or '<li>(no runs yet)</li>'}</ul>"
        "</main>"
    )
    (worktree / "index.html").write_text(body, encoding="utf-8")


def publish(ensemble_root: Path, scratch: Path) -> None:
    _ensure_gh_pages_branch_exists()
    worktree = _worktree(scratch)

    # Copy the viewer assets at the top level so the index page has
    # styling; also drop them next to each trace so deep links work.
    _copy_site(ensemble_root, worktree)

    runs: list[tuple[str, str]] = []
    for trace in _runs():
        slug = trace.stem
        target = worktree / slug
        target.mkdir(parents=True, exist_ok=True)
        _copy_site(ensemble_root, target)
        shutil.copy2(trace, target / "trace.jsonl")
        runs.append((slug, slug))

    _build_index(worktree, runs)

    _git(["add", "."], cwd=worktree)
    diff = _git(["status", "--porcelain"], cwd=worktree)
    if not diff:
        print("nothing to publish", file=sys.stderr)
        return
    _git(["commit", "-m", f"publish traces {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z"], cwd=worktree)
    push = _run(["git", "push", "origin", "gh-pages"], cwd=worktree, check=False)
    if push.returncode != 0:
        print(push.stderr, file=sys.stderr)
        print("publish: push failed (see stderr above)", file=sys.stderr)
        return
    print(f"published {len(runs)} runs to gh-pages", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    ensemble_root = Path(args.ensemble_root).expanduser().resolve()
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            publish(ensemble_root, scratch)
        except Exception as e:
            print(f"publish failed: {e}", file=sys.stderr)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())

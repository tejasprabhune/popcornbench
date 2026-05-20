"""Run a (level, problem_id, model, persona, seed) sweep through ensemble.

Usage:
    uv run python scripts/run_sweep.py --config path/to/sweep.toml

The sweep config declares a matrix and a scenario. The runner expands
the matrix into one cell per tuple, writes a `cell.toml` with the
exact env vars for that cell, runs `ensemble run` per cell, and lands
the trace JSONL under `<results_root>/<sweep.name>/<cell_slug>/`.

Resume: when a cell's trace file already exists and is non-empty, the
runner skips it. Pass `--no-resume` (or set `[sweep.run].resume = false`
in the config) to force a re-run.

Concurrency: serial by default. Pass `--concurrency N` (or set
`[sweep.run].concurrency = N` in the config) to dispatch N cells in
parallel. The ensemble runtime serialises GPU tools via resource
locks, so concurrency above 1 only helps when cells target different
devices (vary `POPCORN_DEVICE_INDEX` in the matrix) or when many cells
spend most of their wall clock in non-GPU work.

The runner emits a `runs.jsonl` per sweep at
`<results_root>/<sweep.name>/runs.jsonl`, one line per completed
cell, recording the matrix point and the cell's grader scores. This
is what the leaderboard's manifest builder consumes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


REPO_TOP = Path(__file__).resolve().parent.parent


@dataclass
class SweepConfig:
    name: str
    scenario: str
    results_root: Path
    matrix: Dict[str, List]
    max_turns: int = 20
    gpu_seconds: float = 600.0
    concurrency: int = 1
    resume: bool = True
    extra_env: Dict[str, str] = field(default_factory=dict)


def load_config(path: Path) -> SweepConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    sweep = raw.get("sweep")
    if not sweep:
        raise ValueError(f"{path}: missing [sweep] section")
    matrix = sweep.get("matrix") or {}
    if not matrix:
        raise ValueError(f"{path}: missing [sweep.matrix] section")
    budget = sweep.get("budget") or {}
    run = sweep.get("run") or {}
    name = sweep.get("name")
    if not name or not re.match(r"^[A-Za-z0-9_.-]+$", name):
        raise ValueError(
            f"{path}: sweep.name must be set and contain only [A-Za-z0-9_.-]"
        )
    return SweepConfig(
        name=name,
        scenario=sweep.get("scenario", "popcorn.single_problem"),
        results_root=Path(sweep.get("results_root", "traces")),
        matrix={k: list(v) for k, v in matrix.items()},
        max_turns=int(budget.get("max_turns", 20)),
        gpu_seconds=float(budget.get("gpu_seconds", 600.0)),
        concurrency=int(run.get("concurrency", 1)),
        resume=bool(run.get("resume", True)),
        extra_env=dict(run.get("extra_env") or {}),
    )


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def _persona_slug(persona: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", persona)


@dataclass(frozen=True)
class Cell:
    level: int
    problem_id: int
    model: str
    persona: str
    seed: Optional[int]

    @property
    def slug(self) -> str:
        seed_part = f"_s{self.seed}" if self.seed is not None else ""
        return (
            f"l{self.level}_p{self.problem_id}_"
            f"{_model_slug(self.model)}_{_persona_slug(self.persona)}"
            f"{seed_part}"
        )

    def env(self, scenario: str, gpu_seconds: float, max_turns: int) -> Dict[str, str]:
        env = {
            "POPCORN_LEVEL": str(self.level),
            "POPCORN_PROBLEM_ID": str(self.problem_id),
            "POPCORN_MAX_TURNS": str(max_turns),
            "POPCORN_GPU_BUDGET": str(gpu_seconds),
        }
        # judge_review uses POPCORN_AUTHOR_MODEL; single_problem uses
        # POPCORN_AGENT_MODEL. Set both so the same matrix entry maps
        # to either scenario without per-scenario branches.
        env["POPCORN_AGENT_MODEL"] = self.model
        env["POPCORN_AUTHOR_MODEL"] = self.model
        env["POPCORN_PERSONA"] = self.persona
        env["POPCORN_AUTHOR_PERSONA"] = self.persona
        if self.seed is not None:
            env["POPCORN_HELD_OUT_SEED"] = str(self.seed)
        return env


def _expand_matrix(cfg: SweepConfig) -> List[Cell]:
    levels = cfg.matrix.get("levels") or [1]
    problem_ids = cfg.matrix.get("problem_ids") or [19]
    models = cfg.matrix.get("models") or ["claude-sonnet-4-5"]
    personas = cfg.matrix.get("personas") or ["normal"]
    seeds = cfg.matrix.get("seeds")
    if not seeds:
        # No seed dimension: emit cells with seed=None (held-out off).
        seeds_iter = [None]
    else:
        seeds_iter = list(seeds)
    cells: List[Cell] = []
    for L in levels:
        for P in problem_ids:
            for M in models:
                for persona in personas:
                    for s in seeds_iter:
                        cells.append(Cell(
                            level=int(L), problem_id=int(P),
                            model=str(M), persona=str(persona),
                            seed=int(s) if s is not None else None,
                        ))
    return cells


def _safe_scenario_name(scenario: str) -> str:
    # ensemble.cli_run computes the same shape; mirror it.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", scenario)


def _trace_path_for(cell_dir: Path, scenario: str) -> Path:
    return cell_dir / f"{_safe_scenario_name(scenario)}.jsonl"


def _run_cell(cfg: SweepConfig, cell: Cell, sweep_dir: Path) -> Dict:
    cell_dir = sweep_dir / cell.slug
    cell_dir.mkdir(parents=True, exist_ok=True)
    trace_path = _trace_path_for(cell_dir, cfg.scenario)

    if cfg.resume and trace_path.exists() and trace_path.stat().st_size > 0:
        return {
            "slug": cell.slug,
            "status": "skipped",
            "reason": "trace exists; resume on",
            "trace_path": str(trace_path),
        }

    env = os.environ.copy()
    env.update(cfg.extra_env)
    env.update(cell.env(cfg.scenario, cfg.gpu_seconds, cfg.max_turns))

    cmd = [
        "ensemble", "run", cfg.scenario,
        "--world", "popcorn",
        "--traces-dir", str(cell_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    # ensemble run prints one JSON line on stdout with scores + trace_path.
    scores: Dict = {}
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            scores = json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass

    return {
        "slug": cell.slug,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "trace_path": str(trace_path),
        "level": cell.level,
        "problem_id": cell.problem_id,
        "model": cell.model,
        "persona": cell.persona,
        "seed": cell.seed,
        "scenario": cfg.scenario,
        "scores": scores.get("scores") if isinstance(scores, dict) else None,
        "stderr_tail": proc.stderr.splitlines()[-10:] if proc.returncode != 0 else None,
    }


def _append_run_record(runs_jsonl: Path, record: Dict) -> None:
    runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with runs_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_sweep(cfg: SweepConfig) -> int:
    cells = _expand_matrix(cfg)
    sweep_dir = (REPO_TOP / cfg.results_root / cfg.name).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs_jsonl = sweep_dir / "runs.jsonl"

    print(f"sweep {cfg.name}: {len(cells)} cells, concurrency={cfg.concurrency}")
    print(f"  scenario: {cfg.scenario}")
    print(f"  results:  {sweep_dir.relative_to(REPO_TOP)}")

    failed = 0
    if cfg.concurrency <= 1:
        for cell in cells:
            rec = _run_cell(cfg, cell, sweep_dir)
            _append_run_record(runs_jsonl, rec)
            print(f"  [{rec['status']}] {rec['slug']} ({rec.get('elapsed_s', 0)}s)")
            if rec["status"] == "failed":
                failed += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            futures = {ex.submit(_run_cell, cfg, c, sweep_dir): c for c in cells}
            for fut in concurrent.futures.as_completed(futures):
                rec = fut.result()
                _append_run_record(runs_jsonl, rec)
                print(f"  [{rec['status']}] {rec['slug']} ({rec.get('elapsed_s', 0)}s)")
                if rec["status"] == "failed":
                    failed += 1

    print(f"done. {len(cells) - failed} ok, {failed} failed.")
    return 0 if failed == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_sweep")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to a sweep.toml.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run cells even when a trace already exists.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override [sweep.run].concurrency.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_resume:
        cfg.resume = False
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency

    return run_sweep(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

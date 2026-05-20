# Sweep configuration

A sweep is a matrix of `(level, problem_id, model, persona, seed)` tuples that get expanded into per-cell scenario runs and aggregated into the leaderboard. This page documents the sweep configuration format, the runner that consumes it, where results land on disk, the resume semantics, and how concurrency is controlled.

Before this page existed, sweeps were a copy-paste exercise: one `[scenario.<name>]` entry per cell in `popcorn_world/scenarios.toml`. That format still works for hand-written cells, but it does not scale to a matrix of more than a few entries. The sweep runner consumes a much smaller TOML that declares the matrix dimensions, and generates the per-cell invocations for you.

## File format

A sweep config lives in any TOML file (commonly named `sweep.toml`, `sweep.smoke.toml`, or `sweep.full.toml`). The example checked in at `sweep.example.toml` is a working starting point; the schema is documented below.

The top-level table is `[sweep]`:

- `name` (string, required): identifier used as the results subdirectory. Must match `[A-Za-z0-9_.-]+`.
- `scenario` (string, default `"popcorn.single_problem"`): the registered scenario each cell runs. Either Python (`popcorn.single_problem`, `popcorn.judge_review`) or a TOML scenario from a manifest (currently not auto-loaded by the runner; pass `--manifest` to ensemble through a wrapper if you need this).
- `results_root` (string, default `"traces"`): directory under the repo root that holds the sweep's results subtree.

`[sweep.matrix]` declares the matrix dimensions. Each field is a list; the runner takes the Cartesian product.

- `levels` (list of int, default `[1]`): KernelBench levels (1 to 5).
- `problem_ids` (list of int, default `[19]`): problem ids within each level. The runner applies every problem id to every level; entries that name a missing `(level, problem_id)` pair cause the cell's `ensemble run` to fail at `fetch_problem`.
- `models` (list of string, default `["claude-sonnet-4-5"]`): model identifiers passed straight to ensemble's backend selector.
- `personas` (list of string, default `["normal"]`): persona names that resolve through the world's `personas_dir`.
- `seeds` (list of int, optional): held-out seeds. When omitted, cells run without held-out re-verification and the `held_out_correctness_*` predicates always read false. When set to a single value, every cell uses the same held-out seed; pass multiple values to replicate each `(level, problem_id, model, persona)` cell with different held-out seeds.

`[sweep.budget]` controls per-cell limits. Both fields propagate into the cell's env so the scenario reads them just like a normal manual invocation.

- `max_turns` (int, default `20`): turn budget passed as `POPCORN_MAX_TURNS`.
- `gpu_seconds` (float, default `600.0`): GPU-seconds cap passed as `POPCORN_GPU_BUDGET`.

`[sweep.run]` controls how the runner dispatches cells.

- `concurrency` (int, default `1`): number of cells to dispatch in parallel. The ensemble runtime serialises GPU tools via resource locks, so going above 1 only helps when separate cells target different devices (vary `POPCORN_DEVICE_INDEX` in `[sweep.run.extra_env]`, but note the current matrix dimensions do not include device index, so you would partition by hand) or when many cells are not GPU-bound.
- `resume` (bool, default `true`): when true, cells whose trace file already exists and is non-empty are skipped. Pass `--no-resume` on the runner CLI to force a re-run.

`[sweep.run.extra_env]` is an inline table of `KEY = "value"` entries propagated to every cell. Use this for cluster-wide settings like `POPCORN_BACKEND`, `POPCORN_PRECISION`, `POPCORN_GPU_ARCH`, or `POPCORN_BUILD_DIR` that should be the same across every cell but you do not want to set in your shell.

## Worked example

```toml
# sweep.example.toml
[sweep]
name = "smoke_l1"
scenario = "popcorn.single_problem"
results_root = "traces"

[sweep.matrix]
levels = [1]
problem_ids = [1, 5, 19]
models = ["claude-haiku-4-5", "claude-sonnet-4-5"]
personas = ["normal", "methodical_engineer"]
seeds = [42]

[sweep.budget]
max_turns = 20
gpu_seconds = 600

[sweep.run]
concurrency = 1
resume = true

[sweep.run.extra_env]
POPCORN_BACKEND = "cuda"
POPCORN_PRECISION = "fp32"
```

This config expands to `1 * 3 * 2 * 2 * 1 = 12` cells. Each cell runs `ensemble run popcorn.single_problem --world popcorn --traces-dir traces/smoke_l1/<cell_slug>/` with the matrix point baked into the env (`POPCORN_LEVEL`, `POPCORN_PROBLEM_ID`, `POPCORN_AGENT_MODEL`, `POPCORN_PERSONA`, `POPCORN_HELD_OUT_SEED`).

## Cell slug

The runner computes a stable slug per cell:

```
l<level>_p<problem_id>_<model_slug>_<persona_slug>[_s<seed>]
```

Model and persona names are sanitised to `[A-Za-z0-9._-]+` (any other character becomes `-`). The `_s<seed>` suffix is omitted when the matrix declares no `seeds`. Example: a cell for level 1, problem 19, `claude-sonnet-4-5`, `methodical_engineer`, seed 42 lands at `l1_p19_claude-sonnet-4-5_methodical_engineer_s42`.

## Where results land

Per cell:

- `traces/<sweep.name>/<cell_slug>/<safe_scenario_name>.jsonl` is the trace JSONL the cell's `ensemble run` writes. `<safe_scenario_name>` is the scenario name with non-`[A-Za-z0-9._-]` characters replaced by `_`, so `popcorn.single_problem` becomes `popcorn.single_problem.jsonl`.

Per sweep:

- `traces/<sweep.name>/runs.jsonl` is the runner's per-cell record file. One JSON line per completed cell, recording `slug`, `status` (`ok` / `failed` / `skipped`), `returncode`, `elapsed_s`, `trace_path`, the full matrix point (`level`, `problem_id`, `model`, `persona`, `seed`), the `scenario`, the cell's grader `scores` parsed from `ensemble run`'s stdout, and a `stderr_tail` (last ten lines) when the cell failed.

The `scripts/publish_traces.py` leaderboard manifest builder reads `runs.jsonl` (and the per-cell trace files) when assembling the published leaderboard.

## Runner command

```
uv run python scripts/run_sweep.py --config sweep.toml [--no-resume] [--concurrency N]
```

Flags:

- `--config PATH` (required): path to the sweep TOML.
- `--no-resume`: re-run every cell even when the trace file already exists. Equivalent to setting `[sweep.run].resume = false` in the config.
- `--concurrency N`: override `[sweep.run].concurrency` for this invocation only.

The runner exits 0 when every cell either succeeded or was skipped due to resume. It exits 1 when at least one cell returned a non-zero status. The per-cell records in `runs.jsonl` always reflect the actual outcome regardless of exit code, so a downstream consumer never loses information about a failed cell.

## Resume semantics

A cell is skipped when:

1. `[sweep.run].resume` is true (the default).
2. The cell's trace file (`traces/<sweep.name>/<cell_slug>/<safe_scenario_name>.jsonl`) exists.
3. The trace file is non-empty.

This means a cell that crashed before writing any trace events is retried on the next sweep run; a cell that wrote a partial trace and then crashed is *not* retried because the file is non-empty. The default policy errs on the side of "skip what looks complete". Pass `--no-resume` when you want to force every cell to re-run, or delete the cell's directory by hand for targeted retries.

The runner appends to `runs.jsonl` rather than overwriting it. A resumed sweep adds a `skipped` record for each cell it does not re-run, so `runs.jsonl` is the authoritative per-cell record across invocations.

## Concurrency control

`[sweep.run].concurrency = 1` is the default and runs cells one at a time. Above 1, the runner uses a `ThreadPoolExecutor` with `max_workers = concurrency` and dispatches cells as workers become available.

The trade-offs:

- Ensemble's resource locks serialise GPU tools across actors *within* a world instance. They do not serialise across cells, because each cell is a separate process. Two cells targeting the same GPU will fight for the device, and the second cell's `compile_kernel` or `submit_kernel` will be slow at best and OOM at worst.
- Partitioning by device is the way to scale concurrency across a multi-GPU host. Today the matrix has no `device_index` dimension; you can work around this by running multiple sweeps in parallel, each with `[sweep.run.extra_env].POPCORN_DEVICE_INDEX = "0"`, `"1"`, etc., and a different subset of the matrix.
- The `runs.jsonl` append uses a single file handle; concurrent appends from threads use Python's GIL plus a per-line write, which is safe in practice. Do not run two `run_sweep.py` invocations against the same sweep name simultaneously; the `runs.jsonl` file will interleave records.

## Adding to the leaderboard

After a sweep finishes, point `scripts/publish_traces.py` at the repo and it walks every `traces/*/<cell_slug>/<scenario>.jsonl` plus the matching `runs.jsonl` entries to rebuild the leaderboard. The summary fields the leaderboard uses (timestamp, scenario, model, persona, level, problem, outcome, speedup, cost) come from the per-cell trace plus the cell's matrix point recorded in `runs.jsonl`. See `scripts/publish_traces.py` for the precise shape of `runs.json` the front-end consumes.

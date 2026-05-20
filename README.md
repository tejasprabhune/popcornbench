# PopcornBench

A KernelBench fork driven by [ensemble](https://github.com/tejasprabhune/ensemble).
Kernel evaluation primitives (correctness, timing, profiling, static
checking) stay here as a Python library. The agent loop, sweep runner,
and report builder have been replaced by ensemble scenarios that target
the `popcorn` world.

## Layout

```
src/kernelbench/            kernel eval library (eval, timing, dataset,
                            profile, sass, ert, kernel_static_checker,
                            sass_parser, nsight_parser, prompts, ...)
KernelBench/                bench dataset (level1..level5)
popcorn_world/              ensemble plugin: tools, predicates, scenarios
                            personas, world.toml
results/timing/             baseline times for several GPUs
assets/                     figures used in the original readme
```

## Setup

You need Rust 1.80+, Python 3.10, and `uv`. PopcornBench is a uv
workspace; `popcorn_world` is a workspace member that depends on
`popcornbench` (this package) and `ensemble`.

```sh
# 1. Sync the workspace. Installs PopcornBench + popcorn_world; ensemble
#    must be installed separately (see step 2).
uv sync --extra gpu

# 2. From your ensemble checkout, build the ensemble extension and CLI:
#    cd ~/Documents/ensemble
#    uv sync
#    cargo build -p ensemble-cli
#    The CLI lands at ~/Documents/ensemble/target/debug/ensemble.

# 3. Register the popcorn world with ensemble (one-time).
~/Documents/ensemble/target/debug/ensemble worlds add \
    popcorn $(pwd)/popcorn_world

# 4. API keys.
cp .env.example .env
$EDITOR .env
```

`uv pip install` of ensemble from a local checkout works if you prefer
not to symlink: `uv pip install -e ~/Documents/ensemble/python/ensemble`.

## Run a single problem

```sh
# Pick a problem and persona via env (or edit popcorn_world/scenarios.toml).
export POPCORN_LEVEL=1
export POPCORN_PROBLEM_ID=19
export POPCORN_AGENT_MODEL=claude-sonnet-4-5
export POPCORN_GPU_BUDGET=600

~/Documents/ensemble/target/debug/ensemble run \
    popcorn.single_problem --world popcorn

# The trace lands in ./traces/popcorn.single_problem.jsonl
# Inspect it:
~/Documents/ensemble/target/debug/ensemble trace view \
    traces/popcorn.single_problem.jsonl
```

## Run with a code-reviewer auditor

```sh
~/Documents/ensemble/target/debug/ensemble run \
    popcorn.judge_review --world popcorn
```

The author and reviewer share the world but have different tool sets.
The reviewer cannot author kernels; it can run correctness, static-check
a submission, and disassemble it.

## Run a declarative variant

The TOML matrix at `popcorn_world/scenarios.toml` compiles to the same
runtime. Each entry is named:

```sh
~/Documents/ensemble/target/debug/ensemble run l1p19_methodical --world popcorn
```

## SSH workflow

Local checkout, work on a feature branch, push to GitHub, pull on the
GPU box, run there:

```sh
# laptop
git push -u origin tejas/ensemble

# GPU box
git clone git@github.com:arjun-banerjee/POPCORNBENCH.git
git -C POPCORNBENCH checkout tejas/ensemble
cd POPCORNBENCH
uv sync --extra gpu
ensemble worlds add popcorn $(pwd)/popcorn_world
ensemble run popcorn.single_problem --world popcorn
```

## What changed vs the previous PopcornBench

Gone:

- `src/kernelbench/agent/` (custom OpenAI-Responses agent loop)
- `scripts/` (sweep runner, report builder, gh-pages publisher,
  one-off generate/eval entry points)
- `notebooks/`, `configs/`, `EVAL.md`, `tmp.bash`, `requirements.txt`

The two parser modules (`nsight_parser.py`, `sass_parser.py`) moved out
of `src/kernelbench/agent/` into `src/kernelbench/` since they are pure
utilities; popcorn_world's tools import them by their new paths.

Kept:

- The eval/timing/dataset/profile/sass/ert/static-checker library code.
- The full KernelBench dataset.
- `results/timing/` (baseline times for several GPUs).

Added:

- `popcorn_world/`: ensemble plugin defining the world, tools (mapped
  1:1 from the old agent tool set), predicates for grading, three
  personas (methodical_engineer, speed_obsessed, code_reviewer), two
  Python scenarios, and a declarative `scenarios.toml`.

## Reward-hacking posture

The static checker in `src/kernelbench/kernel_static_checker.py`
covers the original KernelBench list (try/except fallbacks, timing
function monkey patches, fp32 to fp16 downgrades, lazy tensor
subclasses, stream and thread injection, backend implementation
requirements) plus four new patterns added for popcorn_world:

- `super_forward`: a `ModelNew.forward` that just calls
  `super().forward(...)` and so passes correctness for free.
- `equal_nan`: hides NaN-producing kernels.
- `sleep_calls`: pushes compute outside the timing window.
- `init_heavy` (warning): suggests `ModelNew.__init__` is doing the
  real work that belongs in `forward`.

Beyond the static checker, popcorn_world adds:

- `static_check` is a regular tool the agent is encouraged to call
  before `submit_kernel`. The grader exposes
  `submitted_without_static_check` so a scenario can penalize skipping
  it.
- `submit_kernel` returns the kernel's runtime in microseconds but
  never reveals the reference runtime or the speedup ratio. The
  `excessive_speedup` flag from `eval_kernel_against_ref` is surfaced
  as its own grader predicate.
- When `POPCORN_HELD_OUT_SEED` is set, `submit_kernel` re-runs
  correctness against that seed after the agent-visible check
  passes. The result lands on the kernel record but is hidden from
  the agent; the grader reads it via the `held_out_correctness_passed`
  and `held_out_correctness_failed` predicates.
- Setting `POPCORN_SANDBOX_GPU_TOOLS=true` opts the heavy GPU tools
  (`compile_kernel`, `run_correctness`, `submit_kernel`,
  `profile_kernel`, `disassemble_kernel`) into ensemble's subprocess
  sandbox, so a CUDA-context-fatal kernel only kills its worker
  rather than poisoning the rest of the run. Off by default because
  the sandbox worker reimports popcorn_world from scratch (i.e. it
  does not see the parent's `fetch_problem` state); enabling it
  requires the scenario to pass the problem reference through the
  args of every sandboxed call.
- The `code_reviewer` persona pairs with `judge_review` to add a
  second pass: a separate agent describes what the kernel does, names
  the speedup mechanism, and flags anything suspicious. Its tools are
  read-only with respect to authoring.

## Publishing traces to GitHub Pages

`scripts/publish_traces.py` copies the ensemble trace viewer plus
every file in `traces/` into a worktree on the `gh-pages` branch and
pushes. Each run gets its own page (`https://<user>.github.io/<repo>/<run>/viewer.html`),
plus a top-level index. Run once after a scenario, or pass
`--watch 300` to republish on a five-minute cadence while a sweep is
running. Set `ENSEMBLE_ROOT` (or pass `--ensemble-root`) to the
ensemble checkout that holds `site/`.

## What ensemble grew along with popcorn_world

A few small enhancements were added upstream so popcorn_world (and
other future worlds) get them for free:

- `World(trace_path=...)` and `world.set_trace_path(...)` mirror every
  event to a JSONL file, flushed per line. The CLI sets it
  automatically so `traces/<scenario>.jsonl` is readable while the
  run is happening.
- The trace viewer polls `trace.jsonl` every two seconds, sticks to
  the tail unless the user scrubs back, renders fenced code blocks
  and long tool args as collapsible code, and emits a grader-output
  panel at the end of every run.
- `set_budget(unit, amount, actor=...)` and `record_cost(unit, amount, actor=...)`
  attribute spend per actor. Halts fire on whichever cap (world-wide
  or per-actor) is crossed first.
- `PluginTool.sandbox=True` runs the tool in a fresh subprocess via
  `python -m ensemble.tool_worker`. Useful for any tool whose work
  fully fits in its args.
- The python `tool()` helper now forwards `costs` and `progress` from
  the wrapped function into the trace.

## License

MIT. See `LICENSE`.

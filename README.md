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
already covers try/except fallbacks, timing-function monkey patches,
fp32 to fp16 downgrades, lazy tensor subclasses, stream and thread
injection, backend-specific implementation requirements. popcorn_world
layers more on top:

- `static_check` is a regular tool the agent is encouraged to call
  before `submit_kernel`. The grader exposes
  `submitted_without_static_check` so a scenario can penalize skipping
  it.
- `submit_kernel` returns the kernel's runtime in microseconds but
  never reveals the reference runtime or the speedup ratio. The
  `excessive_speedup` flag from `eval_kernel_against_ref` is surfaced
  as its own grader predicate.
- The `code_reviewer` persona pairs with `judge_review` to add a
  second pass: a separate agent describes what the kernel does, names
  the speedup mechanism, and flags anything suspicious. Its tools are
  read-only with respect to authoring.

## Known gaps in ensemble that popcorn_world works around

- ensemble has no SSH tool runner. We pull the branch on the GPU box
  and run scenarios in-process, which works because PopcornBench's
  eval code already runs on whatever device CUDA exposes.
- ensemble's `tool()` helper does not forward `costs` or `progress`.
  popcorn_world bypasses that helper and constructs `PluginTool`
  records directly with the full JSON ABI. (An upstream fix is in
  flight; see `~/Documents/ensemble/python/ensemble/world.py`.)
- ensemble's trace is serialized at the end of a run. While a run is
  going, the trace is not on disk. The intent is to add a "live"
  writer that flushes per event; for now use `ensemble trace view`
  after the run finishes.
- ensemble's trace viewer does not pretty-print fenced code blocks.
  Kernel source shows as a single-line `tool_call.args.kernel_code`
  for now.

## License

MIT. See `LICENSE`.

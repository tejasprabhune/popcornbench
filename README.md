# PopcornBench

PopcornBench is a KernelBench-derived benchmark for evaluating LLM agents on GPU kernel tasks. An agent is given a PyTorch reference module and a set of tools (compile, run for correctness, run for timing, static-check, profile, disassemble), and it iterates until it submits a custom CUDA kernel that matches the reference and beats it for runtime. The benchmark is driven by [ensemble](https://github.com/tejasprabhune/ensemble), which provides the agent loop, the trace recorder, the per-actor cost budgeting, the resource locks, and the static viewer; this repo provides the world (tools, predicates, personas, scenarios) and the underlying kernel evaluation library.

The interesting parts of the design are on the grader side. The agent has real profiling and disassembly tools, so it can read Nsight roofline metrics and SASS disassembly without leaving the agent loop. The grader has ground-truth access the agent does not: the reference runtimes per problem, a held-out shape seed that re-verifies a passing kernel against an unseen input, and a static checker that pattern-matches the obvious reward-hacking templates. The combination of layered defense (static checker plus held-out re-verification plus the excessive-speedup flag from the eval) is what lets a result be reported with some confidence that the kernel actually computes the operator rather than memorising the test seed.

## Setup

PopcornBench is a `uv` workspace with `popcornbench` as the root package and `popcorn_world` as a member. The ensemble runtime is a separate checkout that you install once and re-use across projects.

The first step syncs this repo's workspace. This installs `popcornbench` (the eval library), `popcorn_world` (the ensemble plugin), and their dependencies. The `gpu` extra adds Triton, CUTLASS DSL, TileLang, CuPy, and Nsight Python; on a host without CUDA you can omit it and the smoke flow still works through the mock backend.

```sh
uv sync --extra gpu
```

The second step installs ensemble. Build the Rust CLI and the Python extension once from the ensemble checkout. Use any path you like; the rest of this README assumes `~/Documents/ensemble`.

```sh
cd ~/Documents/ensemble
uv sync
cargo build -p ensemble-cli
# Binary lands at ~/Documents/ensemble/target/debug/ensemble.
```

The third step registers the popcorn world with ensemble. This is a one-time write to `~/.ensemble/worlds.toml`. Without it, every scenario invocation needs `--package-dir popcorn_world` instead of resolving the world by name.

```sh
~/Documents/ensemble/target/debug/ensemble worlds add \
    popcorn $(pwd)/popcorn_world
```

The fourth step is API keys. The `auto` LLM backend picks the first key it can resolve, so set the providers you intend to use and leave the rest blank.

```sh
cp .env.example .env
$EDITOR .env
```

A quick verification run uses the mock backend (no provider keys required) and exercises the trace pipeline end to end:

```sh
~/Documents/ensemble/target/debug/ensemble run popcorn.single_problem \
    --world popcorn --backend mock
```

That writes `traces/popcorn.single_problem.jsonl` and prints a JSON line with the grader scores. If that line appears with `"status": "ok"`, the workspace is set up correctly.

## Organization

The benchmark is laid out across five levels that test progressively harder targets.

Level 1 is 100 single-operator problems: matrix multiplication variants, convolutions, activations, reductions, normalisation. The agent writes a CUDA kernel that replaces one PyTorch operator. This is where most published baselines run.

Level 2 is 100 fused-operator problems: `Conv2D_ReLU_BiasAdd`, `ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh`, and similar compositions. The agent writes a kernel that fuses the sequence so the intermediate tensors stay in registers or shared memory.

Level 3 is 50 full-model problems: `MLP`, `ResNet101`, `VGG16`. The agent writes a kernel (or a small library of kernels) that runs the whole model end to end. The reference is the PyTorch model class.

Level 4 is 20 transformer benchmarks: real Hugging Face model identifiers at fixed batch and sequence sizes (`EleutherAI-gpt-neo-2p7B_bs32_seq256`, `gpt2_bs1_seq1023`, and so on). The reference is a transformer forward pass; the agent is expected to fuse attention, normalisation, and feed-forward layers.

Level 5 is the hardware translation task. The agent gets an A100-tuned CUDA source file (paged attention, Flash Attention 2, Marlin INT4 GEMM, SwiGLU, fused RMSNorm) and re-optimises it for H100. Paired source kernels live under `KernelBench/level5/kernels/a100/` and `KernelBench/level5/kernels/h100/`. The translation scenario itself is not yet wired into `popcorn_world`; the `normal_translation` persona is in place for when it lands.

## Running a single problem

The minimum invocation is one command:

```sh
ensemble run popcorn.single_problem --world popcorn
```

The defaults (level 1, problem 19, `claude-sonnet-4-5`, the `methodical_engineer` persona, 20 turns, 600 GPU-seconds) come from the scenario; override any of them through env vars (the full list is in [docs/reference/env-vars.md](docs/reference/env-vars.md)).

A typical invocation that pins everything explicitly looks like this:

```sh
POPCORN_LEVEL=1 POPCORN_PROBLEM_ID=19 \
POPCORN_AGENT_MODEL=claude-sonnet-4-5 POPCORN_PERSONA=normal \
POPCORN_HELD_OUT_SEED=2026 \
ensemble run popcorn.single_problem --world popcorn
```

After the run, the trace lands at `traces/popcorn.single_problem.jsonl` and ensemble prints one JSON line on stdout:

```json
{"scenario": "popcorn.single_problem",
 "status": "ok",
 "scores": {"submitted": 1.0, "submission_passed": 1.0,
            "any_correct": 1.0, "lint_skipped": 0.0,
            "excessive_speedup_flagged": 0.0},
 "trace_path": "traces/popcorn.single_problem.jsonl"}
```

The `scores` object is what predicate-driven graders return. A `lint_skipped` of 0.0 means the agent passed `static_check` before submitting; an `excessive_speedup_flagged` of 0.0 means the eval did not raise the suspect-fast flag. The full predicate catalog is in [docs/reference/predicates.md](docs/reference/predicates.md).

To view the trace, use the ensemble viewer:

```sh
ensemble trace view traces/popcorn.single_problem.jsonl
```

The viewer polls the trace file every two seconds and surfaces grader output at the end of the run.

## Running with the code-reviewer auditor

The `popcorn.judge_review` scenario pairs two agents on the same problem. The author works the kernel; a separate reviewer (the `code_reviewer` persona, with a read-only tool set) audits the trace as it builds. The reviewer describes what the kernel does, names the speedup mechanism, flags anything the static checker may have missed, and records a verdict in its hidden state.

```sh
ensemble run popcorn.judge_review --world popcorn
```

The reviewer defaults to the same model as the author so the two actors are roughly capacity-matched; set `POPCORN_REVIEWER_MODEL` to use a smaller model when running an asymmetric pairing experiment. The full env-var surface for both actors is in [env-vars.md](docs/reference/env-vars.md).

## Running a sweep

A sweep expands a `(level, problem_id, model, persona, seed)` matrix into one cell per tuple and runs each through ensemble. The runner lives at `scripts/run_sweep.py` and consumes a small TOML:

```sh
uv run python scripts/run_sweep.py --config sweep.example.toml
```

Per-cell traces land under `traces/<sweep.name>/<cell_slug>/`, plus a `runs.jsonl` per sweep that the leaderboard manifest consumes. Cells with an existing non-empty trace are skipped (resume on by default). The schema for the config, the cell-slug rules, and the concurrency trade-offs are documented in [docs/reference/sweep-config.md](docs/reference/sweep-config.md).

## Personas

The persona catalog under `popcorn_world/personas/` splits into baselines and interventions. `normal` is the control condition for the standard PyTorch-to-CUDA write task; `normal_translation` is the control for the level-5 hardware-translation task. Both are unprescriptive: the system prompt is the task description, the correctness contract, the tool surface, and the scoring objective, with no behavioural modifiers. Comparing any other persona against the matching baseline is what makes a persona experiment meaningful.

The three intervention personas (`methodical_engineer`, `speed_obsessed`, `code_reviewer`) compose on top of `normal`: each prompt begins with the verbatim baseline and adds a section that adjusts style (methodical, speed-obsessed) or role (reviewer). The per-persona detail, the intended use case, and the file paths are in [docs/reference/personas.md](docs/reference/personas.md).

## Where results go

Locally, every run writes a JSONL trace under `traces/`. Single runs land at `traces/<scenario>.jsonl`; sweep cells land at `traces/<sweep>/<cell_slug>/<scenario>.jsonl`. Each line is one event (tool call, tool result, state diff, cost, grader). The viewer polls the file while a run is in flight, so the trace is readable as it grows.

For publishing, `scripts/publish_traces.py` copies the ensemble viewer plus every published run into a worktree on the `gh-pages` branch and pushes. Each run gets its own per-run viewer URL (`<repo>.github.io/<run-slug>/viewer.html`); the script also rebuilds a top-level leaderboard at the repo root. Run once after a scenario or pass `--watch 300` to republish every five minutes while a sweep is running. The aggregate leaderboard is at https://arjun-banerjee.github.io/POPCORNBENCH/.

## SSH and GPU box workflow

Most development happens on a laptop and most actual evaluations happen on a GPU box. The pattern is to keep the local checkout as the source of truth, push to GitHub, and pull on the GPU box.

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

The GPU box needs the same ensemble checkout, the same world registration step, and the same set of API keys. Once the world is registered, the runner command is identical to the laptop's. Long sweeps are typically run in `tmux` with `scripts/publish_traces.py --watch 300` in a second pane so the leaderboard refreshes without manual intervention.

## Reward-hacking posture

The grader's defense is layered rather than monolithic, because no single check catches every category of reward hacking that the agent can attempt. The static checker (`src/kernelbench/kernel_static_checker.py`) pattern-matches obvious templates at the source level: try/except fallbacks to the reference, timing-function monkey patches, thread or stream injection, lazy-tensor subclasses, NaN-suppressing tolerance, sleep calls inside the kernel module, `super().forward()` shortcuts, and the backend-specific implementation requirement. The agent can call `static_check` itself as a tool; the `submitted_without_static_check` predicate then penalises submissions that skipped the lint. This catches the cheats that show up as a regex match.

The dynamic layer catches the cheats that survive the source-level pass. `submit_kernel` deliberately hides the reference runtime and the speedup ratio from the agent, so a model cannot read its own score and reward-hack toward it. The eval's `excessive_speedup` flag is surfaced as its own grader predicate, which lets the grader treat a suspiciously fast submission as a suspect rather than an automatic pass. When `POPCORN_HELD_OUT_SEED` is set, every passing correctness check is followed by a rerun against the held-out seed; the result is recorded on the kernel ledger but hidden from the agent. A kernel that memorised the default-seed output passes the agent-visible check and fails the held-out re-verification; the grader reads both. The full predicate catalog and the per-pattern reasoning are in [docs/reference/predicates.md](docs/reference/predicates.md).

## Reference

The reference section under [`docs/reference/`](docs/reference/index.md) is the contract documentation. Each page is a single scrollable document that reads in isolation.

- [tools.md](docs/reference/tools.md) catalogs every tool `popcorn_world` registers with argument schemas, return shapes, state effects, cost annotations, resource locks, sandbox eligibility, timeouts, and progress events.
- [predicates.md](docs/reference/predicates.md) catalogs every grader predicate and every static-checker pattern, including why each pattern is treated as a reward-hacking risk.
- [personas.md](docs/reference/personas.md) catalogs every persona and the baseline-versus-intervention distinction the experimental design rests on.
- [scenarios.md](docs/reference/scenarios.md) covers the two Python scenarios and the `scenarios.toml` matrix, including the grader expressions each one returns.
- [env-vars.md](docs/reference/env-vars.md) is the catalog of every environment variable the project reads, grouped by what it controls.
- [sweep-config.md](docs/reference/sweep-config.md) covers the sweep configuration TOML schema, the runner CLI, and the concurrency and resume semantics.

The ensemble runtime, the persona schema, the scenario decorator, the trace event format, the predicate concept, and the CLI flags are documented separately at https://tejasprabhune.github.io/ensemble/.

## License

MIT. See `LICENSE`.

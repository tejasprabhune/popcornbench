# Scenarios

This page documents the scenarios `popcorn_world` ships: the two Python scenarios driven by `@scenario`, and the declarative `scenarios.toml` matrix that compiles to the same runtime. The ensemble scenario surface (the `@scenario` decorator, `World` methods, the grader DSL) is documented in the [ensemble scenarios reference](https://tejasprabhune.github.io/ensemble/reference/scenarios.html); this page covers what each PopcornBench scenario does, the env vars it reads, and how the actors are wired.

## popcorn.single_problem

File: `popcorn_world/scenarios/single_problem.py`. World: `popcorn`. One agent against one problem.

The scenario spawns a single `kernel_author` agent, kicks off `fetch_problem` as the initial action, then yields `world.until(world.turn_count > POPCORN_MAX_TURNS)` to run until the turn budget is exhausted. The agent's persona, model, and tool set are all env-driven so the same scenario file works for a default smoke run and a parameter sweep.

Env vars read (defaults in parentheses):

- `POPCORN_LEVEL` (1), `POPCORN_PROBLEM_ID` (19): the KernelBench problem to fetch.
- `POPCORN_DATASET_SRC` ("huggingface"): passed to `fetch_problem`.
- `POPCORN_AGENT_MODEL` ("claude-sonnet-4-5"): the LLM behind the agent.
- `POPCORN_PERSONA` ("methodical_engineer"): persona TOML name; resolves through the world's `personas_dir`.
- `POPCORN_TOOLS`: comma-separated tool names that override the default tool set. Empty (the default) keeps the six-tool default below.
- `POPCORN_MAX_TURNS` (20): turn budget.
- `POPCORN_GPU_BUDGET` (600): GPU-seconds cap, applied via `world.set_budget("gpu_seconds", ...)`.

Default tool set when `POPCORN_TOOLS` is empty: `fetch_problem`, `compile_kernel`, `run_correctness`, `get_gpu_specs`, `static_check`, `submit_kernel`. The opt-in profiling and disassembly tools (`profile_kernel`, `disassemble_kernel`, `ert_roofline`) are excluded by default; pass them in `POPCORN_TOOLS` if the host has `ncu` / `cuobjdump` / the ERT driver and the run wants them.

Graders returned:

```python
{
    "submitted": 1.0 if submit_called else 0.0,
    "submission_passed": 1.0 if submit_passed else 0.0,
    "any_correct": 1.0 if any_correct else 0.0,
    "lint_skipped": 1.0 if submitted_without_static_check else 0.0,
    "excessive_speedup_flagged": 1.0 if excessive_speedup_flagged else 0.0,
}
```

The corresponding predicates are documented in [predicates.md](predicates.md). The `lint_skipped` cell is the raw predicate value rather than `not submitted_without_static_check`, so a higher value is worse for that cell; this differs from the TOML scenarios below, which use the negated form so all cells point the same way.

Typical invocation (assumes the world is registered and the ensemble CLI is on `PATH`):

```bash
POPCORN_LEVEL=1 POPCORN_PROBLEM_ID=19 \
POPCORN_AGENT_MODEL=claude-sonnet-4-5 POPCORN_PERSONA=normal \
ensemble run popcorn.single_problem --world popcorn
```

## popcorn.judge_review

File: `popcorn_world/scenarios/judge_review.py`. World: `popcorn`. Two agents: one author, one reviewer.

The scenario spawns `author` (default persona `speed_obsessed`) with the full kernel-author tool kit and `reviewer` (persona `code_reviewer`, not overridable) with a read-only audit tool kit: `run_correctness`, `static_check`, `disassemble_kernel`. The author's `fetch_problem` runs as `act` (so the event appears in the trace as an author action). The reviewer is seeded with a `say` to the author, `"I'm reviewing this kernel. Walk through your approach as you go."`, so the trace starts with the audit conversation explicit.

Env vars read (defaults in parentheses):

- `POPCORN_LEVEL` (1), `POPCORN_PROBLEM_ID` (19): the problem.
- `POPCORN_DATASET_SRC` ("huggingface"): `fetch_problem`'s source.
- `POPCORN_AUTHOR_MODEL` ("claude-sonnet-4-5"): the LLM behind the author.
- `POPCORN_REVIEWER_MODEL` (defaults to `POPCORN_AUTHOR_MODEL`): the LLM behind the reviewer; defaulting to the same model is a deliberate choice so the reviewer is roughly capacity-matched to the author.
- `POPCORN_AUTHOR_PERSONA` ("speed_obsessed"): the author's persona.
- `POPCORN_MAX_TURNS` (30): turn budget. Larger default than `single_problem` because two actors are sharing the budget.
- `POPCORN_GPU_BUDGET` (900): GPU-seconds cap.

Graders returned are the same five cells as `popcorn.single_problem`. The reviewer's verdict is not yet wired into the grader output; it lands in `reviewer.hidden_state["verdict"]` and can be read post-run by a downstream consumer. Surfacing it as a grader cell is a follow-up.

Typical invocation:

```bash
POPCORN_LEVEL=1 POPCORN_PROBLEM_ID=19 \
POPCORN_AUTHOR_MODEL=claude-sonnet-4-5 \
POPCORN_REVIEWER_MODEL=claude-haiku-4-5 \
ensemble run popcorn.judge_review --world popcorn
```

## popcorn.translate_problem

File: `popcorn_world/scenarios/translate_problem.py`. World: `popcorn`. One agent against one hardware-translation problem (the level-5 task: re-optimise an A100 source kernel for H100).

The scenario spawns a single `kernel_translator` agent (default persona `normal_translation`), kicks off `fetch_translation_problem` as the initial action (defaulting to `level=5`, `problem_id=1`, `source_arch="a100"`, `target_arch="h100"`), and yields the same `world.turn_count > POPCORN_MAX_TURNS` until predicate as the other scenarios. The agent's default tool set is `fetch_translation_problem`, `compile_kernel`, `get_gpu_specs`, `static_check`, and `submit_kernel`. `run_correctness` is left out of the default set because the level-5 problems have no PyTorch reference; including it returns a short-circuit message.

Env vars (defaults in parentheses):

- `POPCORN_LEVEL` (5), `POPCORN_PROBLEM_ID` (1): the translation problem.
- `POPCORN_SOURCE_ARCH` ("a100"), `POPCORN_TARGET_ARCH` ("h100"): source and target GPU architectures. The current dataset only has A100 and H100, but the arg signatures are open so new pairs (Hopper to Blackwell, RDNA3 to RDNA4) can land without scenario changes.
- `POPCORN_PERSONA` ("normal_translation"): the persona that resolves through the world's `personas_dir`. Set to a translation-flavoured intervention persona once any land.
- `POPCORN_AGENT_MODEL` ("claude-sonnet-4-5"): the LLM behind the agent.
- `POPCORN_TOOLS`, `POPCORN_MAX_TURNS` (20), `POPCORN_GPU_BUDGET` (600): same shape as `popcorn.single_problem`.

Graders returned:

```python
{
    "submitted": 1.0 if submit_called else 0.0,
    "submission_recorded": 1.0 if submit_passed else 0.0,
    "lint_skipped": 1.0 if submitted_without_static_check else 0.0,
}
```

`submission_recorded` is the participation signal in translation mode: the kernel was submitted and the tool returned `ok=true`. Correctness and speedup signals are absent until per-problem PyTorch wrappers land for level 5. The [tools page](tools.md#fetch_translation_problem) covers the current limitation in detail.

Typical invocation:

```bash
POPCORN_LEVEL=5 POPCORN_PROBLEM_ID=1 \
POPCORN_AGENT_MODEL=claude-sonnet-4-5 POPCORN_PERSONA=normal_translation \
ensemble run popcorn.translate_problem --world popcorn
```

## scenarios.toml matrix

File: `popcorn_world/scenarios.toml`. The declarative form lets a sweep enumerate `(level, problem_id, persona, model)` cells without writing a Python scenario per cell. The loader at `ensemble.load_manifest(path)` parses the file and registers a scenario per top-level `[scenario.<name>]` entry into the global registry, so the same runner code drives both Python `@scenario` and TOML scenarios.

### Schema

A scenarios.toml entry has five sections.

`[scenario.<name>]` is the scenario header. Required fields: `world`, the world name to construct (`"popcorn"` for everything in this file). Optional: `duration_turns` (used as `world.turn_count > N` for the `until` predicate, default 20); `seed` (informational; not consumed by the runtime today).

`[[scenario.<name>.users]]` declares each simulated user. Each entry supplies `id`, `persona`, optional `hidden_goal`, optional `model`, and an optional `initial_action = { tool = "...", args = {...} }`. The loader calls `world.spawn_user(...)` with these fields, then invokes `user.act(...)` for the initial action. `popcorn_world` does not currently use simulated users (it is agent-only), so the existing scenarios omit this section.

`[[scenario.<name>.agents]]` declares each agent. Each entry supplies `id`, `persona`, `model`, `tools`, and an optional `initial_action`. `tools` is the per-agent tool restriction; `None` would mean every tool the world registered, but the scenarios here always pass an explicit list so the run does not accidentally grant a profiling tool the host cannot serve.

`[scenario.<name>.graders]` is a table of `<name> = "<expression>"`. Each expression is evaluated against the grader namespace (every world predicate by name, plus the boolean DSL of `and`/`or`/`not`/parens). Truthy values become `1.0`, falsy `0.0`. Per the [scenarios reference](https://tejasprabhune.github.io/ensemble/reference/scenarios.html#grader-expressions), comparisons and arbitrary calls are rejected.

### Worked example

The first entry in `popcorn_world/scenarios.toml` looks like this:

```toml
[scenario.l1p19_methodical]
world = "popcorn"
duration_turns = 20
seed = 1

[[scenario.l1p19_methodical.agents]]
id = "kernel_author"
persona = "methodical_engineer"
model = "claude-sonnet-4-5"
tools = [
    "fetch_problem",
    "compile_kernel",
    "run_correctness",
    "get_gpu_specs",
    "static_check",
    "submit_kernel",
]
initial_action = { tool = "fetch_problem", args = { level = 1, problem_id = 19, dataset_src = "huggingface" } }

[scenario.l1p19_methodical.graders]
submitted = "submit_called"
correct = "submit_passed"
lint_hygiene = "not submitted_without_static_check"
no_excessive_speedup = "not excessive_speedup_flagged"
```

`l1p19_methodical` is the level-1 problem 19 cell with the `methodical_engineer` persona. The grader expressions point all cells the same way: `1.0` is the favourable outcome (`submitted == 1.0` means the agent submitted, `lint_hygiene == 1.0` means the agent linted before submitting, and so on). This is the convention to follow when adding new cells; sweep aggregations that compute "favourable outcomes per model" rely on it.

The second entry, `l1p19_judge`, uses two agents (author plus reviewer), 30 turns, and the same grader cells minus the lint hygiene one (since the reviewer does the lint pass externally). It is functionally the TOML equivalent of `popcorn.judge_review` for a single problem.

### Adding cells

A sweep over multiple problems and personas is currently a copy-paste exercise: one entry per `(level, problem_id, persona, model)` tuple. The [sweep configuration](sweep-config.md) page documents the matrix-driven runner that generates these entries from a smaller config, including how `seed` is varied across replicas and how the runner deduplicates against an existing results table for resumes.

### Running TOML scenarios

The CLI loads a manifest via `--manifest`:

```bash
ensemble run l1p19_methodical \
  --world popcorn \
  --manifest popcorn_world/scenarios.toml
```

Per the [ensemble CLI reference](https://tejasprabhune.github.io/ensemble/reference/cli.html#ensemble-run), `--world popcorn` resolves the world through `~/.ensemble/worlds.toml` (registered via `ensemble worlds add`). Without the registry entry, pass `--package-dir popcorn_world` so the CLI imports the world's Python package.

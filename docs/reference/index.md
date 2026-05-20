# Reference

This section is the contract documentation for PopcornBench. Each page reads in isolation; cross-links go to a specific section rather than expecting you to read several pages in order. Tutorial-flavoured material lives in the top-level [README](../../README.md); these pages cover what a thing is, what it does, and how it composes with the rest of the project.

The pages below mirror the major surfaces of `popcorn_world`, plus two operational pages (env vars, sweep configuration) that pull together settings scattered across code.

The shortest path through the reference is to read the [scenarios](scenarios.md) page first to see how a run is shaped, the [env vars](env-vars.md) page to see how it is parameterised, and then [tools](tools.md) plus [predicates](predicates.md) for the surface the agent actually touches and the surface the grader reads back. [Personas](personas.md) and [sweep config](sweep-config.md) are the experimental-design layer on top.

## [Tools](tools.md)

Every tool registered by `popcorn_world`. Per tool: argument schema with types and defaults, return shape, what world state the tool reads or mutates, whether it accrues `gpu_seconds`, whether it declares a GPU resource lock, whether it emits progress, and what timeout it carries. The pages are grouped by what the tool does: problem access (`fetch_problem`), evaluation (`compile_kernel`, `run_correctness`, `submit_kernel`), introspection (`static_check`, `profile_kernel`, `disassemble_kernel`, `ert_roofline`, `get_gpu_specs`). Each tool has a worked example drawn from a real scenario.

## [Predicates](predicates.md)

The grader predicates `popcorn_world` exposes, and the static-checker patterns the `static_check` tool runs. Predicates are the named yes/no questions a grader composes; the [ensemble predicates reference](https://tejasprabhune.github.io/ensemble/reference/predicates.html) covers the concept end to end. This page is the catalog for the specific predicates the world ships and the static-check rules that feed them.

## [Personas](personas.md)

Every persona under `popcorn_world/personas/`. Two control-condition baselines (`normal`, `normal_translation`) and three interventions composed over the CUDA baseline (`methodical_engineer`, `speed_obsessed`, `code_reviewer`). The baseline-versus-intervention distinction is the first thing on the page; the per-persona detail covers role, tool set, system prompt summary, intended scenario, and what the intervention is meant to test relative to the baseline.

## [Scenarios](scenarios.md)

The two Python scenarios (`popcorn.single_problem`, `popcorn.judge_review`) and the declarative `scenarios.toml` matrix. Per scenario: world, actors and personas involved, tool set per actor, grader predicates, env vars consumed, what the run looks like turn by turn. The TOML matrix gets a schema walk-through and a worked example.

## [Environment variables](env-vars.md)

Every env var the project reads, grouped by what it controls: problem selection, agent model, budgets, grading, runtime, API keys, ensemble paths. Per var: type, default, what it controls, when you would change it. Use this page when you are reading a scenario for the first time and want to know which knobs are exposed.

## [Sweep configuration](sweep-config.md)

The sweep configuration format and runner. A sweep is a matrix of `(level, problem_id, model, persona, seed)` tuples that get expanded into per-cell scenario runs, executed by an ensemble-driven runner, and aggregated into the leaderboard. This page documents the schema, the runner command, where results land, the resume semantics, and how concurrency is controlled.

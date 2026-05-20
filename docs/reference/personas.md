# Personas

This page is the catalog for every persona under `popcorn_world/personas/`. The persona TOML schema, the hidden-state mechanism, the `PromptedPersona` wrapper, and the trained-adapter auto-wiring all live in the [ensemble personas reference](https://tejasprabhune.github.io/ensemble/reference/personas.html); this page is just the catalog of what `popcorn_world` ships and how to read the lineup.

## Baseline versus intervention

The personas split into two roles. `normal` and `normal_translation` are control-condition baselines: the system prompt is the task description, the correctness contract, the tool surface, and the scoring objective, with no prescriptive tool ordering and no behavioral modifiers. They are the prompts a PopcornBench experiment compares everything else against.

`methodical_engineer`, `speed_obsessed`, and `code_reviewer` are interventions composed on top of the baseline. Each persona's system prompt begins with the verbatim baseline section from `normal.toml` and adds a persona-specific section that adjusts style (methodical, speed-obsessed) or role (reviewer) without dropping any baseline information. The composition discipline is what makes persona-vs-persona comparisons meaningful: outcome differences are attributable to the intervention, not to the absence of the baseline information one persona happened to have and another did not.

The verbatim duplication is a hand-roll because ensemble's persona loader does not currently support template includes. Pulled out into a separate file the personas could reference an externalised baseline; until that lands in ensemble, every intervention persona repeats the baseline literally. See the comment at the top of each intervention TOML for the same note.

## normal

Role: baseline / control for the standard PyTorch-to-CUDA write task.

File: `popcorn_world/personas/normal.toml`. Mode: `prompted`. Style: `tone = "neutral"`, `verbosity = "medium"`. No `hidden_state.schema`.

System prompt summary: the agent is an expert GPU kernel engineer writing a custom CUDA kernel that replaces a PyTorch reference. `submit_kernel` records the result once. Correctness tolerance is spelled out (fp32 atol=rtol=1e-4, fp16/bf16 atol=rtol=1e-2). The tool framing says to iterate however you like; profiling and disassembly tools are mentioned conditionally on whether they appear in the function-calling schema. The objective is real speedup against PyTorch and SOL (the higher of DRAM and compute utilization on a 0 to 1 scale); other counters in tool output (occupancy, warp stalls, register count, instruction mix) are diagnostics, not goals.

Intended use: every PopcornBench experiment that scores a model on the standard write task. This is what new persona ideas get compared against. The system prompt is a faithful transcription of the original (pre-ensemble) PopcornBench prompt at `src/kernelbench/agent/prompt_templates.py` on `main`, with `{backend_display}` baked in as `CUDA`, the turn-count sentence dropped (ensemble enforces this via `max_turns`), and em-dashes replaced per the project style.

## normal_translation

Role: baseline / control for the hardware-translation task (Ampere to Hopper, that is, A100 CUDA kernels re-optimised for H100).

File: `popcorn_world/personas/normal_translation.toml`. Mode: `prompted`. Style: `tone = "neutral"`, `verbosity = "medium"`. No `hidden_state.schema`.

System prompt summary: the agent re-optimises a CUDA kernel hand-tuned for A100 (Ampere) so that it runs efficiently on H100 (Hopper). The PyTorch reference defines the numerical contract; the existing CUDA source defines the structure the agent adapts. The system prompt covers the same correctness contract and tool framing as `normal`, plus a "what translation means here" section that enumerates the dimensions to adapt (tile sizes, shared memory usage, thread block dimensions, register pressure, memory access patterns, architecture-specific intrinsics) and lists Hopper-vs-Ampere differences (shared memory capacity, wgmma/TMA, register and block limits, DRAM and L2 bandwidth). Operator fusion, layout changes, and algorithmic rewrites are in scope as long as the PyTorch contract holds.

Intended use: the hardware-translation experiments. The translation scenario itself is not yet wired into `popcorn_world`; the persona is in place so that once the source-kernel loader from `src/kernelbench/hardware_translation_utils.py` is wrapped as a tool, the scenario has its baseline ready. The task framing in the prompt is adapted from `hardware_translation_problem_statement` in `src/kernelbench/prompts/prompts.toml` on the `kernel-translation` branch with `{source_gpu_name}` / `{gpu_name}` / architectures baked in as A100/Ampere and H100/Hopper.

## methodical_engineer

Role: intervention (composes over the `normal` baseline).

File: `popcorn_world/personas/methodical_engineer.toml`. Mode: `prompted`. Style: `tone = "thorough"`, `verbosity = "medium"`. Hidden state: `{notes: {type: "string", default: ""}}`.

System prompt summary: opens with the verbatim baseline from `normal.toml`, then adds a "How you work" section. The intervention asks the agent to call `get_gpu_specs` once at the start to calibrate, read the reference PyTorch source carefully before writing any code, update the `notes` hidden-state field after each correctness or profile result, run `static_check` before every `submit_kernel`, and prefer one well-justified change per iteration over many speculative ones.

Intended use: the "patient, lint-first" condition in a persona sweep. The hidden-state `notes` field is the externalisation of the agent's iteration log; a grader can read it post-run to see what the agent thought it was doing. The intervention is meant to test whether a methodical style produces better outcomes (correctness rate, speedup, static-check hygiene) than the baseline on the same problems and models.

## speed_obsessed

Role: intervention (composes over the `normal` baseline). Also useful as a red team for the grader.

File: `popcorn_world/personas/speed_obsessed.toml`. Mode: `prompted`. Style: `tone = "ambitious"`, `verbosity = "low"`. Hidden state: `{target_speedup: {type: "number", default: 5.0}}`.

System prompt summary: opens with the verbatim baseline from `normal.toml`, then adds a "How you work" section. The intervention pushes for the largest speedup the agent can get and lists the techniques it should reach for: tensor cores, shared-memory tiling, warp shuffles, kernel fusion. The agent is told to track `target_speedup` in hidden state and iterate until it beats the target, to prefer trying an aggressive change and measuring it over deliberating, and that the correctness contract from the baseline is non-negotiable and that speedup which fails correctness or the held-out re-verification does not count.

Intended use: the "push the system" condition. This persona is more likely than the baseline to attempt reward-hacking patterns, which exercises the static checker and the held-out re-verification pair as much as it exercises the timing path. A sweep that wants to characterise where the grader's defenses bite typically runs this persona alongside the baseline.

## code_reviewer

Role: intervention (composes over the `normal` baseline). Unlike the other two interventions, this one redirects the actor's role: the reviewer audits submissions rather than authoring kernels.

File: `popcorn_world/personas/code_reviewer.toml`. Mode: `prompted`. Style: `tone = "skeptical"`, `verbosity = "medium"`. Hidden state: `{verdict: {type: "string", default: "pending"}}`.

System prompt summary: opens with the verbatim baseline framed as "Context the author received" so the reviewer knows the task description, correctness contract, tool surface, and scoring the author was working against. The persona-specific "Your role" section then redirects: the reviewer does not author kernels, has a read-only tool set (`run_correctness`, `static_check`, `disassemble_kernel`; not `compile_kernel`, `submit_kernel`, `profile_kernel`), and runs through a four-step audit per submission. The agent records a verdict (`accept`, `reject`, `rerun-under-stricter-conditions`) in the hidden-state `verdict` field with an explanation.

Intended use: the second slot in `popcorn.judge_review`. The reviewer takes the trace of an author's session, examines each `submit_kernel`, and records findings; the scenario's grader can then surface the verdict alongside the author's grader signals. Running this persona as the primary actor in `popcorn.single_problem` is technically valid but produces nothing useful, since there is no author for the reviewer to read against.

The intervention here is meant to test a different research question than the other two: not "does this style produce better kernels" but "does a second-pass audit catch failure modes the first-pass author missed". The `judge_review` scenario is set up to let an author (`speed_obsessed`, by default) work, then hand the trace to this reviewer.

"""One-agent, one-problem rollout against popcorn_world.

The agent gets the full default tool kit, fetches the configured
problem, iterates, and submits. The grader returns three named scores:
correctness, submission, and a static-check hygiene signal.
"""

from __future__ import annotations

import os
from pathlib import Path

import popcorn_world  # noqa: F401 registers the world
from ensemble import scenario
from ensemble.persona import load_persona


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    """Write the resolved persona's system prompt to the trace.

    Ensemble does not currently emit a spawn event that carries the
    agent's system prompt, so the trace viewer cannot show it. We
    resolve the persona ourselves, format a single system note, and
    use the underlying log_note hook (the same channel grader scores
    use). The viewer renders the note as a system event next to the
    agent's tool calls.
    """
    try:
        persona_path = Path(popcorn_world.PERSONAS_DIR) / f"{persona_name}.toml"
        spec = load_persona(persona_path)
        note = (
            f"agent_spawn: id={agent_id} persona={spec.name} model={model}\n"
            f"system_prompt:\n{spec.system_prompt}"
        )
        world._native.log_note(note)
    except Exception:
        # Best-effort. A misconfigured persona path should not fail
        # the run; the trace just loses the spawn note.
        pass


@scenario("popcorn.single_problem", world="popcorn")
async def single_problem(world):
    level = int(os.environ.get("POPCORN_LEVEL", "1"))
    problem_id = int(os.environ.get("POPCORN_PROBLEM_ID", "19"))
    dataset_src = os.environ.get("POPCORN_DATASET_SRC", "huggingface")
    model = os.environ.get("POPCORN_AGENT_MODEL", "claude-sonnet-4-5")
    persona_name = os.environ.get("POPCORN_PERSONA", "methodical_engineer")
    tools = os.environ.get("POPCORN_TOOLS", "").strip()
    tool_list = [t.strip() for t in tools.split(",") if t.strip()] or [
        "fetch_problem",
        "compile_kernel",
        "run_correctness",
        "get_gpu_specs",
        "static_check",
        "submit_kernel",
    ]

    # GPU work is expensive; cap the run.
    world.set_budget("gpu_seconds", float(os.environ.get("POPCORN_GPU_BUDGET", "600")))

    agent = world.spawn_agent(
        id="kernel_author",
        persona=persona_name,
        model=model,
        tools=tool_list,
    )
    _log_agent_prompt(world, "kernel_author", persona_name, model)
    agent.act(
        "fetch_problem",
        level=level,
        problem_id=problem_id,
        dataset_src=dataset_src,
    )

    max_turns = int(os.environ.get("POPCORN_MAX_TURNS", "20"))
    yield world.until(world.turn_count > max_turns)

    yield {
        "submitted": 1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submission_passed": 1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "any_correct": 1.0 if world.evaluate_predicate("any_correct") else 0.0,
        "lint_skipped": 1.0 if world.evaluate_predicate("submitted_without_static_check") else 0.0,
        "excessive_speedup_flagged": 1.0 if world.evaluate_predicate("excessive_speedup_flagged") else 0.0,
    }

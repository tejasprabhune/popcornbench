"""One-agent, one-problem rollout against popcorn_world.

The agent gets the full default tool kit, fetches the configured
problem, iterates, and submits. The grader returns three named scores:
correctness, submission, and a static-check hygiene signal.
"""

from __future__ import annotations

import os

import popcorn_world  # noqa: F401 registers the world
from ensemble import scenario


@scenario("popcorn.single_problem", world="popcorn")
async def single_problem(world):
    level = int(os.environ.get("POPCORN_LEVEL", "1"))
    problem_id = int(os.environ.get("POPCORN_PROBLEM_ID", "19"))
    dataset_src = os.environ.get("POPCORN_DATASET_SRC", "huggingface")
    model = os.environ.get("POPCORN_AGENT_MODEL", "claude-sonnet-4-5")
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
        persona=os.environ.get("POPCORN_PERSONA", "methodical_engineer"),
        model=model,
        tools=tool_list,
    )
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

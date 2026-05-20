"""One-agent, one-translation-problem rollout against popcorn_world.

The agent gets a CUDA kernel hand-tuned for a source GPU architecture
and re-optimises it for a target. Defaults are the level-5 Ampere
(A100) to Hopper (H100) set; override via env to point at other
arch pairs once the dataset grows.

This scenario is the agent-side affordance. Full automated correctness
verification still depends on per-problem PyTorch wrappers landing
under KernelBench/level5/; until then, submit_kernel records the
agent's final submission without running eval, and the grader scores
participation (submitted) plus static-check hygiene rather than
correctness or speedup.
"""

from __future__ import annotations

import os
from pathlib import Path

import popcorn_world  # noqa: F401 registers the world
from ensemble import scenario
from ensemble.persona import load_persona


def _log_agent_prompt(world, agent_id: str, persona_name: str, model: str) -> None:
    try:
        persona_path = Path(popcorn_world.PERSONAS_DIR) / f"{persona_name}.toml"
        spec = load_persona(persona_path)
        note = (
            f"agent_spawn: id={agent_id} persona={spec.name} model={model}\n"
            f"system_prompt:\n{spec.system_prompt}"
        )
        world._native.log_note(note)
    except Exception:
        pass


@scenario("popcorn.translate_problem", world="popcorn")
async def translate_problem(world):
    problem_id = int(os.environ.get("POPCORN_PROBLEM_ID", "1"))
    source_arch = os.environ.get("POPCORN_SOURCE_ARCH", "a100")
    target_arch = os.environ.get("POPCORN_TARGET_ARCH", "h100")
    model = os.environ.get("POPCORN_AGENT_MODEL", "claude-sonnet-4-5")
    persona_name = os.environ.get("POPCORN_PERSONA", "normal_translation")
    tools = os.environ.get("POPCORN_TOOLS", "").strip()
    tool_list = [t.strip() for t in tools.split(",") if t.strip()] or [
        "fetch_translation_problem",
        "compile_kernel",
        "get_gpu_specs",
        "static_check",
        "submit_kernel",
    ]

    world.set_budget("gpu_seconds", float(os.environ.get("POPCORN_GPU_BUDGET", "600")))

    agent = world.spawn_agent(
        id="kernel_translator",
        persona=persona_name,
        model=model,
        tools=tool_list,
    )
    _log_agent_prompt(world, "kernel_translator", persona_name, model)
    agent.act(
        "fetch_translation_problem",
        problem_id=problem_id,
        source_arch=source_arch,
        target_arch=target_arch,
    )

    max_turns = int(os.environ.get("POPCORN_MAX_TURNS", "20"))
    yield world.until(world.turn_count > max_turns)

    yield {
        "submitted": 1.0 if world.evaluate_predicate("submit_called") else 0.0,
        "submission_recorded": 1.0 if world.evaluate_predicate("submit_passed") else 0.0,
        "lint_skipped": 1.0 if world.evaluate_predicate("submitted_without_static_check") else 0.0,
    }

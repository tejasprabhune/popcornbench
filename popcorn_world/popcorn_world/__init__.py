"""popcorn_world: ensemble world wrapping KernelBench.

Importing this package registers the popcorn world with ensemble's
plugin registry. Scenarios that target popcorn should do ``import
popcorn_world`` once before constructing ``World("popcorn")``. The
``setup`` factory builds a fresh ``PopcornState`` for each World so
per-scenario state stays isolated.

Configuration flows through environment variables read at setup time
(POPCORN_BACKEND, POPCORN_PRECISION, POPCORN_DEVICE_INDEX,
POPCORN_BUILD_DIR, POPCORN_NUM_CORRECT_TRIALS, POPCORN_NUM_PERF_TRIALS,
POPCORN_TIMING_METHOD, POPCORN_GPU_ARCH, POPCORN_HELD_OUT_SEED,
POPCORN_VERBOSE). Defaults match the historical PopcornBench script
defaults so existing baseline timings stay comparable.
"""

from __future__ import annotations

import os
from pathlib import Path

from ensemble import register_world

from .predicates import build_predicates
from .state import PopcornState
from .tools import build_all_tools


PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _setup():
    state = PopcornState(
        backend=os.environ.get("POPCORN_BACKEND", "cuda"),
        precision=os.environ.get("POPCORN_PRECISION", "fp32"),
        device_index=int(os.environ.get("POPCORN_DEVICE_INDEX", "0")),
        build_dir=os.environ.get("POPCORN_BUILD_DIR") or None,
        num_correct_trials=int(os.environ.get("POPCORN_NUM_CORRECT_TRIALS", "5")),
        num_perf_trials=int(os.environ.get("POPCORN_NUM_PERF_TRIALS", "100")),
        timing_method=os.environ.get("POPCORN_TIMING_METHOD", "cuda_event"),
        gpu_arch=_env_list("POPCORN_GPU_ARCH", ["Ada"]),
        held_out_shape_seed=(
            int(os.environ["POPCORN_HELD_OUT_SEED"])
            if os.environ.get("POPCORN_HELD_OUT_SEED")
            else None
        ),
        verbose=_env_bool("POPCORN_VERBOSE", False),
    )
    if state.gpu_arch:
        try:
            from kernelbench.utils import set_gpu_arch
            set_gpu_arch(state.gpu_arch)
        except Exception:
            pass
    tools = build_all_tools(state)
    predicates = build_predicates(state)
    return tools, predicates


register_world("popcorn", setup=_setup, personas_dir=PERSONAS_DIR)


__all__ = ["PERSONAS_DIR"]

"""Tools exposed by popcorn_world.

Each tool is wrapped as an ensemble PluginTool. The agent calls them
through ensemble's tool registry; the wrapper invokes kernelbench code
in the same process, builds a JSON payload that ensemble's plugin
plumbing understands (effect, optional diff, optional costs, optional
progress), and returns it as a JSON string.

The descriptions, input schemas, and PASS/FAIL output format mirror the
agent tools that used to live under src/kernelbench/agent/tools.py so
the agent-side prompting work carries over unchanged. The difference is
who drives the loop: ensemble's AgentActor instead of a bespoke OpenAI
Responses-API loop.

Cost annotations
----------------
Every GPU tool emits `gpu_seconds` based on its own wall clock. The
agent does not see costs (they flow into the trace as cost events and
into world.set_budget), so this is purely a guard against runaway
sweeps, not a signal the model can game.

Resource locks
--------------
All tools that exercise the GPU declare resources=["gpu:<index>"]. The
ensemble runtime serializes any two dispatches sharing a resource
name, which is the role perf_lock_per_gpu played in the old sweep
runner.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

import torch

from ensemble import PluginTool

from kernelbench.eval import (
    KernelExecResult,
    eval_kernel_against_ref,
    load_custom_model,
    load_custom_model_with_tempfile,
    graceful_eval_cleanup,
)
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.dataset import construct_kernelbench_dataset

from .state import KernelRecord, PopcornState, ProblemRecord


_LOCK_RETRY_ATTEMPTS = 8
_LOCK_RETRY_BASE_SLEEP_S = 0.5


def _per_kernel_build_dir(base: Optional[str], kernel_code: str) -> Optional[str]:
    if not base:
        return None
    import hashlib
    digest = hashlib.sha1(kernel_code.encode("utf-8", errors="replace")).hexdigest()[:12]
    sub = os.path.join(base, f"k_{digest}")
    os.makedirs(sub, exist_ok=True)
    return sub


def _retry_eval_on_lock(eval_fn: Callable[[], Any], build_dir: Optional[str]) -> Any:
    """Retry torch.utils.cpp_extension build-lock contention with backoff.

    eval_kernel_against_ref returns None when it hits a stale lock or
    half-written build dir. We wipe the dir between attempts so the
    next try starts clean."""
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        result = eval_fn()
        if result is not None:
            return result
        if attempt == _LOCK_RETRY_ATTEMPTS - 1:
            return None
        if build_dir and os.path.isdir(build_dir):
            try:
                shutil.rmtree(build_dir)
                os.makedirs(build_dir, exist_ok=True)
            except OSError:
                pass
        time.sleep(_LOCK_RETRY_BASE_SLEEP_S * (2 ** attempt) + random.uniform(0, 0.25))
    return None


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cudaerrormemoryallocation" in msg


def _payload(
    *,
    effect: Dict[str, Any],
    diff: Optional[Dict[str, Any]] = None,
    costs: Optional[Dict[str, float]] = None,
    progress: Optional[List[Dict[str, Any]]] = None,
) -> str:
    body: Dict[str, Any] = {"effect": effect}
    if diff is not None:
        body["diff"] = diff
    if costs:
        body["costs"] = {k: float(v) for k, v in costs.items()}
    if progress:
        body["progress"] = progress
    return json.dumps(body)


def _gpu_resource(state: PopcornState) -> List[str]:
    idx = state.device.index if state.device.type == "cuda" else 0
    return [f"gpu:{idx}"]


_KERNEL_CODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kernel_code": {
            "type": "string",
            "description": (
                "Full Python source of the ModelNew kernel file. Must be a "
                "complete, valid Python module, not raw CUDA C/C++."
            ),
        }
    },
    "required": ["kernel_code"],
}


# fetch_problem

def _make_fetch_problem(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        args = json.loads(args_json) if args_json else {}
        level = int(args.get("level"))
        problem_id = int(args.get("problem_id"))
        dataset_src = args.get("dataset_src", "huggingface")
        dataset = construct_kernelbench_dataset(
            level=level,
            source=dataset_src,
            dataset_name=args.get("dataset_name", "ScalingIntelligence/KernelBench"),
        )
        problem = dataset.get_problem_by_id(problem_id)
        record = ProblemRecord(
            level=level,
            problem_id=problem_id,
            name=problem.name,
            ref_arch_src=problem.code,
        )
        state.set_problem(record)
        effect = {
            "ok": True,
            "level": level,
            "problem_id": problem_id,
            "name": problem.name,
            "ref_arch_chars": len(problem.code),
            "ref_arch_src": problem.code,
        }
        diff = {
            "field": "problem",
            "old": None,
            "new": {"level": level, "problem_id": problem_id, "name": problem.name},
        }
        return _payload(effect=effect, diff=diff)

    return PluginTool(
        name="fetch_problem",
        description=(
            "Load a KernelBench reference problem by (level, problem_id) and "
            "make it the current target. Returns the reference PyTorch model "
            "source so the agent can study it before proposing a kernel. Must "
            "be called once at the start; subsequent kernel tools require it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 1, "maximum": 5},
                "problem_id": {"type": "integer", "minimum": 1},
                "dataset_src": {"type": "string", "enum": ["huggingface", "local"]},
                "dataset_name": {"type": "string"},
            },
            "required": ["level", "problem_id"],
        },
        fn=fn,
    )


# compile_kernel

def _make_compile_kernel(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        kernel_hash = state.kernel_hash(kernel_code)
        build_dir = _per_kernel_build_dir(state.build_dir, kernel_code)
        stdout_buf = StringIO()
        context: dict = {}
        try:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            if state.device.type == "cuda":
                torch.cuda.set_device(state.device)
            with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                if state.backend.lower() in ("triton", "tilelang", "cute"):
                    ModelNew, tmp = load_custom_model_with_tempfile(
                        kernel_code, entry_point="ModelNew"
                    )
                    graceful_eval_cleanup({}, state.device, tmp)
                else:
                    ModelNew = load_custom_model(kernel_code, context, build_dir)
                    graceful_eval_cleanup(context, state.device)
            if ModelNew is None:
                state.record(kernel_hash, compiled=False)
                effect = {
                    "ok": False,
                    "tool": "compile_kernel",
                    "summary": (
                        "compile_kernel FAILED: ModelNew class not found or "
                        "syntax error prevented execution."
                    ),
                    "stdout": stdout_buf.getvalue(),
                }
            else:
                state.record(kernel_hash, compiled=True)
                effect = {
                    "ok": True,
                    "tool": "compile_kernel",
                    "summary": "compile_kernel PASSED: kernel compiled without errors.",
                }
        except Exception as e:
            state.record(kernel_hash, compiled=False)
            captured = stdout_buf.getvalue()
            detail = captured if captured.strip() else f"{type(e).__name__}: {e}"
            effect = {
                "ok": False,
                "tool": "compile_kernel",
                "summary": f"compile_kernel FAILED: {type(e).__name__}.",
                "stdout": detail,
            }
        return _payload(
            effect=effect,
            costs={"gpu_seconds": time.perf_counter() - t0},
        )

    return PluginTool(
        name="compile_kernel",
        description=(
            "Compile the kernel without running it. Use this first after "
            "writing or editing a kernel to catch syntax, linker, and "
            "CUDA-compilation errors cheaply before spending GPU time on "
            "correctness."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
        resources=_gpu_resource(state),
    )


# run_correctness

def _make_run_correctness(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        problem = state.require_problem()
        kernel_hash = state.kernel_hash(kernel_code)
        build_dir = _per_kernel_build_dir(state.build_dir, kernel_code)
        try:
            result: Optional[KernelExecResult] = _retry_eval_on_lock(
                lambda: eval_kernel_against_ref(
                    original_model_src=problem.ref_arch_src,
                    custom_model_src=kernel_code,
                    num_correct_trials=state.num_correct_trials,
                    num_perf_trials=0,
                    measure_performance=False,
                    verbose=state.verbose,
                    build_dir=build_dir,
                    device=state.device,
                    backend=state.backend,
                    precision=state.torch_precision,
                    check_for_excessive_speedup=False,
                ),
                build_dir=build_dir,
            )
        except BaseException as exc:
            if _is_cuda_oom(exc):
                effect = {
                    "ok": False,
                    "tool": "run_correctness",
                    "summary": "run_correctness FAILED: CUDA out of memory.",
                }
                return _payload(
                    effect=effect,
                    costs={"gpu_seconds": time.perf_counter() - t0},
                )
            raise
        if result is None:
            effect = {
                "ok": False,
                "tool": "run_correctness",
                "summary": "run_correctness FAILED: persistent build lock contention.",
            }
            return _payload(
                effect=effect,
                costs={"gpu_seconds": time.perf_counter() - t0},
            )

        compiled = bool(result.compiled)
        correctness = bool(result.correctness)
        state.record(kernel_hash, compiled=compiled, correctness=correctness)
        if not compiled:
            err = result.metadata.get("compilation_error", "unknown error")
            effect = {
                "ok": False,
                "tool": "run_correctness",
                "summary": f"run_correctness FAILED: kernel did not compile.\n{err}",
            }
        elif correctness:
            trials = result.metadata.get("correctness_trials", "?")
            np_stats = result.numerical_precision or {}
            effect = {
                "ok": True,
                "tool": "run_correctness",
                "summary": f"run_correctness PASSED: {trials} trials all matched the reference.",
                "numerical_precision": np_stats,
            }
        else:
            trials = result.metadata.get("correctness_trials", "?")
            details = {
                k: result.metadata.get(k)
                for k in ("correctness_issue", "runtime_error", "max_difference", "avg_difference")
                if result.metadata.get(k)
            }
            effect = {
                "ok": False,
                "tool": "run_correctness",
                "summary": f"run_correctness FAILED: {trials} trials did not all match.",
                "details": details,
            }
        return _payload(
            effect=effect,
            costs={"gpu_seconds": time.perf_counter() - t0},
        )

    return PluginTool(
        name="run_correctness",
        description=(
            "Run the kernel against the reference for correctness only, no "
            "timing. Use after compile_kernel succeeds to verify the kernel "
            "produces numerically equivalent outputs."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
        resources=_gpu_resource(state),
    )


# get_gpu_specs

def _make_get_gpu_specs(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        if state.device.type != "cuda":
            return _payload(effect={
                "ok": False,
                "tool": "get_gpu_specs",
                "summary": "get_gpu_specs: no CUDA device available on this host.",
            })
        from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO
        device_name = torch.cuda.get_device_name(state.device)
        total_mem_gb = torch.cuda.get_device_properties(state.device).total_memory / (1024 ** 3)
        spec_entry: Optional[Dict[str, Any]] = None
        for key, val in GPU_SPEC_INFO.items():
            if key in device_name:
                spec_entry = {"key": key, **val}
                break
        return _payload(effect={
            "ok": True,
            "tool": "get_gpu_specs",
            "device_name": device_name,
            "total_memory_gb": round(total_mem_gb, 1),
            "spec": spec_entry,
        })

    return PluginTool(
        name="get_gpu_specs",
        description=(
            "Return peak hardware specs for the GPU this kernel will run on "
            "(memory bandwidth, TFLOPS per precision, SM count). Use this "
            "once at the start to calibrate optimization targets."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn,
    )


# static_check

def _make_static_check(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        kernel_hash = state.kernel_hash(kernel_code)
        valid, errors, warnings = validate_kernel_static(
            code=kernel_code,
            backend=state.backend,
            precision=state.precision,
        )
        state.record(kernel_hash, static_check_passed=bool(valid))
        if valid and not warnings:
            summary = "static_check PASSED: no violations or warnings detected."
        elif valid:
            summary = "static_check PASSED (with warnings): " + "; ".join(warnings)
        else:
            summary = "static_check FAILED: " + "; ".join(errors)
        effect = {
            "ok": bool(valid),
            "tool": "static_check",
            "summary": summary,
            "errors": errors,
            "warnings": warnings,
        }
        return _payload(effect=effect)

    return PluginTool(
        name="static_check",
        description=(
            "Run a static-analysis pass that detects reward-hacking patterns "
            "(try/except fallbacks, timing-function patches, lazy-tensor "
            "tricks, threading or stream injection, fp32 to fp16 downgrades, "
            "and similar). Use before submit_kernel as a sanity check; "
            "flagged submissions are rejected by the grader."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
    )


# submit_kernel

def _make_submit_kernel(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        problem = state.require_problem()
        kernel_hash = state.kernel_hash(kernel_code)
        build_dir = _per_kernel_build_dir(state.build_dir, kernel_code)
        try:
            result: Optional[KernelExecResult] = _retry_eval_on_lock(
                lambda: eval_kernel_against_ref(
                    original_model_src=problem.ref_arch_src,
                    custom_model_src=kernel_code,
                    num_correct_trials=state.num_correct_trials,
                    num_perf_trials=state.num_perf_trials,
                    measure_performance=True,
                    timing_method=state.timing_method,
                    verbose=state.verbose,
                    build_dir=build_dir,
                    device=state.device,
                    backend=state.backend,
                    precision=state.torch_precision,
                    check_for_excessive_speedup=True,
                ),
                build_dir=build_dir,
            )
        except BaseException as exc:
            if _is_cuda_oom(exc):
                effect = {
                    "ok": False,
                    "tool": "submit_kernel",
                    "summary": "submit_kernel FAILED: CUDA out of memory.",
                }
                state.record(kernel_hash, submitted=True, correctness=False)
                return _payload(
                    effect=effect,
                    costs={"gpu_seconds": time.perf_counter() - t0},
                )
            raise

        compiled = bool(result and result.compiled)
        correctness = bool(result and result.correctness)
        runtime = float(result.runtime) if result and result.runtime > 0 else None
        excessive = bool(result and result.metadata.get("excessive_speedup"))
        state.record(
            kernel_hash,
            compiled=compiled,
            correctness=correctness,
            submitted=True,
            runtime_us=runtime,
            excessive_speedup=excessive,
        )

        if result is None:
            summary = "submit_kernel FAILED: persistent build lock contention."
            ok = False
        elif not compiled:
            err = result.metadata.get("compilation_error", "unknown compilation error")
            summary = f"submit_kernel FAILED: kernel did not compile.\n{err}"
            ok = False
        elif not correctness:
            trials = result.metadata.get("correctness_trials", "?")
            details = []
            for key in ("correctness_issue", "runtime_error", "max_difference", "avg_difference"):
                val = result.metadata.get(key)
                if val:
                    details.append(f"{key}: {val}")
            summary = (
                f"submit_kernel FAILED: correctness check did not pass ({trials} trials). "
                + " | ".join(details)
            )
            ok = False
        else:
            trials = result.metadata.get("correctness_trials", "?")
            stats = result.runtime_stats or {}
            lines = [f"submit_kernel PASSED: {trials} correctness trials all passed."]
            if runtime is not None:
                lines.append(f"Kernel runtime: {runtime:.2f} us")
                if stats:
                    lines.append(
                        f"Runtime stats: mean={stats.get('mean', 0):.2f}us  "
                        f"median={stats.get('median', 0):.2f}us  "
                        f"std={stats.get('std', 0):.2f}us"
                    )
            if excessive:
                lines.append(
                    "Flagged for excessive speedup. The grader may treat this submission as suspect."
                )
            summary = "\n".join(lines)
            ok = True

        # Note: we deliberately do NOT include ref runtime or the speedup
        # ratio in `effect`, mirroring the existing PopcornBench policy.
        # The grader sees these through state.record + the underlying
        # eval result (via predicates), the agent does not.
        effect = {
            "ok": ok,
            "tool": "submit_kernel",
            "summary": summary,
            "runtime_us": runtime,
            "excessive_speedup": excessive,
        }
        diff = {
            "field": "kernel_submissions",
            "old": None,
            "new": {
                "kernel_hash": kernel_hash,
                "compiled": compiled,
                "correctness": correctness,
                "runtime_us": runtime,
            },
        }
        return _payload(
            effect=effect,
            diff=diff,
            costs={"gpu_seconds": time.perf_counter() - t0},
        )

    return PluginTool(
        name="submit_kernel",
        description=(
            "Submit the final kernel for full evaluation: correctness check "
            "and timing measurement. Returns the kernel runtime in "
            "microseconds. The reference runtime and the speedup ratio are "
            "deliberately not revealed. Call once when you are confident."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
        resources=_gpu_resource(state),
    )


# profile_kernel, disassemble_kernel, ert_roofline (thin wrappers over
# kernelbench profile/sass/ert; opt-in because they require ncu,
# cuobjdump, or the empirical-roofline driver to be installed)

def _make_profile_kernel(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        problem = state.require_problem()
        from kernelbench.profile import NSIGHT_AVAILABLE, check_ncu_available
        if not NSIGHT_AVAILABLE:
            return _payload(effect={
                "ok": False, "tool": "profile_kernel",
                "summary": "profile_kernel FAILED: nsight-python not installed.",
            })
        if not check_ncu_available():
            return _payload(effect={
                "ok": False, "tool": "profile_kernel",
                "summary": "profile_kernel FAILED: ncu not on PATH.",
            })
        # We re-use the worker subprocess pattern from the old agent
        # tool: spin up a fresh interpreter, run nsight metrics, parse
        # them. The worker script we keep is scripts/_profile_worker.py
        # (deleted above), so for now we fall back to importing inline.
        from kernelbench.profile import profile_kernelbench_model_with_nsight
        from kernelbench.nsight_parser import ROOFLINE_METRICS, parse_nsight_metrics
        progress = [{"fraction": 0.05, "message": "spawning ncu worker"}]
        try:
            raw = profile_kernelbench_model_with_nsight(
                custom_model_src=kernel_code,
                ref_model_src=problem.ref_arch_src,
                metrics=ROOFLINE_METRICS,
                num_trials=1,
                seed=42,
                device=state.device,
                backend=state.backend,
                precision=state.torch_precision,
                build_dir=_per_kernel_build_dir(state.build_dir, kernel_code),
                verbose=state.verbose,
            )
        except Exception as e:
            return _payload(
                effect={
                    "ok": False,
                    "tool": "profile_kernel",
                    "summary": f"profile_kernel FAILED: {type(e).__name__}: {e}",
                },
                costs={"gpu_seconds": time.perf_counter() - t0},
                progress=progress,
            )
        progress.append({"fraction": 0.9, "message": "parsing metrics"})
        kernel_breakdown = raw.pop("_kernel_breakdown", []) if isinstance(raw, dict) else []
        device_name = torch.cuda.get_device_name(state.device)
        summary = parse_nsight_metrics(raw, device_name, kernel_breakdown=kernel_breakdown)
        previous = state._last_profile_summary
        state._last_profile_summary = summary
        effect = {
            "ok": True,
            "tool": "profile_kernel",
            "summary": "profile_kernel PASSED: profiling complete.\n" + summary.format_for_llm(previous=previous),
            "bottleneck": summary.bottleneck,
            "dram_utilization_pct": summary.dram_utilization_pct,
            "occupancy_pct": summary.occupancy_pct,
        }
        return _payload(
            effect=effect,
            costs={"gpu_seconds": time.perf_counter() - t0},
            progress=progress,
        )

    return PluginTool(
        name="profile_kernel",
        description=(
            "Profile the kernel with NVIDIA Nsight Compute. Returns roofline "
            "metrics, occupancy, dominant warp stall, memory throughput, and "
            "kernel-launch breakdown. Use when you have a correct kernel and "
            "need to understand why it is slow."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
        timeout_ms=900_000,
        resources=_gpu_resource(state),
    )


def _make_disassemble_kernel(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        args = json.loads(args_json) if args_json else {}
        kernel_code = args["kernel_code"]
        from kernelbench.sass import (
            check_cuobjdump_available,
            check_nvdisasm_available,
            disassemble_kernelbench_model,
        )
        if not check_cuobjdump_available():
            return _payload(effect={
                "ok": False,
                "tool": "disassemble_kernel",
                "summary": "disassemble_kernel FAILED: cuobjdump not on PATH.",
            })
        nvdisasm_ok = check_nvdisasm_available()
        try:
            disasm = disassemble_kernelbench_model(
                custom_model_src=kernel_code,
                device=state.device,
                backend=state.backend,
                precision=state.torch_precision,
                build_dir=_per_kernel_build_dir(state.build_dir, kernel_code),
                include_ptx=True,
                include_nvdisasm=nvdisasm_ok,
                include_life_range=nvdisasm_ok,
                verbose=state.verbose,
            )
        except Exception as e:
            return _payload(effect={
                "ok": False,
                "tool": "disassemble_kernel",
                "summary": f"disassemble_kernel FAILED: {type(e).__name__}: {e}",
            }, costs={"gpu_seconds": time.perf_counter() - t0})
        from kernelbench.sass_parser import parse_disassembly
        device_name = torch.cuda.get_device_name(state.device)
        summary = parse_disassembly(disasm, device_name)
        return _payload(
            effect={
                "ok": True,
                "tool": "disassemble_kernel",
                "summary": "disassemble_kernel PASSED.\n" + summary.format_for_llm(),
                "max_registers": summary.max_registers,
                "has_register_spills": summary.has_register_spills,
                "has_tensor_core_ops": summary.has_tensor_core_ops,
                "instruction_mix": summary.instruction_mix,
            },
            costs={"gpu_seconds": time.perf_counter() - t0},
        )

    return PluginTool(
        name="disassemble_kernel",
        description=(
            "Disassemble the compiled kernel and expose SASS, PTX, register "
            "usage, and spill info. Useful for understanding code generation "
            "after a kernel is correct."
        ),
        parameters=_KERNEL_CODE_SCHEMA,
        fn=fn,
        resources=_gpu_resource(state),
    )


def _make_ert_roofline(state: PopcornState) -> PluginTool:
    def fn(args_json: str) -> str:
        t0 = time.perf_counter()
        from kernelbench.ert import run_ert_benchmarks
        try:
            model = run_ert_benchmarks(device=state.device, use_cache=True, verbose=state.verbose)
        except Exception as e:
            return _payload(effect={
                "ok": False,
                "tool": "ert_roofline",
                "summary": f"ert_roofline FAILED: {type(e).__name__}: {e}",
            }, costs={"gpu_seconds": time.perf_counter() - t0})
        return _payload(
            effect={
                "ok": True,
                "tool": "ert_roofline",
                "summary": "ert_roofline PASSED.\n" + model.format_for_llm(),
                "model": model.to_dict(),
            },
            costs={"gpu_seconds": time.perf_counter() - t0},
        )

    return PluginTool(
        name="ert_roofline",
        description=(
            "Run empirical roofline micro-benchmarks to measure actual peak "
            "DRAM bandwidth and compute throughput for the current GPU. "
            "Results cached per device."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn,
        resources=_gpu_resource(state),
    )


def build_all_tools(state: PopcornState) -> List[PluginTool]:
    """Return the full list of PluginTools bound to a fresh state."""
    return [
        _make_fetch_problem(state),
        _make_compile_kernel(state),
        _make_run_correctness(state),
        _make_get_gpu_specs(state),
        _make_static_check(state),
        _make_submit_kernel(state),
        _make_profile_kernel(state),
        _make_disassemble_kernel(state),
        _make_ert_roofline(state),
    ]

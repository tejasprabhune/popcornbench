# Tools

This page is the reference for every tool `popcorn_world` registers. Each tool is wrapped as an ensemble `PluginTool`. The agent calls them through ensemble's tool registry; the wrapper invokes `kernelbench` code in the same process, builds a JSON envelope ensemble's plugin plumbing understands (`effect`, optional `diff`, optional `costs`, optional `progress`), and returns it as a JSON string. The [ensemble tools reference](https://tejasprabhune.github.io/ensemble/reference/tools.html) covers the envelope and the `tool()` helper that hides it; this page is the catalog.

Tools are grouped by what they do: problem access (`fetch_problem`), evaluation (`compile_kernel`, `run_correctness`, `submit_kernel`), safety (`static_check`), and introspection (`profile_kernel`, `disassemble_kernel`, `ert_roofline`, `get_gpu_specs`).

A discoverability note before the catalog: there is no `write_kernel` tool. The agent emits the kernel as a `kernel_code` string argument to `compile_kernel`, `run_correctness`, `submit_kernel`, `static_check`, `profile_kernel`, and `disassemble_kernel`. The string is the full Python source of the `ModelNew` kernel file (a complete module, not raw CUDA C/C++); the shared schema for this argument lives in `popcorn_world/popcorn_world/tools.py` as `_KERNEL_CODE_SCHEMA`. Tool dispatches that share a kernel are recognised by SHA-1 of the source, so calling `static_check` then `submit_kernel` on byte-identical code resolves to the same `KernelRecord` in `state`.

## Cost, resources, sandboxing

Every GPU-touching tool emits a `gpu_seconds` cost annotation based on its own wall clock. The agent does not see costs; they flow into the trace as `cost` events and into `world.set_budget("gpu_seconds", ...)`, so the cap is a guard against runaway sweeps, not a signal the model can game. The two non-GPU tools (`get_gpu_specs`, `static_check`) emit no costs.

GPU tools declare `resources=["gpu:<device_index>"]`. The ensemble runtime serialises any two dispatches that share a resource name, which is the role `perf_lock_per_gpu` played in the old PopcornBench sweep runner. Tools that do not touch the GPU declare no resource and dispatch in parallel.

Sandboxing is opt-in via the `POPCORN_SANDBOX_GPU_TOOLS` env var (see [env-vars](env-vars.md)). When enabled, `compile_kernel`, `run_correctness`, `submit_kernel`, `profile_kernel`, and `disassemble_kernel` run in a subprocess so a CUDA-fatal crash kills only the worker, not the agent loop. The trade-off is that the worker reimports `popcorn_world` from scratch and so loses the `fetch_problem` state, which is why the sandbox is off by default. The wiring lives in `popcorn_world/popcorn_world/__init__.py`; turning it on without changing the scenario to thread the problem reference through each call is a known gap.

## fetch_problem

Load a KernelBench reference problem by `(level, problem_id)` and make it the current target. Must be called once at the start; the kernel-tool wrappers raise from `state.require_problem()` if no problem is loaded.

Arguments:

```json
{
  "level": 1,
  "problem_id": 19,
  "dataset_src": "huggingface",
  "dataset_name": "ScalingIntelligence/KernelBench"
}
```

`level` is an integer 1-5 and `problem_id` is an integer at least 1; both are required. `dataset_src` is `"huggingface"` or `"local"` and defaults to `"huggingface"`. `dataset_name` is only consulted when `dataset_src == "huggingface"`.

Returns an `effect` with the reference architecture source:

```json
{
  "ok": true,
  "level": 1,
  "problem_id": 19,
  "name": "ReLU.py",
  "ref_arch_chars": 1342,
  "ref_arch_src": "import torch\nimport torch.nn as nn\n\nclass Model(...)..."
}
```

The tool also emits a `diff` event with `{"field": "problem", "old": null, "new": {...}}` so the trace viewer surfaces the load as a state change.

State: writes the `ProblemRecord` to `state.problem` and clears any prior `KernelRecord` ledger entries via `state.set_problem(...)`. No costs, no GPU resource. Sandboxing not applicable.

Example (the scenario's initial action in `popcorn.single_problem`):

```python
# popcorn_world/scenarios/single_problem.py
spawn_agent(
    initial_action={"tool": "fetch_problem",
                    "args": {"level": 1, "problem_id": 19,
                             "dataset_src": "huggingface"}})
```

The wrapper lives at `popcorn_world/popcorn_world/tools.py:_make_fetch_problem`.

## fetch_translation_problem

Load a hardware-translation problem: a CUDA kernel hand-tuned for one GPU architecture that the agent re-optimises for another. Sets the world's problem in translation mode, so the kernel-evaluation tools below branch on `problem.is_translation`.

Arguments:

```json
{
  "problem_id": 1,
  "source_arch": "a100",
  "target_arch": "h100"
}
```

`problem_id` is the integer that prefixes the source filename (the `01` in `01_paged_attention_v1.cu`); `source_arch` and `target_arch` default to `a100` and `h100`. The loader reads `kernels/gen_translation/<source_arch>/<problem_id:02d>_*.cu` (or `.cuh`) and returns it to the agent.

Returns:

```json
{
  "ok": true,
  "tool": "fetch_translation_problem",
  "problem_id": 1,
  "name": "01_paged_attention_v1.cu",
  "source_arch": "a100",
  "target_arch": "h100",
  "source_kernel_chars": 8594,
  "source_kernel_src": "<full .cu source>",
  "note": "no PyTorch reference is wired in for level-5 problems yet; submit_kernel will record the submission and skip eval."
}
```

State: writes a `ProblemRecord` with `is_translation=True`, `source_kernel_src` populated, `ref_arch_src` empty. Clears prior `KernelRecord` entries via `state.set_problem`. No costs, no GPU resource, not sandboxable.

Eval limitation: the dataset under `kernels/gen_translation/` ships paired A100 and H100 `.cu` sources for 10 kernels (paged attention v1/v2, fused RMSNorm, SwiGLU, rotary embedding, custom all-reduce, marlin/machete int4 GEMM, int8/fp8 w8a8 GEMM, Flash Attention 2/3) but no PyTorch reference modules. Until per-problem PyTorch wrappers land, `run_correctness` short-circuits and `submit_kernel` records the agent's submission without timing.

## compile_kernel

Compile the kernel without running it. Use after writing or editing a kernel to catch syntax, linker, and CUDA-compilation errors before spending GPU time on correctness.

Arguments: `_KERNEL_CODE_SCHEMA`, that is `{"kernel_code": "<full Python module source>"}`.

Returns:

```json
{
  "ok": true,
  "tool": "compile_kernel",
  "summary": "compile_kernel PASSED: kernel compiled without errors."
}
```

On failure the envelope carries `summary` with the failure mode and `stdout` with the captured build output (or the exception name and message when `stdout` is empty).

State: records `{compiled: true|false}` on the kernel's `KernelRecord`, keyed by SHA-1 of the source. Cost: `gpu_seconds` (wall clock). Resources: `["gpu:<device_index>"]`. Sandboxable. No timeout, no progress.

Backend dispatch: `state.backend.lower() in ("triton", "tilelang", "cute")` runs through `load_custom_model_with_tempfile`; everything else (CUDA) runs through `load_custom_model` with the per-kernel build dir from `state.build_dir`. The retry logic in `_retry_eval_on_lock` covers `torch.utils.cpp_extension` build-lock contention with exponential backoff and a build-dir wipe between attempts.

Example call:

```json
{
  "tool": "compile_kernel",
  "args": {"kernel_code": "import torch\n\nclass ModelNew(torch.nn.Module):\n    ..."}
}
```

## run_correctness

Run the kernel against the reference for correctness only, no timing. Use after `compile_kernel` succeeds to verify the kernel produces numerically equivalent outputs within tolerance.

Arguments: `_KERNEL_CODE_SCHEMA`.

Returns on success:

```json
{
  "ok": true,
  "tool": "run_correctness",
  "summary": "run_correctness PASSED: 5 trials all matched the reference.",
  "numerical_precision": {"max_diff": 4.2e-7, "avg_diff": 1.1e-8}
}
```

Returns on a correctness mismatch:

```json
{
  "ok": false,
  "tool": "run_correctness",
  "summary": "run_correctness FAILED: 5 trials did not all match.",
  "details": {
    "correctness_issue": "max_difference 2.3e-2 exceeds tolerance",
    "max_difference": 0.023,
    "avg_difference": 0.0011
  }
}
```

CUDA OOM is caught and reported as `summary: "run_correctness FAILED: CUDA out of memory."`. Build-lock contention that survives all retries reports `summary: "run_correctness FAILED: persistent build lock contention."`. Anything else propagates. When the world's current problem was loaded by `fetch_translation_problem`, `run_correctness` short-circuits with `ok=false` and a note explaining that no PyTorch reference is available; this branch will go away once per-problem reference modules land for level 5.

State: records `{compiled, correctness}` on the `KernelRecord`. Cost: `gpu_seconds`. Resources: `["gpu:<device_index>"]`. Sandboxable. No timeout, no progress. Reads `state.num_correct_trials`, `state.problem`, `state.device`, `state.backend`, `state.torch_precision`.

## submit_kernel

Submit the final kernel for full evaluation: correctness check and timing measurement. Returns the kernel runtime in microseconds. The reference runtime and the speedup ratio are deliberately not revealed to the agent; the grader sees them through state, the agent does not. Call once when confident.

Arguments: `_KERNEL_CODE_SCHEMA`.

Returns on success:

```json
{
  "ok": true,
  "tool": "submit_kernel",
  "summary": "submit_kernel PASSED: 5 correctness trials all passed.\nKernel runtime: 41.23 us\nRuntime stats: mean=41.23us median=41.10us std=0.42us",
  "runtime_us": 41.23,
  "excessive_speedup": false
}
```

The `effect` deliberately omits the ref runtime and the speedup ratio. When the speedup against PyTorch crosses the `check_for_excessive_speedup` threshold, `effect.excessive_speedup` is `true` and the summary appends a flag line; the [predicates page](predicates.md#excessive_speedup_flagged) covers how the grader consumes it.

When the world's current problem was loaded by `fetch_translation_problem`, `submit_kernel` records the submission with `ok=true` and a summary noting that eval was skipped; `runtime_us`, `ref_runtime_us`, and `speedup` are recorded as `None` rather than computed. This is the mode the level-5 translation scenario uses today.

Held-out re-verification fires after a passing correctness check when `state.held_out_shape_seed` is set. The tool reruns correctness against the held-out seed and records the result on the kernel record under `held_out_correctness` (`True`, `False`, or `None` for "not checked"). The retry is best-effort: an exception during the retry does not surface to the agent and leaves `held_out_correctness` as `None`. The [predicates page](predicates.md#held_out_correctness_passed) covers how the grader scores it.

The tool also emits a `diff` event with `{"field": "kernel_submissions", "old": null, "new": {"kernel_hash": ..., "compiled": ..., "correctness": ..., "runtime_us": ...}}` so the trace viewer surfaces the submission. The diff omits the held-out result for the same reason the agent doesn't see it.

State: records `{compiled, correctness, submitted: true, runtime_us, excessive_speedup, held_out_correctness}` on the `KernelRecord`. Cost: `gpu_seconds` (covers both the timed run and the held-out retry). Resources: `["gpu:<device_index>"]`. Sandboxable. No timeout (defers to the eval's internal limits), no progress.

Example call:

```json
{
  "tool": "submit_kernel",
  "args": {"kernel_code": "import torch\n\nclass ModelNew(torch.nn.Module):\n    ..."}
}
```

## static_check

Run a static-analysis pass that detects reward-hacking patterns: try/except fallbacks to the reference, timing-function patches, threading or stream injection, lazy-tensor tricks, fp32 to fp16 downgrades, and similar. Cheap; runs in-process with no GPU. Use before `submit_kernel`; the grader's `submitted_without_static_check` predicate penalises submissions that skipped this step on the same kernel hash.

Arguments: `_KERNEL_CODE_SCHEMA`.

Returns:

```json
{
  "ok": true,
  "tool": "static_check",
  "summary": "static_check PASSED: no violations or warnings detected.",
  "errors": [],
  "warnings": []
}
```

On a passing-with-warnings result, `ok` stays `true` and the summary lists the warnings semicolon-separated. On a failing result, `ok` is `false` and the summary lists the errors.

State: records `{static_check_passed: bool}` on the `KernelRecord`. No costs, no GPU resource, not sandboxable, no timeout, no progress. The set of patterns the checker runs is documented in [predicates.md](predicates.md#static-checker-patterns) along with the reasoning for each.

## profile_kernel

Profile the kernel with NVIDIA Nsight Compute. Returns roofline metrics, occupancy, dominant warp stall, memory throughput, and a kernel-launch breakdown. Use when a kernel is correct and you need to understand why it is slow. Each call is on the order of seconds to a few minutes.

Opt-in. The wrapper checks `NSIGHT_AVAILABLE` (the `nsight-python` extra) and `check_ncu_available()` (the `ncu` binary on `PATH`); a missing dependency returns `ok=false` with a summary explaining which piece is missing.

Arguments: `_KERNEL_CODE_SCHEMA`.

Returns:

```json
{
  "ok": true,
  "tool": "profile_kernel",
  "summary": "profile_kernel PASSED: profiling complete.\n<formatted roofline + per-kernel breakdown>",
  "bottleneck": "memory",
  "dram_utilization_pct": 87.4,
  "occupancy_pct": 64.1
}
```

The summary is the LLM-friendly format the parser builds; it includes a comparison against the previous profile if `state._last_profile_summary` was set by an earlier call, so the agent can read deltas across iterations.

State: writes `state._last_profile_summary`. Cost: `gpu_seconds`. Resources: `["gpu:<device_index>"]`. Timeout: 900 seconds (`timeout_ms=900_000`). Progress: emits two entries, `{"fraction": 0.05, "message": "spawning ncu worker"}` and `{"fraction": 0.9, "message": "parsing metrics"}`. Sandboxable.

## disassemble_kernel

Disassemble the compiled kernel and expose SASS, PTX, register usage, and spill info. Useful for understanding code generation after a kernel is correct.

Opt-in. The wrapper requires `cuobjdump` on `PATH`; `nvdisasm` is optional and unlocks lifetime-range information when present.

Arguments: `_KERNEL_CODE_SCHEMA`.

Returns:

```json
{
  "ok": true,
  "tool": "disassemble_kernel",
  "summary": "disassemble_kernel PASSED.\n<formatted SASS / PTX / register summary>",
  "max_registers": 32,
  "has_register_spills": false,
  "has_tensor_core_ops": true,
  "instruction_mix": {"FFMA": 0.42, "LDG": 0.18}
}
```

State: no record updates. Cost: `gpu_seconds`. Resources: `["gpu:<device_index>"]`. Sandboxable. No timeout, no progress.

## ert_roofline

Run empirical roofline micro-benchmarks to measure actual peak DRAM bandwidth and compute throughput for the current GPU. Results are cached per device (`use_cache=True`), so the first call on a fresh host is slow and subsequent calls return instantly.

Opt-in. Requires the empirical roofline driver (`kernelbench.ert.run_ert_benchmarks`).

Arguments: none, `{"type": "object", "properties": {}, "required": []}`.

Returns:

```json
{
  "ok": true,
  "tool": "ert_roofline",
  "summary": "ert_roofline PASSED.\n<formatted roofline model>",
  "model": {"peak_gflops": 14982, "peak_dram_gbs": 730}
}
```

State: no record updates. Cost: `gpu_seconds`. Resources: `["gpu:<device_index>"]`. Sandboxable. No timeout, no progress.

## get_gpu_specs

Return peak hardware specs for the GPU this kernel will run on (memory bandwidth, TFLOPS per precision, SM count). Useful once at the start of a run to calibrate optimization targets.

Arguments: none.

Returns:

```json
{
  "ok": true,
  "tool": "get_gpu_specs",
  "device_name": "NVIDIA H100 80GB HBM3",
  "total_memory_gb": 80.0,
  "spec": {"key": "H100", "fp32_tflops": 67, "fp16_tflops": 989, "mem_bw_gbs": 3350}
}
```

When no CUDA device is available the wrapper returns `ok=false` with a summary noting the absence. The `spec` entry is looked up in `kernelbench.prompts.hardware.gpu_specs.GPU_SPEC_INFO` by substring match against the device name.

State: no record updates. No costs, no GPU resource (the call is a metadata read on the device, not a compute kernel). Not sandboxable. No timeout, no progress.

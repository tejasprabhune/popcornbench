# Environment variables

This page is the catalog for every env var the PopcornBench project reads, grouped by what it controls. The catalog is exhaustive across `popcorn_world/popcorn_world/__init__.py`, `popcorn_world/scenarios/*.py`, and the LLM-backend selection ensemble does on its own. Per var: type, default, what it controls, and when you would change it.

## LLM provider keys

Read by ensemble when it constructs the LLM backend. The `auto` backend (selected when no scenario overrides the choice) picks the first one that resolves to a non-empty value. Setting more than one is fine; ensemble falls back through them in declared order.

`ANTHROPIC_API_KEY` (string, required for Anthropic models): the key the `AnthropicBackend` uses. Set this when you are running against `claude-sonnet-*`, `claude-opus-*`, or `claude-haiku-*` models.

`OPENAI_API_KEY` (string, required for OpenAI models): the key the `OpenAIBackend` uses. Set this when you are running against `gpt-*` models.

`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_AI_API_KEY`, `SGLANG_API_KEY` (strings, optional): keys for non-Anthropic, non-OpenAI providers. ensemble routes the model identifier (`POPCORN_AGENT_MODEL`) through the matching backend if its key is set.

When no key is set, ensemble falls back to the `MockBackend`, which is what makes the test suite and the `scripts/publish_traces.py` smoke run work without provider access. The [ensemble runtime reference](https://tejasprabhune.github.io/ensemble/reference/runtime.html) covers the backend-selection precedence in full.

## World setup

Read once in `popcorn_world.__init__._setup`, which runs each time a scenario constructs `World("popcorn")`. Changes to these vars only take effect at world construction; mutating the environment mid-run does not retro-apply.

`POPCORN_BACKEND` (string, default `"cuda"`): the kernel backend `state.backend`. Drives the static-checker's backend-specific implementation check, the load path in `compile_kernel` (CUDA goes through `load_custom_model`; `"triton"`, `"tilelang"`, `"cute"` go through `load_custom_model_with_tempfile`), and the tool descriptions the agent sees. Change when running a non-CUDA backend; leave alone for the default CUDA write task.

`POPCORN_PRECISION` (string, default `"fp32"`): the kernel precision. Resolved to a `torch.dtype` via `kernelbench.eval.get_torch_dtype_from_string`. Drives the tolerance the eval applies and the warning behaviour of the `precision_downgrade` static check (which only fires when the configured precision is fp32). Change when running fp16 or bf16 experiments.

`POPCORN_DEVICE_INDEX` (int, default `0`): the CUDA device index. Picks the device for `state.device` and is what the GPU resource lock string `gpu:<idx>` derives from. Change when running on a multi-GPU host and pinning a scenario to a specific device.

`POPCORN_BUILD_DIR` (string, default empty): the root build directory passed through to `torch.utils.cpp_extension`. Per-kernel subdirectories are derived as `<root>/k_<sha1-first-12>`. Change to give the build cache a stable location (helpful across runs that recompile identical kernels); leave empty for the default temporary build dir.

`POPCORN_NUM_CORRECT_TRIALS` (int, default `5`): how many independent trials `eval_kernel_against_ref` runs for correctness checking. Affects `run_correctness` and `submit_kernel`. Change up to make correctness more robust to a kernel that passes by luck on one shape, down to speed up smoke runs.

`POPCORN_NUM_PERF_TRIALS` (int, default `100`): how many timing trials `submit_kernel` runs. Drives the `runtime_us` value the agent sees and the runtime stats the trace records. Change up for more precise timing on noisy hosts, down to reduce GPU spend per submission.

`POPCORN_TIMING_METHOD` (string, default `"cuda_event"`): the timing method `eval_kernel_against_ref` uses for performance trials. The other supported value is `"wall"` (Python wall clock). Change when the host's CUDA events are unreliable; otherwise leave as `cuda_event`.

`POPCORN_GPU_ARCH` (comma-separated string, default `"Ada"`): the GPU architecture list passed to `kernelbench.utils.set_gpu_arch`. This affects how the build system compiles kernels (the `-gencode` flags). Set to `"Hopper"` on H100 hosts; `"Ada"` is the default and matches L40 / 6000 Ada.

`POPCORN_HELD_OUT_SEED` (int, default unset): the seed used for the held-out correctness retry inside `submit_kernel`. When set, every passing correctness check is followed by a rerun against this seed; the result is recorded on `state` under `held_out_correctness` and is what the `held_out_correctness_passed` / `held_out_correctness_failed` predicates read. Set this to any integer the agent has never observed; the default policy is to vary it across sweep replicas. Leave unset to disable held-out re-verification (the predicates then always read false).

`POPCORN_VERBOSE` (bool, default `false`): controls `state.verbose` and the verbosity of the underlying `kernelbench.eval` calls. Change to `true` when debugging an eval failure; leave off for sweep runs to keep logs manageable.

`POPCORN_SANDBOX_GPU_TOOLS` (bool, default `false`): turns on the subprocess sandbox for the heavy GPU tools (`compile_kernel`, `run_correctness`, `submit_kernel`, `profile_kernel`, `disassemble_kernel`). When on, a CUDA-fatal error kills only the worker, not the agent loop. The trade-off is that the worker reimports `popcorn_world` from scratch and so loses `fetch_problem` state, which means turning this on currently requires scenario-level rework to thread the problem reference through each call. Keep `false` until that lands.

## Scenario knobs

Read inside the scenario functions. Changing these between runs is the supported way to drive a sweep.

`POPCORN_LEVEL` (int, default `1`): the KernelBench level. Both scenarios pass it to `fetch_problem`. Range is 1 to 5 per the dataset.

`POPCORN_PROBLEM_ID` (int, default `19`): the problem id within the level. Passed to `fetch_problem` alongside `POPCORN_LEVEL`. The set of valid ids depends on the level; the dataset constructor raises on a missing id.

`POPCORN_DATASET_SRC` (string, default `"huggingface"`): the dataset source. Other supported value is `"local"`. Switch to `"local"` when running offline against a checked-out KernelBench dataset.

`POPCORN_AGENT_MODEL` (string, default `"claude-sonnet-4-5"`): the LLM behind the agent in `popcorn.single_problem`. The model identifier is passed straight to ensemble's backend selector.

`POPCORN_AUTHOR_MODEL` (string, default `"claude-sonnet-4-5"`): same role as `POPCORN_AGENT_MODEL` but for the author in `popcorn.judge_review`.

`POPCORN_REVIEWER_MODEL` (string, defaults to `POPCORN_AUTHOR_MODEL`): the LLM behind the reviewer in `popcorn.judge_review`. Defaulting to the author's model keeps the two actors capacity-matched; set to a smaller model when running an asymmetric pairing experiment.

`POPCORN_PERSONA` (string, default `"methodical_engineer"`): the persona name for the single agent in `popcorn.single_problem`. Resolves through the world's `personas_dir`. Set to `"normal"` for a baseline run, `"speed_obsessed"` for a red-team run, etc.

`POPCORN_AUTHOR_PERSONA` (string, default `"speed_obsessed"`): the author persona in `popcorn.judge_review`. The reviewer is always `code_reviewer`, so this is the only persona knob the judge scenario exposes.

`POPCORN_MAX_TURNS` (int, default `20` for single_problem, `30` for judge_review): turn budget. Used as `world.turn_count > MAX_TURNS` for the `until` predicate.

`POPCORN_GPU_BUDGET` (float, default `600.0` for single_problem, `900.0` for judge_review): the GPU-seconds cap, applied via `world.set_budget("gpu_seconds", ...)`. The runtime halts the scenario with `BudgetExceeded` if a tool would push the accumulated cost over this cap. Set higher for runs that include the profiling tools (which are slow per call); set lower for a smoke run.

`POPCORN_TOOLS` (comma-separated string, default empty): tool restriction for `popcorn.single_problem` and `popcorn.translate_problem`. Empty means use the scenario's default tool set; non-empty replaces it with the listed names. Use this to add the profiling tools (`POPCORN_TOOLS=fetch_problem,compile_kernel,...,profile_kernel,disassemble_kernel`) without editing the scenario. `popcorn.judge_review` does not read this var; its tool sets are hardcoded.

`POPCORN_SOURCE_ARCH` (string, default `"a100"`) and `POPCORN_TARGET_ARCH` (string, default `"h100"`): source and target GPU architectures for `popcorn.translate_problem`. Picks the subdirectory under `KernelBench/level<level>/kernels/` that `fetch_translation_problem` reads. The current dataset ships A100 and H100 only; adding a new pair means dropping matching subdirectories into the dataset and pointing the env vars at them.

## Ensemble paths

`ENSEMBLE_ROOT` (string, default `~/Documents/ensemble`): the path to the ensemble checkout, used by `scripts/publish_traces.py` as the source for `site/` (the static trace viewer). Set when ensemble lives somewhere other than the default; the script's `--ensemble-root` flag overrides this.

`ENSEMBLE_VLLM_BASE_URL` (string, optional): used by ensemble's persona resolver when wiring a `mode = "trained"` persona to a vLLM-served adapter. Not used by any persona PopcornBench ships today (all current personas are `mode = "prompted"`), but the env var is documented here so it is one search away when adding a trained adapter.

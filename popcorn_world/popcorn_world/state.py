"""Per-World mutable state for popcorn_world.

One PopcornState is constructed per ensemble World instance via the
setup factory. It carries the reference architecture source (set by
fetch_problem), the cached eval context the tool wrappers share, and
a small ledger of kernels we have already seen so predicates can
answer questions like "was a kernel ever submitted that passed
correctness."

State is intentionally light. The heavy lifting (compiling kernels,
running them, profiling) lives in kernelbench.eval; this class is the
piece ensemble's tool dispatch closes over.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from kernelbench.eval import get_torch_dtype_from_string


@dataclass
class KernelRecord:
    """One row in the ledger of kernels the agent has touched."""

    kernel_hash: str
    compiled: Optional[bool] = None
    correctness: Optional[bool] = None
    submitted: bool = False
    runtime_us: Optional[float] = None
    ref_runtime_us: Optional[float] = None
    excessive_speedup: bool = False
    static_check_passed: Optional[bool] = None
    # Result of a post-submission correctness retry with a held-out
    # seed the agent never observed. None means "not checked", True
    # means matched the reference, False means did not.
    held_out_correctness: Optional[bool] = None


@dataclass
class ProblemRecord:
    """The currently-loaded KernelBench problem."""

    level: int
    problem_id: int
    name: str
    ref_arch_src: str


class PopcornState:
    """The thing ensemble's per-instance setup factory closes over.

    Construction reads config from process environment variables so a
    scenario can override per-run (backend, precision, build_dir,
    device, trial counts) without changing the world definition.
    """

    def __init__(
        self,
        *,
        backend: str = "cuda",
        precision: str = "fp32",
        device_index: int = 0,
        build_dir: Optional[str] = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timing_method: str = "cuda_event",
        gpu_arch: Optional[List[str]] = None,
        held_out_shape_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.backend = backend
        self.precision = precision
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timing_method = timing_method
        self.verbose = verbose
        self.gpu_arch = gpu_arch or ["Ada"]
        self.held_out_shape_seed = held_out_shape_seed

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")

        if build_dir:
            self.build_dir = os.path.abspath(build_dir)
            os.makedirs(self.build_dir, exist_ok=True)
        else:
            self.build_dir = None

        self.problem: Optional[ProblemRecord] = None
        self._kernels: Dict[str, KernelRecord] = {}
        self._lock = threading.Lock()
        self._last_profile_summary: Any = None

    @property
    def torch_precision(self) -> torch.dtype:
        return get_torch_dtype_from_string(self.precision)

    def set_problem(self, problem: ProblemRecord) -> None:
        with self._lock:
            self.problem = problem
            self._kernels.clear()
            self._last_profile_summary = None

    def require_problem(self) -> ProblemRecord:
        if self.problem is None:
            raise RuntimeError(
                "no problem loaded. Call fetch_problem(level, problem_id) first."
            )
        return self.problem

    def kernel_hash(self, kernel_code: str) -> str:
        return hashlib.sha1(kernel_code.encode("utf-8", errors="replace")).hexdigest()

    def record(self, kernel_hash: str, **fields: Any) -> KernelRecord:
        with self._lock:
            rec = self._kernels.setdefault(kernel_hash, KernelRecord(kernel_hash=kernel_hash))
            for key, value in fields.items():
                setattr(rec, key, value)
            return rec

    def get_record(self, kernel_hash: str) -> Optional[KernelRecord]:
        with self._lock:
            return self._kernels.get(kernel_hash)

    def all_records(self) -> List[KernelRecord]:
        with self._lock:
            return list(self._kernels.values())

    def any_submitted(self) -> bool:
        return any(r.submitted for r in self.all_records())

    def best_submitted_runtime(self) -> Optional[float]:
        runtimes = [
            r.runtime_us for r in self.all_records()
            if r.submitted and r.runtime_us is not None and r.correctness
        ]
        return min(runtimes) if runtimes else None

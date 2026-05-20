# kernels/domains/

Domain-specific kernel problems that are not part of upstream KernelBench. Bio / physics / signal-processing operators whose PyTorch references live alongside the agent's target.

Each subdirectory under `domains/` is a domain (e.g. `domains/bio/`, `domains/physics/`). Each subdirectory contains `.py` files that define a PyTorch `Model` class the agent must replace with a custom CUDA / Triton / Helion kernel, in the same shape as `kernels/kernelbench/levelN/`.

The dataset loader resolves these paths through the same `LocalKernelBenchDataset` once the `dataset.py` constants are pointed at this tree. Adding a new domain is a matter of dropping `.py` files in and (optionally) registering the domain name in `prompt_constructor_toml.py` for prompt customisation.

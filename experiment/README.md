# Experiment harness (to be added from the original machine)

The report's measurements were produced on the author's laptop (RTX 5070 Laptop GPU, Qwen2.5-Coder 7B Q4_K_M, local runtime). The scripts that ran those experiments are **not part of this repository yet**. Copy them here unchanged, so the repo contains the code that actually produced the numbers:

- [ ] runner (task/seed/policy loop, prompt builders, stop rules, 8 dev + 24 held-out problems, seeds 11/29/47)
- [ ] power logger (instantaneous board power, about 100 ms interval) and the trapezoid energy integration, Eq. (3)
- [ ] EvalPlus scoring wrapper (30 s sandbox limit per test program)
- [ ] prefix-cache probe (Section 2.1)
- [ ] exact model file/tag, runtime version, GPU driver version, OS, Python/package versions

Do not rewrite these after the fact. If anything must change for sharing (paths, secrets), say so in the README.

"""P5 — Router Correctness.

Hand-counted mixed-ndim parameter set. After one step, query
``get_telemetry()["num_2d_params"]`` and ``["num_1d_params"]`` and
assert they match the ground-truth count from
``[p.ndim for p in params]``.

Why this problem: Liger's contribution *is* the dispatch. A broken
router would invalidate every other claim. P5 is a fast sanity gate
that runs in CPU seconds and is the first gate in the RunPod launch
checklist.

The parameter set covers all relevant ndim values:
  - 3 scalars (ndim = 0) — Yogi route
  - 2 vectors (ndim = 1) — Yogi route
  - 4 matrices (ndim = 2) — Lion route
  - 1 3-D tensor (ndim = 3) — Lion route (Liger treats ndim >= 2 as Lion)
  - 1 4-D tensor (ndim = 4) — Lion route

Expected: ``num_2d_params = 6`` (4 + 1 + 1), ``num_1d_params = 5``
(3 + 2).

This problem only runs meaningfully against Liger itself; for other
optimizers we just record that the problem ran without NaN.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class P5RouterCorrectness(BenchProblem):
    """Telemetry vs hand-counted ground-truth dispatch census."""

    name = "p5_router_correctness"
    max_steps = 1
    converged_tol = float("inf")  # binary pass/fail via telemetry check

    EXPECTED_LION = 6  # 4 matrices + 1 3-D + 1 4-D
    EXPECTED_YOGI = 5  # 3 scalars + 2 vectors

    def init_params(self) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        d = self.device
        # Scalars (ndim = 0).
        for _ in range(3):
            params.append(torch.tensor(0.5, device=d, requires_grad=True))
        # Vectors (ndim = 1).
        params.append(torch.randn(8, device=d, requires_grad=True))
        params.append(torch.randn(16, device=d, requires_grad=True))
        # Matrices (ndim = 2).
        for size in [(4, 4), (8, 16), (16, 32), (32, 32)]:
            params.append(torch.randn(*size, device=d, requires_grad=True))
        # 3-D tensor.
        params.append(torch.randn(2, 3, 4, device=d, requires_grad=True))
        # 4-D tensor (mock-conv-weight shape).
        params.append(torch.randn(2, 3, 3, 3, device=d, requires_grad=True))
        return params

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        # Sum of squared norms; gradient is just 2*p for every param.
        # This is enough to populate optimizer state on the first step.
        return sum((p * p).sum() for p in params)

    def converged(self, current_loss: float, step: int) -> bool:
        return step >= 1

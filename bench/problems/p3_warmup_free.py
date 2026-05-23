"""P3 — Warmup-Free.

Small MLP with matrix-only parameters (no biases, no scalars — pure 2-D
weights). Optimizer is run with no LR warmup. Loss is measured at step
1, step 5, step 50.

Why this problem: Lion-family sign-momentum optimizers are operational
from step 1; Adam-family adaptive optimizers need a β2 accumulation
window before ``v_hat`` is meaningful, and practitioners usually compose
that with an LR warmup. P3 isolates the warmup-independence claim from
§4.1 of the paper.

This problem uses a 64×64 → 64 → 1 regression MLP on synthetic linear
data with mean-squared loss. Matrix-only deliberately — the goal is to
isolate the Lion-path warmup property; the Yogi-path scalar/vector
contribution is tested in P1 and P2.

Convergence tolerance: by step 50, loss < 0.5 × initial_loss. The
optimizer must demonstrate that the first 50 steps did meaningful work.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


_HIDDEN = 64
_BATCH = 128


class P3WarmupFree(BenchProblem):
    """Matrix-only MLP, no LR warmup, early-loss measurement."""

    name = "p3_warmup_free"
    max_steps = 100
    converged_tol = float("inf")  # we judge by loss-at-step-50 separately

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        gen = self._generator
        self._W1_star = (torch.randn(_HIDDEN, _HIDDEN, generator=gen) * 0.1).to(self.device)
        self._W2_star = (torch.randn(_HIDDEN, 1, generator=gen) * 0.1).to(self.device)
        self._x = torch.randn(_BATCH, _HIDDEN, generator=gen).to(self.device)
        with torch.no_grad():
            self._y = torch.tanh(self._x @ self._W1_star) @ self._W2_star

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        W1 = (torch.randn(_HIDDEN, _HIDDEN, generator=gen) * 0.05).to(self.device)
        W2 = (torch.randn(_HIDDEN, 1, generator=gen) * 0.05).to(self.device)
        for p in (W1, W2):
            p.requires_grad_(True)
        return [W1, W2]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        W1, W2 = params
        y_pred = torch.tanh(self._x @ W1) @ W2
        return ((y_pred - self._y) ** 2).mean()

    def converged(self, current_loss: float, step: int) -> bool:
        # No early-exit; we want the trajectory and the step-50 datapoint.
        return False

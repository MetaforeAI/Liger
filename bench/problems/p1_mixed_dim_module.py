"""P1 — Mixed-Dim Module (headline).

Module with one 64×64 matrix (Lion route target), one 64-dim bias
(Yogi route target), and one scalar gate (Yogi route target). Loss is
mean-squared regression on synthetic linear data:

    y_pred = (x @ W + b) * gate
    loss   = mean((y_pred - y_target)²)

Why this problem: Liger's contribution is dispatch by dimensionality.
This is the canonical setup where each baseline pays a tax somewhere —
AdamW's β2 warmup wastes the first 50-200 steps; Lion-only treats the
bias and gate identically to the matrix (sign-update on a bursty scalar
gate destroys rank-1 information); Yogi-only burns full Adam memory on
the matrix and gains nothing for it.

The data-generating process picks a random target matrix W*, target
bias b*, target gate g*, and produces ``y = (x @ W* + b*) * g*`` for
random Gaussian x. With the model initialized at small random weights,
the optimum is unique and the loss reaches ~0 at convergence.

Convergence tolerance: loss < 1e-2 (relative to typical initial loss of
~10-50).
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


_DIM = 64
_BATCH = 128


class P1MixedDimModule(BenchProblem):
    """Mixed-dimensional regression module."""

    name = "p1_mixed_dim_module"
    max_steps = 2000
    converged_tol = 1e-2

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        gen = self._generator
        # Target parameters (held fixed; the optimizer recovers these).
        self._W_star = torch.randn(_DIM, _DIM, generator=gen) * 0.1
        self._b_star = torch.randn(_DIM, generator=gen) * 0.1
        self._gate_star = torch.tensor(1.5)
        # Held-out batch of inputs (one batch reused throughout to make
        # the loss landscape stationary across steps).
        self._x = torch.randn(_BATCH, _DIM, generator=gen)
        # Cache the target output so we don't recompute it.
        with torch.no_grad():
            self._y = (self._x @ self._W_star + self._b_star) * self._gate_star

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        W = torch.randn(_DIM, _DIM, generator=gen) * 0.01
        b = torch.zeros(_DIM)
        gate = torch.tensor(1.0)
        for p in (W, b, gate):
            p.requires_grad_(True)
        return [W, b, gate]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        W, b, gate = params
        y_pred = (self._x @ W + b) * gate
        return ((y_pred - self._y) ** 2).mean()

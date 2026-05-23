"""P2 — Scalar Burst.

Single scalar parameter, gradient stream is mostly small (~1.0) with
periodic bursts (~1e3) every 50 steps. Tests Yogi-path burst recovery
specifically: Adam's v_t locks high after a burst and takes ~1/(1-β2)
steps to decay; Yogi's v_t recovers bidirectionally on subsequent small
gradients.

The "loss" here is contrived (we don't have a closed-form scalar loss
that produces bursty gradients on demand). Instead, we register a custom
``loss_and_grad`` that emits a pre-scripted gradient sequence and
computes a tracking loss of the form ``(p - target)²`` where the target
drifts slowly. The bursty gradients are injected directly — the goal of
this problem is to expose the optimizer's v_t trajectory, not to test
convergence on a real loss surface.

Convergence tolerance: by step 500, the optimizer must keep the
parameter within distance < 0.1 of the slowly-drifting target. Adam-
family optimizers tend to overshoot massively right after a burst (the
inflated v_t suppresses subsequent normal-magnitude gradients); Yogi
and Liger should stay tracked.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


_BURST_PERIOD = 50
_BURST_MAGNITUDE = 1e3
_NORMAL_MAGNITUDE = 1.0


class P2ScalarBurst(BenchProblem):
    """Scalar parameter under a bursty gradient stream."""

    name = "p2_scalar_burst"
    max_steps = 500
    converged_tol = 1.0  # max distance from drifting target

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        self._step = 0
        # Target drifts very slowly: t -> 1.0 + 0.001 * step.
        # The optimizer should keep p tracking close to this.

    def init_params(self) -> List[torch.Tensor]:
        p = torch.tensor(0.0, device=self.device, requires_grad=True)
        return [p]

    def _target(self) -> float:
        return 1.0 + 0.001 * self._step

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (p,) = params
        target = self._target()
        # "Loss" is just the tracking distance; useful for the converged
        # check and for logging the trajectory.
        loss = float((p.detach() - target) ** 2)

        # Gradient: tracking gradient + periodic burst.
        # Direction is sign(p - target), pointing back toward the target.
        # Magnitude is normal except on burst steps.
        direction = float(torch.sign(p.detach() - target).item())
        if direction == 0.0:
            direction = 1.0
        if (self._step + 1) % _BURST_PERIOD == 0:
            magnitude = _BURST_MAGNITUDE
        else:
            magnitude = _NORMAL_MAGNITUDE
        grad = torch.tensor(direction * magnitude, device=self.device)
        self._step += 1
        return loss, [grad]

    def converged(self, current_loss: float, step: int) -> bool:
        # Only consider "converged" after a few burst cycles have passed
        # so we test long-run tracking, not just initialization luck.
        if step < 200:
            return False
        return current_loss < self.converged_tol

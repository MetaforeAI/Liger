"""P4 — Memory Footprint.

Allocates a synthetic 'transformer-scale' parameter set, instantiates
optimizer state, and reports the total bytes of state. No training is
run — this problem measures the static memory footprint of the
optimizer alone.

The synthetic parameter set is a stripped-down transformer block:

    - 4 attention projections : ``d × d``       (matrix; Lion route)
    - 2 MLP matrices          : ``d × 4d``      (matrix; Lion route)
    - 2 RMSNorm gains         : ``d``           (vector; Yogi route)
    - 1 attention output scale: ``1``           (scalar; Yogi route)
    - 2 attention biases      : ``d``           (vector; Yogi route)

Per block: 6 matrix params + 5 vector/scalar params. With ``d = 1024``
this is ~12.6M matrix elements + ~4K vector/scalar elements per block.
We replicate by ``num_blocks = 80`` to reach the 1B-equivalent scale
roughly used in §4.3.

Convergence "tolerance" is just a sentinel — the problem reports its
finding via ``loss_and_grad`` returning a zero loss after one step (so
the harness records ``optimizer_state_bytes`` in the CSV row).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


_HIDDEN = 1024
_NUM_BLOCKS = 80  # ~1B effective parameter count


class P4MemoryScaling(BenchProblem):
    """Allocator-only memory measurement; no training."""

    name = "p4_memory_scaling"
    max_steps = 1
    converged_tol = float("inf")  # the test is byte-count, not loss

    def init_params(self) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        for _ in range(_NUM_BLOCKS):
            # 4 attention projections: Q, K, V, O — each d × d.
            for _ in range(4):
                params.append(torch.zeros(_HIDDEN, _HIDDEN, requires_grad=True))
            # 2 MLP matrices: d × 4d and 4d × d.
            params.append(torch.zeros(_HIDDEN, 4 * _HIDDEN, requires_grad=True))
            params.append(torch.zeros(4 * _HIDDEN, _HIDDEN, requires_grad=True))
            # 2 RMSNorm gains.
            for _ in range(2):
                params.append(torch.ones(_HIDDEN, requires_grad=True))
            # 1 attention output scale.
            params.append(torch.tensor(1.0, requires_grad=True))
            # 2 attention biases.
            for _ in range(2):
                params.append(torch.zeros(_HIDDEN, requires_grad=True))
        return params

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        # Emit zero gradients so the optimizer's state is fully
        # instantiated on the first step() but the parameters don't move.
        grads = [torch.zeros_like(p) for p in params]
        return 0.0, grads

    def converged(self, current_loss: float, step: int) -> bool:
        # Run exactly one step then stop.
        return step >= 1


def measure_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Sum tensor.numel() * tensor.element_size() across all state buffers.

    Note that some optimizers stash non-tensor scalars (step counts,
    booleans) which contribute zero bytes here — we want the *tensor*
    memory cost, which is what dominates at scale.
    """
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total

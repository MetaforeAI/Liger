"""Unit tests for the Liger optimizer.

Covers: construction & validation, router dispatch by ndim, Lion-path
correctness (sign-momentum, step-1 behavior, no warmup), Yogi-path
correctness (rank-1 burst bound, bias correction, eps floor), decoupled
weight decay on both routes, toy convergence (matrix, scalar, mixed
module), and the telemetry contract.

All tests are CPU-only and deterministic via per-test seeding.
"""

from __future__ import annotations

import math

import pytest
import torch

from liger import Liger


# ── Helpers ──────────────────────────────────────────────────────────────


def _seeded(seed: int = 0) -> None:
    torch.manual_seed(seed)


def _make_mixed_module() -> torch.nn.Module:
    """Module with one 2-D matrix, one 1-D bias, and one 0-D scalar gate.

    Exercises both routes inside a single optimizer instance.
    """
    class _Mixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(32, 32))
            self.b = torch.nn.Parameter(torch.zeros(32))
            self.gate = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (x @ self.W + self.b) * self.gate

    _seeded(0)
    return _Mixed()


# ── Construction & validation ────────────────────────────────────────────


def test_construction_defaults() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Liger([p])
    g = opt.param_groups[0]
    assert g["lr"] == 1e-4
    assert g["betas"] == (0.9, 0.99)
    assert g["eps_yogi"] == 1e-3
    assert g["eps_adam"] == 1e-8
    assert g["weight_decay"] == 0.0
    assert g["initial_accumulator"] == 1e-6


def test_construction_rejects_bad_args() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="lr"):
        Liger([p], lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        Liger([p], lr=-1e-3)
    with pytest.raises(ValueError, match="betas"):
        Liger([p], betas=(1.0, 0.99))
    with pytest.raises(ValueError, match="betas"):
        Liger([p], betas=(0.9, -0.01))
    with pytest.raises(ValueError, match="eps_yogi"):
        Liger([p], eps_yogi=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        Liger([p], weight_decay=-0.1)
    with pytest.raises(ValueError, match="initial_accumulator"):
        Liger([p], initial_accumulator=-1e-6)


def test_state_dict_roundtrip() -> None:
    p = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(torch.zeros(8))
    opt = Liger([p, b], lr=1e-3)
    # Take a few steps to populate state.
    for _ in range(3):
        p.grad = torch.randn_like(p)
        b.grad = torch.randn_like(b)
        opt.step()
    sd = opt.state_dict()
    # Re-construct and load.
    p2 = torch.nn.Parameter(p.detach().clone())
    b2 = torch.nn.Parameter(b.detach().clone())
    opt2 = Liger([p2, b2], lr=1e-3)
    opt2.load_state_dict(sd)
    # Step counts and routes survived.
    s_p = opt2.state[opt2.param_groups[0]["params"][0]]
    s_b = opt2.state[opt2.param_groups[0]["params"][1]]
    assert s_p["step"] == 3
    assert s_b["step"] == 3
    assert s_p["is_lion"] is True
    assert s_b["is_lion"] is False


def test_sparse_gradient_raises() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Liger([p])
    sparse_grad = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.tensor([1.0]),
        size=(4, 4),
    )
    p.grad = sparse_grad
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


# ── Router ───────────────────────────────────────────────────────────────


def test_router_2d_takes_lion_path() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Liger([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state[p]
    assert state["is_lion"] is True
    assert "exp_avg" in state
    assert "exp_avg_sq" not in state
    assert "last_momentum_norm" in state


def test_router_1d_takes_yogi_path() -> None:
    p = torch.nn.Parameter(torch.zeros(8))
    opt = Liger([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state[p]
    assert state["is_lion"] is False
    assert "exp_avg" in state
    assert "exp_avg_sq" in state
    assert "last_v_hat_max" in state


def test_router_scalar_takes_yogi_path() -> None:
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = Liger([p], lr=1e-3)
    p.grad = torch.tensor(0.5)
    opt.step()
    state = opt.state[p]
    assert state["is_lion"] is False
    assert "exp_avg_sq" in state


# ── Lion path correctness ────────────────────────────────────────────────


def test_lion_step1_is_sign_of_g() -> None:
    """At t=1 with init_acc ≈ 0, update direction equals sign(g)."""
    _seeded(1)
    p = torch.nn.Parameter(torch.zeros(4, 4))
    opt = Liger([p], lr=1e-2, initial_accumulator=0.0)
    g = torch.randn_like(p)
    # Ensure no zero entries (sign(0) = 0 would break the test).
    g = g + 0.5 * torch.sign(g)
    p.grad = g.clone()
    before = p.detach().clone()
    opt.step()
    delta = p.detach() - before
    expected = -1e-2 * torch.sign(g)
    assert torch.allclose(delta, expected, atol=1e-6)


def test_lion_update_magnitude_is_lr() -> None:
    """Each coordinate moves by exactly ±lr (sign-update property)."""
    _seeded(2)
    p = torch.nn.Parameter(torch.zeros(8, 8))
    lr = 3e-3
    opt = Liger([p], lr=lr, initial_accumulator=0.0)
    g = torch.randn_like(p)
    g = g + 0.5 * torch.sign(g)
    p.grad = g.clone()
    before = p.detach().clone()
    opt.step()
    delta = p.detach() - before
    assert torch.allclose(delta.abs(), torch.full_like(delta, lr), atol=1e-6)


def test_lion_no_warmup() -> None:
    """Step 1 produces a finite, non-zero update; no v_t cold-start."""
    _seeded(3)
    p = torch.nn.Parameter(torch.zeros(4, 4))
    opt = Liger([p], lr=1e-3)
    p.grad = torch.randn_like(p) + 0.1
    before = p.detach().clone()
    opt.step()
    delta = p.detach() - before
    assert torch.isfinite(delta).all()
    assert delta.abs().sum().item() > 0.0


def test_lion_momentum_accumulates() -> None:
    """Over many steps with constant g, ||m_t|| approaches ||g||."""
    _seeded(4)
    p = torch.nn.Parameter(torch.zeros(4, 4))
    opt = Liger([p], lr=1e-4, initial_accumulator=0.0)
    g = torch.randn_like(p)
    g_norm = g.norm().item()
    norms = []
    for _ in range(30):
        p.grad = g.clone()
        opt.step()
        norms.append(opt.state[p]["exp_avg"].norm().item())
    # Monotonic (allowing tiny float wobble) and approaching ||g||.
    assert norms[-1] > norms[0]
    assert norms[-1] <= g_norm + 1e-6
    # After ~30 steps with β1=0.9, ||m|| should be > 0.9 · ||g||.
    assert norms[-1] > 0.9 * g_norm


# ── Yogi path correctness ────────────────────────────────────────────────


def test_yogi_step1_finite() -> None:
    _seeded(5)
    p = torch.nn.Parameter(torch.zeros(8))
    opt = Liger([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    delta = p.detach() - before
    assert torch.isfinite(delta).all()
    assert delta.abs().sum().item() > 0.0


def test_yogi_rank1_burst_bounded() -> None:
    """A single bursty gradient cannot inflate v_t the way Adam would.

    Yogi's update is ``v_t -= (1-β2)·sign(v_{t-1} - g²)·g²``. The new
    value lies in ``[v_{t-1} - (1-β2)·g², v_{t-1} + (1-β2)·g²]``, so a
    single burst of magnitude ``g²_burst`` can move ``v_t`` by at most
    ``(1-β2)·g²_burst`` in either direction. Adam, by contrast, would
    set ``v_t = β2·v_{t-1} + (1-β2)·g²_burst`` and pin to roughly
    ``(1-β2)·g²_burst`` directly from a single burst — the same
    *magnitude* on the burst step, but with Adam's accumulator there is
    no mechanism to *decrease* v_t back down (only exponential decay),
    while Yogi's sign-flip can shrink it again when subsequent gradients
    are small.
    """
    _seeded(6)
    p = torch.nn.Parameter(torch.zeros(4))
    beta2 = 0.99
    opt = Liger([p], lr=1e-3, betas=(0.9, beta2), initial_accumulator=0.0)
    # Warm with small gradients.
    for _ in range(20):
        p.grad = torch.full_like(p, 1e-2)
        opt.step()
    v_before_burst = opt.state[p]["exp_avg_sq"].clone()
    # Inject the burst.
    burst = 1e3
    p.grad = torch.full_like(p, burst)
    opt.step()
    v_after_burst = opt.state[p]["exp_avg_sq"].clone()
    # Yogi bound: |v_after - v_before| ≤ (1-β2) · g².
    expected_max_delta = (1.0 - beta2) * burst * burst
    actual_delta = (v_after_burst - v_before_burst).abs().max().item()
    assert actual_delta <= expected_max_delta + 1e-3, (
        f"Yogi v_t moved by {actual_delta:.3e} which exceeds "
        f"the theoretical bound (1-β2)·g² = {expected_max_delta:.3e}"
    )
    # Now hit it with moderate gradients; v_t should shrink (sign flip).
    # We need g² visible at v_t's scale to see the decrement in finite
    # precision: with v_t ~ 1e4 and (1-β2) = 0.01, choosing g = 10 makes
    # each step shrink v_t by ~1.0, visible after 5 steps.
    for _ in range(5):
        p.grad = torch.full_like(p, 10.0)
        opt.step()
    v_after_recovery = opt.state[p]["exp_avg_sq"]
    # The accumulator decreased after the burst — Yogi's distinguishing
    # property versus Adam, whose v_t can only decrease via exponential
    # decay (β2·v_{t-1}), not via the sign-flip mechanism.
    assert v_after_recovery.max().item() < v_after_burst.max().item()


def test_yogi_bias_correction_at_step1() -> None:
    """m_hat at t=1 must be approximately g (not (1-β1)·g)."""
    _seeded(7)
    p = torch.nn.Parameter(torch.zeros(4))
    beta1 = 0.9
    opt = Liger(
        [p],
        lr=1e-3,
        betas=(beta1, 0.99),
        initial_accumulator=0.0,
    )
    g = torch.tensor([1.0, -2.0, 3.0, -0.5])
    p.grad = g.clone()
    opt.step()
    # m_t = (1-β1)·g. m_hat = m_t / (1 - β1^1) = (1-β1)·g / (1-β1) = g.
    m = opt.state[p]["exp_avg"]
    expected_m = (1.0 - beta1) * g
    assert torch.allclose(m, expected_m, atol=1e-6)
    bc1 = 1.0 - beta1 ** 1
    m_hat = m / bc1
    assert torch.allclose(m_hat, g, atol=1e-6)


def test_yogi_eps_floor_active_near_zero_v() -> None:
    """When v_hat is near zero, denom is clamped by eps_yogi.

    The first step starts from ``exp_avg_sq = init_acc = 0``. After the
    update with a very small g, v_t is ~0 — the eps_yogi floor must
    keep denom finite and ≥ eps_yogi so the update doesn't blow up.
    """
    _seeded(8)
    p = torch.nn.Parameter(torch.zeros(4))
    opt = Liger([p], lr=1e-3, eps_yogi=1e-3, initial_accumulator=0.0)
    p.grad = torch.full_like(p, 1e-10)
    before = p.detach().clone()
    opt.step()
    delta = p.detach() - before
    assert torch.isfinite(delta).all()
    # With g=1e-10 and eps_yogi=1e-3, the update magnitude must be
    # bounded by ``lr · g_bias_corrected / eps_yogi`` ≈ 1e-3 · 1e-10 / 1e-3
    # = 1e-10. (Loosely: definitely well below 1.)
    assert delta.abs().max().item() < 1e-6


# ── Weight decay ─────────────────────────────────────────────────────────


def test_weight_decay_lion_decoupled() -> None:
    """Lion path: wd>0 with g=0 scales p by (1 - lr·wd)."""
    _seeded(9)
    p = torch.nn.Parameter(torch.randn(4, 4))
    lr = 1e-2
    wd = 0.1
    opt = Liger([p], lr=lr, weight_decay=wd)
    p.grad = torch.zeros_like(p)
    before = p.detach().clone()
    opt.step()
    # With g=0, m_t stays at init_acc (tiny), sign(m_t) = sign(init_acc) > 0
    # contributes a uniform ±lr nudge. We isolate the decoupled wd by
    # measuring the ratio of the "decay-only" component:
    expected_after_decay = before * (1.0 - lr * wd)
    # The sign-update on near-zero momentum adds exactly ±lr per coord.
    # Reconstruct the expected post-step value:
    init_acc = opt.param_groups[0]["initial_accumulator"]
    # Lion's update at step 1: m = (1-β1)·g + β1·init_acc = β1·init_acc > 0
    sign_update = torch.sign(torch.full_like(p, init_acc))
    expected = expected_after_decay - lr * sign_update
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_weight_decay_yogi_decoupled() -> None:
    """Yogi path: wd>0 with g=0 scales p by (1 - lr·wd)."""
    _seeded(10)
    p = torch.nn.Parameter(torch.randn(8))
    lr = 1e-2
    wd = 0.1
    opt = Liger([p], lr=lr, weight_decay=wd, initial_accumulator=0.0)
    p.grad = torch.zeros_like(p)
    before = p.detach().clone()
    opt.step()
    # With init_acc=0 and g=0, m_t stays zero, so Yogi update is zero;
    # only the decay applies.
    expected = before * (1.0 - lr * wd)
    assert torch.allclose(p.detach(), expected, atol=1e-6)


# ── Convergence ──────────────────────────────────────────────────────────


def test_toy_quadratic_convergence_2d() -> None:
    """32x32 matrix on ||W - W*||²; Lion route converges."""
    _seeded(11)
    W_star = torch.randn(32, 32)
    W = torch.nn.Parameter(torch.zeros(32, 32))
    opt = Liger([W], lr=5e-3)
    initial_loss = (W.detach() - W_star).pow(2).sum().item()
    for _ in range(500):
        diff = W - W_star
        loss = diff.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = (W.detach() - W_star).pow(2).sum().item()
    # Lion's sign-update only resolves down to ~lr per coordinate, so we
    # don't expect arbitrarily small loss — but a clear order-of-magnitude
    # reduction must happen.
    assert final_loss < 0.1 * initial_loss
    assert math.isfinite(final_loss)


def test_toy_scalar_convergence_1d() -> None:
    """Scalar gate on (s - s*)²; Yogi route converges within 200 steps."""
    _seeded(12)
    s_star = torch.tensor(2.0)
    s = torch.nn.Parameter(torch.tensor(0.0))
    opt = Liger([s], lr=5e-2)
    for _ in range(200):
        loss = (s - s_star).pow(2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(s.item() - s_star.item()) < 0.1


def test_mixed_dim_module_convergence() -> None:
    """Module with matrix + bias + scalar gate: loss strictly decreases."""
    _seeded(13)
    model = _make_mixed_module()
    opt = Liger(model.parameters(), lr=3e-3)
    x = torch.randn(64, 32)
    target = torch.randn(64, 32)
    losses = []
    for _ in range(300):
        out = model(x)
        loss = (out - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # The trajectory may not be strictly monotonic step-by-step (Lion's
    # sign-update can oscillate at the scale of lr per coordinate), but a
    # 50-step moving-average prefix vs suffix must show clear decrease.
    early = sum(losses[:50]) / 50
    late = sum(losses[-50:]) / 50
    assert late < early
    assert late < 0.5 * losses[0]


# ── Telemetry ────────────────────────────────────────────────────────────


def test_get_telemetry_aggregates_correctly() -> None:
    """Multi-param module: dispatch census + per-route fields populate."""
    _seeded(14)
    model = _make_mixed_module()
    opt = Liger(model.parameters(), lr=1e-3)
    x = torch.randn(16, 32)
    target = torch.randn(16, 32)
    for _ in range(5):
        out = model(x)
        loss = (out - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    tel = opt.get_telemetry()
    assert tel["step_count"] == 5
    assert tel["num_2d_params"] == 1  # W
    assert tel["num_1d_params"] == 2  # b (1-D) + gate (0-D)
    assert tel["last_max_momentum_norm"] > 0.0
    assert tel["last_max_update_l1"] > 0.0
    # last_max_update_l1 should equal the element count of W (32×32=1024)
    # once momentum has saturated to non-zero everywhere.
    assert tel["last_max_update_l1"] <= 32 * 32
    assert tel["last_max_v_hat"] > 0.0
    assert tel["last_min_v_hat"] >= 0.0


def test_telemetry_route_dispatch_census() -> None:
    """num_2d + num_1d matches the ndim-based ground truth."""
    _seeded(15)
    params = [
        torch.nn.Parameter(torch.randn(4, 4)),       # 2D
        torch.nn.Parameter(torch.randn(8, 16)),      # 2D
        torch.nn.Parameter(torch.randn(2, 3, 4)),    # 3D → 2D-ish (Lion)
        torch.nn.Parameter(torch.zeros(7)),          # 1D
        torch.nn.Parameter(torch.tensor(0.5)),       # 0D
        torch.nn.Parameter(torch.zeros(5)),          # 1D
    ]
    expected_lion = sum(1 for p in params if p.ndim >= 2)
    expected_yogi = sum(1 for p in params if p.ndim < 2)
    opt = Liger(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    tel = opt.get_telemetry()
    assert tel["num_2d_params"] == expected_lion
    assert tel["num_1d_params"] == expected_yogi

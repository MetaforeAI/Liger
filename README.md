# Liger

**Layered Iterative Gradient Estimator with Rectification.**

A hybrid optimizer that routes parameters by dimensionality. Matrix-shaped parameters (`ndim ≥ 2`) take a **Lion** sign-momentum step; vector- and scalar-shaped parameters (`ndim ≤ 1`) take a **Yogi** variance-rectified step. The dispatch is pinned per parameter at first encounter — no per-step branching cost beyond the `ndim` check.

Lion is unmodified. Yogi is unmodified. The contribution is structural: admit that different parameter shapes deserve different update rules, and dispatch automatically.

## Why

Modern transformer-style modules are gradient-heterogeneous. A single update rule pays a tax somewhere:

  - **AdamW everywhere** — couples β2 warmup to LR warmup, the two race each other, adaptive engine never operates cleanly on cold-start signal.
  - **Lion everywhere** — sign-update on scalar gates destroys rank-1 burst structure; bias terms underperform.
  - **Yogi everywhere** — variance tracking is unnecessary on already-well-conditioned matrix gradients; pays full Adam memory cost for matrices that don't need it.

Liger: Lion on matrices (cheap, warmup-free, half the memory), Yogi on scalars/vectors (variance-bounded, burst-safe). Approximately **55% of AdamW's optimizer-state memory** on transformer-derivative architectures.

## Install

One file, no dependencies beyond PyTorch.

```bash
curl -O https://raw.githubusercontent.com/MetaforeAI/Liger/main/liger.py
```

Or clone:

```bash
git clone https://github.com/MetaforeAI/Liger.git
```

## Use

```python
from liger import Liger

opt = Liger(
    model.parameters(),
    lr=1e-4,                      # base LR (Lion typically wants 3-10x lower than Adam)
    betas=(0.9, 0.99),            # β1 shared, β2 lower than Adam (Yogi bounds v_t)
    eps_yogi=1e-3,                # Yogi paper default
    weight_decay=0.0,             # AdamW-style decoupled when non-zero
)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
```

Param-shape dispatch is automatic. Reshape after step 1 is undefined behavior — the route (`is_lion: bool`) is pinned in `state` at first encounter.

## Telemetry

```python
t = opt.get_telemetry()
print(f"step={t['step_count']}  "
      f"lion={t['num_2d_params']}  yogi={t['num_1d_params']}  "
      f"max_mom={t['last_max_momentum_norm']:.3e}  "
      f"max_vhat={t['last_max_v_hat']:.3e}")
```

What to watch:

  - **`last_max_momentum_norm`** growing toward `||g||` is the Lion-path health signal. Explosion ⇒ LR too high.
  - **`last_max_v_hat`** persistently ≫ typical `g²` ⇒ a burst that hasn't recovered. Yogi will recover bidirectionally; if it doesn't, something is hammering the scalar gate continuously.
  - **`num_2d_params` + `num_1d_params`** ≠ total parameters ⇒ router miscounted (covered by `test_telemetry_route_dispatch_census` — should never trigger).

There is no NS5-success/skip counter (no preconditioner to fail) and no RAdam rectification counter (no warmup gate to throttle). Liger is structurally simple by design.

## Test suite

```bash
pytest test_liger.py -v
```

22 tests covering construction & validation, router dispatch, Lion-path correctness (sign-of-g at step 1, ±lr magnitude, no warmup, momentum accumulation), Yogi-path correctness (rank-1 burst bound, bias correction, eps floor), decoupled weight decay on both routes, toy convergence (matrix + scalar + mixed module), and telemetry contracts. CPU-only, deterministic, ~2 seconds.

## Benchmark

```bash
cd bench/
python run_bench.py --problem p5 --optimizer liger --lr 1e-4 --seed 0   # router sanity, CPU
python run_bench.py --sweep --output results.csv                         # full matrix (needs GPU)
python plot_bench.py --input results.csv --output figs/                  # paper figures
```

Five problems isolate the analytical claims:

  - **P1 — Mixed-Dim Module** (headline): matrix + bias + scalar gate, Liger vs each ablated baseline.
  - **P2 — Scalar Burst**: rank-1 burst injection, `v_hat` trajectory plot.
  - **P3 — Warmup-Free**: no LR warmup, step-1/step-5/step-50 loss comparison.
  - **P4 — Memory**: optimizer-state bytes on a 1B-equivalent module.
  - **P5 — Router Correctness**: telemetry vs hand-counted ground truth.

See `bench/README.md` for the claim ledger and the RunPod launch checklist.

## Paper

[Liger_Paper.md](Liger_Paper.md) — full design rationale, the two motivating pathologies (warmup coupling, rank-1 scalar destruction), the algorithm in math notation, the analytical claims (warmup-independence, bounded scalar variance, ~55% AdamW memory), what Liger doesn't solve, and the optimizer-zoo positioning table.

## When to reach for Liger

Reach for Liger when **all three** of these conditions hold:

  1. The module has **mixed dimensionality** — matrix params + 1-D/scalar params in the same parameter group.
  2. The matrix gradients are **well-conditioned by upstream normalization** (softmax, RMSNorm, LayerNorm).
  3. You want **immediate operation without warmup gating** — either because the architecture has its own cold-start mechanism, or because you're stacking LR warmups and don't want optimizer warmup compounding.

Do not reach for Liger when:

  - The module is matrix-only with ill-conditioned matrix gradients → Shampoo / RACASO / RAMuogi.
  - The module is scalar-only → Yogi directly.
  - You need spectral structure tracking → Muon's NS5 or Shampoo's eigendecomposition.

Liger is for the **middle** of the optimizer landscape: too sophisticated for "everything is uniform, just use AdamW," too restrained for "I need to model gradient covariance." Most transformer-derivative modules sit exactly there.

## License

MIT.

# Liger Benchmark Suite

Five-problem benchmark suite designed to isolate Liger's analytical claims against five baseline optimizers (AdamW, Lion, Yogi, plus reference family: Muogi, RAMuogi, RACASO).

## Phase status

| Phase | Status | Contents |
|---|---|---|
| **Phase 1: Scaffold** | ✅ done | `BenchProblem` ABC (`problems/base.py`), vendored Yogi (`optimizers/yogi.py`). |
| **Phase 2: Optimizers** | ✅ done | `optimizers/wrappers.py` dispatch, vendored Lion (`optimizers/lion.py`), family imports for Muogi/RAMuogi/RACASO. |
| **Phase 3: Problems** | ✅ done | `problems/p1`...`p5` implementations. |
| **Phase 4: Harness** | ✅ done | `run_bench.py` CSV emitter, `plot_bench.py` figure renderer. |
| **Phase 5: GPU sweep** | ✅ done (RunPod RTX A4500, 2026-05-22) | 240-run matrix (5 problems × 4 optimizers × 4 LRs × 3 seeds). Results in `results.csv`, log in `sweep.log`, figures in `figs/`. |

## Verification artifacts

All bench data is committed in this directory for reproducibility:

| File | Size | What |
|---|---|---|
| `results.csv` | ~1.2 MB | 240 rows, one per (problem, optimizer, lr, seed). Schema in §"CSV schema" below. |
| `sweep.log` | ~10 KB | Per-run stdout from `run_bench.py --sweep`. |
| `figs/fig1_p1_loss_curves.png` | ~88 KB | P1 mixed-dim, loss-vs-step across optimizers (log-y). |
| `figs/fig2_p2_v_hat_trajectories.png` | ~32 KB | P2 scalar burst, final `v_hat` per optimizer (log-log scatter). |
| `figs/fig3_p3_early_loss.png` | ~33 KB | P3 warmup-free, loss at step 1/5/50 bar chart. |
| `figs/fig4_p4_state_bytes.png` | ~33 KB | P4 memory, optimizer-state bytes bar chart. |
| `figs/fig5_p5_router_census.png` | ~47 KB | P5 router, telemetry vs ground-truth scatter. |
| `morpheus_v2.2_liger_telemetry.csv` | ~2 KB | Live in-production telemetry parsed from a Neo v2.2 training log (see Liger_Paper.md §9.6). |
| `morpheus_v2.2_liger_trajectory.png` | ~50 KB | Plot of the live telemetry above. |

To re-run from scratch and regenerate these files:

```bash
python bench/run_bench.py --sweep --output bench/results.csv 2>&1 | tee bench/sweep.log
python bench/plot_bench.py --input bench/results.csv --output bench/figs/
```

Numerical headline results from the 2026-05-22 run:

| Problem | Headline number |
|---|---|
| P1 mixed-dim | All 4 optimizers (adamw, lion, yogi, liger) converge to ~9.9e-3 at best LR. Liger is competitive but not statistically separable on this problem size. |
| P3 warmup-free | AdamW reaches lower step-50 loss (5.7e-2) than Liger/Lion (~1.7e-1) on this tanh-MLP regression. AdamW with β2=0.999 is already operational at step 1 here. |
| **P4 memory** | **Liger: 4,029,153,920 bytes (50.02% of AdamW's 8,055,689,280)**. Lion: 50.00%. Yogi: 100.00%. |
| **P5 router** | Liger telemetry: 6 Lion-route, 5 Yogi-route — **exact match** to hand-counted ndim ground truth. |
| P2 scalar burst | Liger's `v_hat` shows the expected `~1e5` post-burst residual; non-Liger optimizers do not expose `v_hat` through the harness's telemetry interface. |

## Claim ledger

Each problem isolates one analytical claim from `Liger_Paper.md` §4 / §9. The structure is one-claim-per-problem so empirical findings map directly to paper sections.

### P1 — Mixed-Dim Module (headline)

**File:** `problems/p1_mixed_dim_module.py`

**Claim:** A single Liger instance beats *every* single-rule baseline on a module with mixed parameter shapes, because each baseline pays a tax on the regime it doesn't natively handle.

**Setup:** Module with one 64×64 matrix (Lion target), one 64-dim bias (Yogi target), one scalar gate (Yogi target). Loss: `||A·x + b·gate - y||²` on synthetic regression data.

**Metric:** Steps to reach `converged_tol`; lower is better. The baseline-comparison matrix is the headline number — Liger should win on at least 2 of {AdamW, Lion, Yogi} and be competitive with the third.

**Failure mode to watch:** if Liger ties Lion exactly, the Yogi path isn't carrying its weight (check that the scalar gate's gradient is actually bursty enough to matter on this problem; if not, the problem isn't a fair test of the dispatch).

### P2 — Scalar Burst

**File:** `problems/p2_scalar_burst.py`

**Claim:** Yogi's bounded `v_t` update recovers from rank-1 gradient bursts; Adam-family `v_t` does not.

**Setup:** Scalar gate alone. Gradient stream is mostly small (`~1.0`) with periodic bursts (`~1e3` every 50 steps).

**Metric:** `v_hat` trajectory across 500 steps, plotted log-scale. Liger and Yogi-only should show post-burst recovery; AdamW should show persistent accumulator inflation for 1/(1-β2) ≈ 1000 steps after each burst.

**This is the figure that sells the Yogi-on-scalars choice** in §4.2 of the paper. Pre-registered prediction: Liger's `v_hat` returns to baseline within ~50 steps of each burst; AdamW's `v_hat` is still inflated when the next burst arrives.

### P3 — Warmup-Free

**File:** `problems/p3_warmup_free.py`

**Claim:** Liger's matrix-route is operational from step 1 without an LR warmup, because Lion's sign-momentum has no `v_t` accumulator to wait on.

**Setup:** Small MLP (matrices only, to isolate the Lion-path claim), no LR warmup. Measure loss at step 1, step 5, step 50.

**Metric:** Loss at each checkpoint. Liger should match Lion-only (both warmup-independent) and beat AdamW (still in its β2 warmup).

**Reading the result:** This problem demonstrates that Liger *inherits* Lion's warmup-independence on the matrix path. It does not by itself argue against AdamW (which can also work without an LR warmup if `β2` is tuned aggressively low), but combined with P1 it shows Liger gets warmup-independence *for free* without paying P1's mixed-dim tax.

### P4 — Memory Footprint

**File:** `problems/p4_memory_scaling.py`

**Claim:** Liger's optimizer state is approximately 55% of AdamW's on transformer-derivative architectures.

**Setup:** Synthetic 1B-parameter-equivalent module (tiled, doesn't actually run forward at scale — just instantiates optimizer state). Measure `sum(tensor.numel() · tensor.element_size())` across all buffers.

**Metric:** Total optimizer-state bytes, normalized to AdamW = 100%.

**Pre-registered prediction:** Liger lands at 50-55% on pure-transformer parameter distributions, 55-65% on RMSNorm-heavy / multi-gate architectures (where the 1-D parameter fraction is materially larger).

This problem is **CPU-runnable locally** — it only allocates state, doesn't run training — and is a useful pre-RunPod gate.

### P5 — Router Correctness

**File:** `problems/p5_router_correctness.py`

**Claim:** The router places each parameter on the route its `ndim` would predict, and the telemetry counts match the ground truth.

**Setup:** Mixed-ndim model with hand-counted parameters across `ndim ∈ {0, 1, 2, 3}`. After 1 step, query `get_telemetry()["num_2d_params"]` and `["num_1d_params"]`.

**Metric:** Counts match `sum(p.ndim >= 2 for p in params)` and `sum(p.ndim < 2 for p in params)`. Binary pass/fail.

**Why this matters:** Liger's contribution is the dispatch. A broken router would invalidate every other claim. This is a fast sanity gate that runs in CPU seconds and is the **first** thing to check in any new environment.

## Baseline optimizers

`optimizers/wrappers.py` exposes `build_optimizer(name, params, lr)` for:

| Name | Source | Memory | Warmup |
|---|---|---|---|
| `adam` | `torch.optim.Adam`, `betas=(0.9, 0.999)` | 100% | β2 |
| `adamw` | `torch.optim.AdamW`, `betas=(0.9, 0.999)`, `wd=0.01` | 100% | β2 |
| `yogi` | vendored from `optimizers/yogi.py` | 100% | β2 |
| `lion` | vendored from `lion-pytorch`, `betas=(0.9, 0.99)` | 50% | none |
| `muogi` | imported from sibling `Muogi/muogi.py` | 100% | β2 |
| `ramuogi` | imported from sibling `Muogi/ramuogi.py` | 100% | β2 + L4 + LR |
| `racaso` | imported from sibling `RACASO/racaso.py` | 100%+ | β2 + L4 + LR |
| `liger` | imported from parent `liger.py` | ~55% | **none on Lion route** |

Hyperparameters are pinned per the canonical config block in `optimizers/README.md`. Only `lr` and `seed` vary in the sweep matrix.

## Sweep matrix

The canonical sweep is:

  - **5 problems** × **8 optimizers** × **3 seeds** × **4 LRs** = **480 runs**.
  - LRs swept per optimizer (Lion-family takes lower LRs): `{1e-5, 3e-5, 1e-4, 3e-4}` for Liger/Lion; `{1e-4, 3e-4, 1e-3, 3e-3}` for AdamW/Yogi; family defaults for Muogi/RAMuogi/RACASO.
  - Each run emits one CSV row.

Run with:

```bash
python run_bench.py --sweep --output results.csv
```

Single run:

```bash
python run_bench.py --problem p1 --optimizer liger --lr 1e-4 --seed 0
```

## CSV schema

Columns:

```
problem, optimizer, lr, seed, steps, convergence_step, final_loss,
wall_clock_per_step_us, nan_count, num_2d_params, num_1d_params,
last_max_v_hat, last_min_v_hat, last_max_momentum_norm,
optimizer_state_bytes, loss_trajectory
```

`loss_trajectory` is semicolon-separated floats (not comma, to avoid CSV-quoting hell). The trajectory column makes the §9 paper figures reproducible from CSV alone — `plot_bench.py` reads only this file.

## Pre-RunPod gates

Before launching the full sweep on leased GPU, all of the following must pass locally:

  1. **`pytest test_liger.py -v`** — 22/22 green.
  2. **P5 router correctness** runs and passes locally (CPU): `python run_bench.py --problem p5 --optimizer liger --lr 1e-4 --seed 0`.
  3. **P4 memory** runs locally (CPU, allocator-only): `python run_bench.py --problem p4 --optimizer liger --lr 1e-4 --seed 0`. Spot-check the bytes-ratio against the analytical prediction from §4.3.
  4. **P1 smoke** runs locally on a tiny problem size for {adamw, lion, yogi, liger} without NaN: `python run_bench.py --problem p1 --optimizer {name} --lr 1e-4 --seed 0`. This validates the wrapper dispatch, not the headline claim — convergence on CPU may be slow.

What we cannot verify locally:

  - GPU correctness (the dev environment is AMD/ROCm and crashes on CUDA paths).
  - Mixed-precision (bf16 / fp16) behavior.
  - At-scale memory measurement (the 1B-equivalent P4 only validates the *formula* on CPU; real GB measurements need a real allocator).

These three are deferred to the first RunPod session and gated by the pre-RunPod checklist above.

## RunPod launch checklist

When leased GPU comes online:

  1. SSH in, clone `MetaforeAI/Liger`, `MetaforeAI/Muogi`, `MetaforeAI/RACASO` side-by-side.
  2. `pip install lion-pytorch torch matplotlib` (Liger has no other deps).
  3. Re-run `pytest test_liger.py -v` on GPU — should be 22/22 green again.
  4. Re-run pre-RunPod gates 2-4 on a tiny GPU problem size.
  5. `python run_bench.py --sweep --output results.csv` (estimate: 4-8 hours on H100 for the full 480-run matrix).
  6. `python plot_bench.py --input results.csv --output figs/`.
  7. Paste numbers from `results.csv` into `Liger_Paper.md` §9 tables (P1, P3) and reference figures in §9.2, §9.4, §9.5.
  8. Render PDF: `pandoc Liger_Paper.md -o Liger_Paper.pdf --pdf-engine=xelatex`.

## License

MIT (same as parent project).

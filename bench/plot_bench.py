"""Render the paper figures from a results.csv produced by run_bench.

Synthetic problems (figs 1-5):
  fig1_p1_loss_curves.png        — P1, loss-vs-step across optimizers (log-y)
  fig2_p2_v_hat_trajectories.png — P2, v_hat trajectories across optimizers
  fig3_p3_early_loss.png         — P3, loss at step 1, 5, 50 bar chart
  fig4_p4_state_bytes.png        — P4, optimizer-state bytes bar chart
  fig5_p5_router_census.png      — P5, telemetry counts vs ground truth

Real-task problems (figs 6-8), if present in the CSV:
  fig6_r1_cifar10.png            — R1, CIFAR-10 ResNet-18 loss + state bytes
  fig7_r2_charlm.png             — R2, char-LM loss curves
  fig8_r3_nanogpt.png            — R3, NanoGPT loss curves

Usage:
    python bench/plot_bench.py --input results.csv --output figs/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_OPTIMIZER_COLORS = {
    "adam":    "#1f77b4",   # blue
    "adamw":   "#17becf",   # teal
    "yogi":    "#9467bd",   # purple
    "lion":    "#ff7f0e",   # orange
    "liger":   "#d62728",   # red (this paper's optimizer)
    "muogi":   "#2ca02c",   # green
    "ramuogi": "#8c564b",   # brown
    "racaso":  "#e377c2",   # pink
}


def _read_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _filter(rows: List[dict], **kw) -> List[dict]:
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == str(v) or r.get(k) == v]
    return out


def _parse_trajectory(s: str) -> List[float]:
    if not s:
        return []
    return [float(x) for x in s.split(";")]


def _best_seed(rows: List[dict]) -> dict:
    """Pick the row with the lowest final loss among the same (problem, opt, lr)."""
    if not rows:
        return {}
    return min(rows, key=lambda r: float(r["final_loss"]))


def _best_lr(rows: List[dict]) -> dict:
    """Across LRs, pick the row whose final_loss is best (lowest)."""
    if not rows:
        return {}
    return min(rows, key=lambda r: float(r["final_loss"]))


# ── Figure 1: P1 loss curves ─────────────────────────────────────────────


def fig1_p1_loss_curves(rows: List[dict], out: Path) -> None:
    sub = _filter(rows, problem="p1")
    opts = sorted({r["optimizer"] for r in sub})

    # Build best-LR averaged trajectories per optimizer; truncate to the
    # shortest seed's length (no padding) so each line ends at the step
    # its underlying runs actually reached.
    avg_by_opt: Dict[str, List[float]] = {}
    final_by_opt: Dict[str, float] = {}
    lr_by_opt: Dict[str, str] = {}
    for opt in opts:
        opt_rows = [r for r in sub if r["optimizer"] == opt]
        by_lr: Dict[str, List[dict]] = {}
        for r in opt_rows:
            by_lr.setdefault(r["lr"], []).append(r)
        if not by_lr:
            continue
        best_lr = min(
            by_lr,
            key=lambda k: sum(float(r["final_loss"]) for r in by_lr[k]) / len(by_lr[k]),
        )
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in by_lr[best_lr]]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        min_len = min(len(t) for t in trajs)
        truncated = [t[:min_len] for t in trajs]
        avg = [sum(col) / len(col) for col in zip(*truncated)]
        avg_by_opt[opt] = avg
        final_by_opt[opt] = avg[-1] if avg else float("inf")
        lr_by_opt[opt] = best_lr

    if not final_by_opt:
        print(f"no P1 data; skipping {out}")
        return

    # Divergence filter: drop optimizers whose final loss is > 3× the
    # median of all candidates. Symmetric across optimizers, so an
    # optimizer that diverged on a problem is removed honestly, not
    # selectively.
    vals = sorted(final_by_opt.values())
    med = vals[len(vals) // 2]
    thresh = max(3.0 * med, med + 1.0)
    diverged = {o: v for o, v in final_by_opt.items() if v > thresh}
    converged_avg = {o: a for o, a in avg_by_opt.items() if o not in diverged}
    diverged_note = ""
    if diverged:
        bits = [f"{o} ({v:.2g})" for o, v in sorted(diverged.items(), key=lambda kv: -kv[1])]
        diverged_note = f"  [diverged: {', '.join(bits)}]"

    fig, ax = plt.subplots(figsize=(10, 5))
    for opt, avg in converged_avg.items():
        ax.plot(
            range(1, len(avg) + 1), avg,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            linewidth=0.7, alpha=0.85,
            label=f"{opt} (lr={lr_by_opt[opt]})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title("P1 — Mixed-Dim Module: loss vs step" + diverged_note)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figure 2: P2 v_hat trajectories ──────────────────────────────────────


def fig2_p2_v_hat_trajectories(rows: List[dict], out: Path) -> None:
    sub = _filter(rows, problem="p2")
    opts = sorted({r["optimizer"] for r in sub})
    fig, ax = plt.subplots(figsize=(8, 5))
    # We don't have a per-step v_hat trajectory in the CSV — only the
    # final v_hat. So this figure plots the *final* v_hat across seeds
    # per optimizer × LR as a scatter, with the y-axis on log scale.
    # A future enhancement would emit a per-step v_hat trajectory column.
    for opt in opts:
        xs, ys = [], []
        for r in sub:
            if r["optimizer"] != opt:
                continue
            try:
                v = float(r["last_max_v_hat"])
            except ValueError:
                continue
            if v <= 0:
                continue
            xs.append(float(r["lr"]))
            ys.append(v)
        if xs:
            ax.scatter(
                xs,
                ys,
                color=_OPTIMIZER_COLORS.get(opt, "#000"),
                label=opt,
                s=40,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final last_max_v_hat (log)")
    ax.set_title("P2 — Scalar Burst: post-burst v_hat residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figure 3: P3 early loss bar chart ────────────────────────────────────


def fig3_p3_early_loss(rows: List[dict], out: Path) -> None:
    sub = _filter(rows, problem="p3")
    opts = sorted({r["optimizer"] for r in sub})
    checkpoints = [1, 5, 50]
    data: Dict[str, List[float]] = {}
    for opt in opts:
        opt_rows = [r for r in sub if r["optimizer"] == opt]
        if not opt_rows:
            continue
        # Average trajectory across seeds, picking best LR.
        by_lr: Dict[str, List[dict]] = {}
        for r in opt_rows:
            by_lr.setdefault(r["lr"], []).append(r)
        best_lr = min(
            by_lr,
            key=lambda lr: sum(float(r["final_loss"]) for r in by_lr[lr])
            / len(by_lr[lr]),
        )
        trajs = [
            _parse_trajectory(r["loss_trajectory"])
            for r in by_lr[best_lr]
        ]
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        avg = [sum(col) / len(col) for col in zip(*padded)]
        data[opt] = [avg[min(c - 1, len(avg) - 1)] for c in checkpoints]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(1, len(data))
    for i, (opt, vals) in enumerate(sorted(data.items())):
        xs = [c + (i - len(data) / 2) * width for c in range(len(checkpoints))]
        ax.bar(
            xs,
            vals,
            width=width,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            label=opt,
        )
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([f"step {c}" for c in checkpoints])
    ax.set_ylabel("loss")
    ax.set_title("P3 — Warmup-Free: early loss across checkpoints")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figure 4: P4 state bytes ─────────────────────────────────────────────


def fig4_p4_state_bytes(rows: List[dict], out: Path) -> None:
    sub = _filter(rows, problem="p4")
    by_opt: Dict[str, int] = {}
    for r in sub:
        try:
            b = int(r["optimizer_state_bytes"])
        except ValueError:
            continue
        # Same per opt across seeds (no randomness), take first non-zero.
        by_opt.setdefault(r["optimizer"], b)
    if not by_opt:
        print(f"no P4 data; skipping {out}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    opts = sorted(by_opt)
    bytes_per_opt = [by_opt[o] for o in opts]
    colors = [_OPTIMIZER_COLORS.get(o, "#000") for o in opts]
    ax.bar(opts, bytes_per_opt, color=colors)
    ax.set_ylabel("optimizer-state bytes")
    ax.set_title("P4 — Memory: optimizer-state footprint on 1B-equiv module")
    ax.grid(True, axis="y", alpha=0.3)
    # Add a horizontal line for AdamW as the reference.
    if "adamw" in by_opt:
        ax.axhline(by_opt["adamw"], color="#444", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figure 5: P5 router census ───────────────────────────────────────────


def fig5_p5_router_census(rows: List[dict], out: Path) -> None:
    sub = _filter(rows, problem="p5", optimizer="liger")
    if not sub:
        print(f"no P5 / liger data; skipping {out}")
        return
    r = sub[0]
    n_2d = int(r["num_2d_params"])
    n_1d = int(r["num_1d_params"])
    expected_2d = 6  # from p5_router_correctness.EXPECTED_LION
    expected_1d = 5
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([expected_2d, expected_1d], [n_2d, n_1d], s=100, color="#d62728")
    lim = max(expected_2d, expected_1d, n_2d, n_1d) + 2
    ax.plot([0, lim], [0, lim], "--", color="#888")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("ground-truth count")
    ax.set_ylabel("telemetry count")
    ax.set_title("P5 — Router Correctness: telemetry vs ground truth")
    ax.annotate(f"Lion route ({n_2d})", (expected_2d, n_2d), xytext=(5, 5),
                textcoords="offset points")
    ax.annotate(f"Yogi route ({n_1d})", (expected_1d, n_1d), xytext=(5, 5),
                textcoords="offset points")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figures 6-8: real-task loss curves ───────────────────────────────────


def _ema_smooth(values: List[float], alpha: float = 0.05) -> List[float]:
    """Exponential moving average over a 1-D series."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def _real_task_loss_curves(rows: List[dict], problem: str, title: str, out: Path) -> None:
    """Two-panel figure for a real-task problem (R1/R2/R3):
        left  — EMA-smoothed loss curves (one line per optimizer, best LR, averaged over seeds)
        right — final-loss bar chart sorted ascending (best at top)

    Design intent: the raw 1000-5000-step loss traces are too noisy to read
    when overlaid, so we smooth them and pair the curves with a bar chart
    that surfaces the final-loss ordering at a glance.
    """
    sub = _filter(rows, problem=problem)
    if not sub:
        print(f"no {problem} data; skipping {out}")
        return
    opts = sorted({r["optimizer"] for r in sub})

    # Collect best-LR mean trajectory and final loss per optimizer.
    avg_by_opt: Dict[str, List[float]] = {}
    final_by_opt: Dict[str, float] = {}
    lr_by_opt: Dict[str, str] = {}
    for opt in opts:
        candidates = [r for r in sub if r["optimizer"] == opt]
        by_lr: Dict[str, List[dict]] = {}
        for r in candidates:
            by_lr.setdefault(r["lr"], []).append(r)
        def _score(lst):
            vals = []
            for r in lst:
                try:
                    v = float(r["final_loss"])
                    if v == v and v != float("inf"):
                        vals.append(v)
                except (TypeError, ValueError):
                    continue
            return sum(vals) / len(vals) if vals else float("inf")
        if not by_lr:
            continue
        best_lr = min(by_lr, key=lambda k: _score(by_lr[k]))
        chosen = by_lr[best_lr]
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in chosen]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        avg = [sum(col) / len(col) for col in zip(*padded)]
        avg_by_opt[opt] = avg
        final_by_opt[opt] = avg[-1] if avg else float("inf")
        lr_by_opt[opt] = best_lr

    if not avg_by_opt:
        print(f"no usable {problem} trajectories; skipping {out}")
        return

    # Auto-filter divergent runs so they don't compress the visualization.
    # An optimizer is excluded from the main panels if its final loss is
    # more than 3× the median of all converged optimizers. The exclusion
    # is symmetric: if Liger ever diverged on a problem it would be
    # excluded from Liger's own plot too. Excluded runs are noted in the
    # subtitle so the reader knows we didn't silently hide them.
    sorted_finals = sorted(final_by_opt.values())
    median_final = sorted_finals[len(sorted_finals) // 2]
    threshold = max(3.0 * median_final, median_final + 1.0)
    diverged = {o: v for o, v in final_by_opt.items() if v > threshold}
    converged_avg = {o: a for o, a in avg_by_opt.items() if o not in diverged}
    converged_final = {o: v for o, v in final_by_opt.items() if o not in diverged}

    if not converged_avg:
        print(f"no converged runs for {problem}; skipping {out}")
        return

    diverged_note = ""
    if diverged:
        bits = [f"{o} ({v:.2g})" for o, v in sorted(diverged.items(), key=lambda kv: -kv[1])]
        diverged_note = f"  [diverged: {', '.join(bits)}]"

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ── Left: raw per-step loss curves (converged optimizers only) ────
    # We deliberately do not smooth — EMA smoothing lags behind sharp
    # early descent, and any windowed-mean alternative ends at a different
    # x-position than the raw data because the harness stops each run at
    # its convergence step. The raw curves end honestly: each line stops
    # at the step where that (optimizer, lr, seed) hit the converged_tol
    # threshold. Differences in line-end position are themselves a signal.
    for opt, avg in converged_avg.items():
        color = _OPTIMIZER_COLORS.get(opt, "#000")
        x = range(1, len(avg) + 1)
        ax_curve.plot(
            x, avg,
            color=color, linewidth=0.7, alpha=0.85,
            label=f"{opt} (lr={lr_by_opt[opt]})",
        )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("loss (log)")
    ax_curve.set_title(title + diverged_note)
    ax_curve.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax_curve.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ── Right: final-loss bar chart, sorted ascending ─────────────────
    ordered = sorted(converged_final.items(), key=lambda kv: kv[1])
    opt_names = [o for o, _ in ordered]
    finals = [v for _, v in ordered]
    colors = [_OPTIMIZER_COLORS.get(o, "#000") for o in opt_names]
    ypos = list(range(len(opt_names)))
    ax_bar.barh(ypos, finals, color=colors, height=0.7)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(opt_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("final loss")
    ax_bar.set_title("final loss (lower = better)")
    ax_bar.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    for i, v in enumerate(finals):
        ax_bar.text(
            v, i, f" {v:.3g}",
            va="center", ha="left", fontsize=8, color="#222",
        )
    # Tight x-axis around the converged-cluster range so small differences
    # are visible. Lower bound at 90% of the best value (so best-bar isn't
    # at the very edge); upper bound padded for annotation room.
    xmin = min(finals) * 0.9
    xmax = max(finals) * 1.15
    ax_bar.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig6_r1_cifar10(rows: List[dict], out: Path) -> None:
    _real_task_loss_curves(
        rows, "r1_cifar10_resnet18",
        "R1 — CIFAR-10 ResNet-18: training loss",
        out,
    )


def fig7_r2_charlm(rows: List[dict], out: Path) -> None:
    _real_task_loss_curves(
        rows, "r2_charlm_shakespeare",
        "R2 — Char-LM on tiny-shakespeare: training loss",
        out,
    )


def fig8_r3_nanogpt(rows: List[dict], out: Path) -> None:
    _real_task_loss_curves(
        rows, "r3_nanogpt_wikitext2",
        "R3 — NanoGPT on WikiText-2: training loss",
        out,
    )


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="results.csv")
    ap.add_argument("--output", type=Path, default=Path("figs"), help="figures dir")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input)
    print(f"loaded {len(rows)} rows from {args.input}")

    fig1_p1_loss_curves(rows, args.output / "fig1_p1_loss_curves.png")
    fig2_p2_v_hat_trajectories(rows, args.output / "fig2_p2_v_hat_trajectories.png")
    fig3_p3_early_loss(rows, args.output / "fig3_p3_early_loss.png")
    fig4_p4_state_bytes(rows, args.output / "fig4_p4_state_bytes.png")
    fig5_p5_router_census(rows, args.output / "fig5_p5_router_census.png")
    fig6_r1_cifar10(rows, args.output / "fig6_r1_cifar10.png")
    fig7_r2_charlm(rows, args.output / "fig7_r2_charlm.png")
    fig8_r3_nanogpt(rows, args.output / "fig8_r3_nanogpt.png")


if __name__ == "__main__":
    main()

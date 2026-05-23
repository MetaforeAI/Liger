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
    "adam": "#888888",
    "adamw": "#444444",
    "yogi": "#1f77b4",
    "lion": "#ff7f0e",
    "liger": "#d62728",
    "muogi": "#2ca02c",
    "ramuogi": "#9467bd",
    "racaso": "#8c564b",
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt in opts:
        opt_rows = [r for r in sub if r["optimizer"] == opt]
        # Pick best LR by final loss across seeds (avg seeds first).
        by_lr: Dict[str, List[dict]] = {}
        for r in opt_rows:
            by_lr.setdefault(r["lr"], []).append(r)
        best_lr = None
        best_score = float("inf")
        for lr, lr_rows in by_lr.items():
            score = sum(float(r["final_loss"]) for r in lr_rows) / len(lr_rows)
            if score < best_score:
                best_score = score
                best_lr = lr
        chosen = [r for r in opt_rows if r["lr"] == best_lr]
        if not chosen:
            continue
        # Average trajectory across seeds.
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in chosen]
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        avg = [sum(col) / len(col) for col in zip(*padded)]
        ax.plot(
            range(1, len(avg) + 1),
            avg,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            label=f"{opt} (lr={best_lr})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title("P1 — Mixed-Dim Module: loss vs step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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


def _real_task_loss_curves(rows: List[dict], problem: str, title: str, out: Path) -> None:
    """Plot loss-vs-step for a real-task problem (R1/R2/R3).

    One line per optimizer, averaged across seeds, best LR per optimizer.
    """
    import numpy as np

    sub = _filter(rows, problem=problem)
    if not sub:
        print(f"no {problem} data; skipping {out}")
        return
    opts = sorted({r["optimizer"] for r in sub})
    fig, ax = plt.subplots(figsize=(10, 5))
    for opt in opts:
        candidates = [r for r in sub if r["optimizer"] == opt]
        # Find best LR by mean final loss.
        by_lr: Dict[str, List[dict]] = {}
        for r in candidates:
            by_lr.setdefault(r["lr"], []).append(r)
        def _score(lst):
            vals = []
            for r in lst:
                try:
                    v = float(r["final_loss"])
                    if v == v and v != float("inf"):  # not NaN/Inf
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
        ax.plot(
            range(1, len(avg) + 1),
            avg,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            label=f"{opt} (lr={best_lr})",
            linewidth=1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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

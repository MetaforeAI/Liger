"""Liger benchmark harness.

Runs a single (problem, optimizer, lr, seed) configuration or a full
sweep. Emits a CSV with per-step Liger telemetry and the full loss
trajectory.

Usage:
    # Single run.
    python bench/run_bench.py --problem p1 --optimizer liger --lr 1e-4 --seed 0

    # Full sweep (480 runs).
    python bench/run_bench.py --sweep --output results.csv

CSV schema (one row per run):
    problem, optimizer, lr, seed, steps, convergence_step, final_loss,
    wall_clock_per_step_us, nan_count, num_2d_params, num_1d_params,
    last_max_v_hat, last_min_v_hat, last_max_momentum_norm,
    optimizer_state_bytes, loss_trajectory
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Make ``bench.*`` imports work whether run as a script or as a module.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from bench.optimizers.wrappers import KNOWN_OPTIMIZERS, build_optimizer
from bench.problems.base import BenchProblem
from bench.problems.p1_mixed_dim_module import P1MixedDimModule
from bench.problems.p2_scalar_burst import P2ScalarBurst
from bench.problems.p3_warmup_free import P3WarmupFree
from bench.problems.p4_memory_scaling import (
    P4MemoryScaling,
    measure_optimizer_state_bytes,
)
from bench.problems.p5_router_correctness import P5RouterCorrectness
from bench.problems.r1_cifar10_resnet18 import R1Cifar10ResNet18
from bench.problems.r2_charlm_shakespeare import R2CharLMShakespeare
from bench.problems.r3_nanogpt_wikitext2 import R3NanoGPTWikitext2


_PROBLEMS = {
    "p1": P1MixedDimModule,
    "p2": P2ScalarBurst,
    "p3": P3WarmupFree,
    "p4": P4MemoryScaling,
    "p5": P5RouterCorrectness,
    "r1_cifar10_resnet18": R1Cifar10ResNet18,
    "r2_charlm_shakespeare": R2CharLMShakespeare,
    "r3_nanogpt_wikitext2": R3NanoGPTWikitext2,
}

_CSV_COLUMNS = [
    "problem",
    "optimizer",
    "lr",
    "seed",
    "steps",
    "convergence_step",
    "final_loss",
    "wall_clock_per_step_us",
    "nan_count",
    "num_2d_params",
    "num_1d_params",
    "last_max_v_hat",
    "last_min_v_hat",
    "last_max_momentum_norm",
    "optimizer_state_bytes",
    "loss_trajectory",
]


def _build_problem(name: str, seed: int, device: str = "cpu") -> BenchProblem:
    if name not in _PROBLEMS:
        raise ValueError(
            f"unknown problem '{name}'; known: {sorted(_PROBLEMS)}"
        )
    return _PROBLEMS[name](seed, device=device)


def _liger_telemetry(opt: torch.optim.Optimizer) -> dict:
    """Return Liger telemetry if available, else zero-filled defaults."""
    fn = getattr(opt, "get_telemetry", None)
    if fn is None:
        return {
            "num_2d_params": 0,
            "num_1d_params": 0,
            "last_max_v_hat": 0.0,
            "last_min_v_hat": 0.0,
            "last_max_momentum_norm": 0.0,
        }
    t = fn()
    return {
        "num_2d_params": t.get("num_2d_params", 0),
        "num_1d_params": t.get("num_1d_params", 0),
        "last_max_v_hat": t.get("last_max_v_hat", 0.0),
        "last_min_v_hat": t.get("last_min_v_hat", 0.0),
        "last_max_momentum_norm": t.get("last_max_momentum_norm", 0.0),
    }


def run_one(
    problem_name: str,
    optimizer_name: str,
    lr: float,
    seed: int,
    max_steps: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    """Run one (problem, optimizer, lr, seed) configuration.

    Returns a dict matching the CSV row schema.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    problem = _build_problem(problem_name, seed, device=device)
    params = problem.init_params()
    opt = build_optimizer(optimizer_name, params, lr)

    cap = max_steps if max_steps is not None else problem.max_steps

    trajectory: List[float] = []
    nan_count = 0
    convergence_step = -1
    use_cuda = (device == "cuda")
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, cap + 1):
        loss, grads = problem.loss_and_grad(params)
        # Wire grads into params (autograd may have already done this if
        # forward() was used; explicit assignment makes the analytic-grad
        # case work uniformly).
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
        for p in params:
            p.grad = None

        if math.isnan(loss) or math.isinf(loss):
            nan_count += 1
            # Record NaN as -1.0 in the trajectory; the analysis side
            # should treat NaNs as DNF.
            trajectory.append(-1.0)
        else:
            trajectory.append(float(loss))

        if convergence_step < 0 and problem.converged(loss, step):
            convergence_step = step
            # For most problems we stop here; for P3 and P4 the converged
            # criterion is never reached intentionally (we want the full
            # trajectory) — those are handled by their own ``converged``.
            if problem.converged(loss, step):
                break

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_us = (t1 - t0) * 1e6
    steps_run = len(trajectory)
    per_step_us = elapsed_us / max(1, steps_run)

    tel = _liger_telemetry(opt)
    state_bytes = measure_optimizer_state_bytes(opt)

    return {
        "problem": problem_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "steps": steps_run,
        "convergence_step": convergence_step,
        "final_loss": trajectory[-1] if trajectory else float("nan"),
        "wall_clock_per_step_us": round(per_step_us, 2),
        "nan_count": nan_count,
        "num_2d_params": tel["num_2d_params"],
        "num_1d_params": tel["num_1d_params"],
        "last_max_v_hat": tel["last_max_v_hat"],
        "last_min_v_hat": tel["last_min_v_hat"],
        "last_max_momentum_norm": tel["last_max_momentum_norm"],
        "optimizer_state_bytes": state_bytes,
        "loss_trajectory": ";".join(f"{x:.6e}" for x in trajectory),
    }


def _write_row(out_path: Optional[Path], row: dict, append: bool) -> None:
    if out_path is None:
        # Pretty-print to stdout, dropping the trajectory column for
        # readability.
        printable = {k: v for k, v in row.items() if k != "loss_trajectory"}
        print("  " + "  ".join(f"{k}={v}" for k, v in printable.items()))
        return
    mode = "a" if append and out_path.exists() else "w"
    write_header = mode == "w"
    with out_path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Sweep matrix ─────────────────────────────────────────────────────────

# Canonical LR sweeps per optimizer family (Lion-family wants ~3-10x
# lower LRs than Adam-family).
_LR_SWEEP = {
    "adam":    [1e-4, 3e-4, 1e-3, 3e-3],
    "adamw":   [1e-4, 3e-4, 1e-3, 3e-3],
    "yogi":    [1e-4, 3e-4, 1e-3, 3e-3],
    "lion":    [1e-5, 3e-5, 1e-4, 3e-4],
    "liger":   [1e-5, 3e-5, 1e-4, 3e-4],
    "muogi":   [3e-5, 1e-4, 3e-4, 1e-3],
    "ramuogi": [3e-5, 1e-4, 3e-4, 1e-3],
    "racaso":  [3e-5, 1e-4, 3e-4, 1e-3],
}

_DEFAULT_SWEEP_OPTIMIZERS = (
    "adam", "adamw", "yogi", "lion", "liger", "muogi", "ramuogi", "racaso",
)
_DEFAULT_SWEEP_SEEDS = (0, 1, 2)
_DEFAULT_SWEEP_PROBLEMS = (
    "p1", "p2", "p3", "p4", "p5",
    "r1_cifar10_resnet18",
    "r2_charlm_shakespeare",
    "r3_nanogpt_wikitext2",
)
# Real-task problems get a reduced LR/seed cardinality because each run
# is much more expensive (CIFAR-10 ResNet, char-LM, NanoGPT). The
# reduction is per-problem so the synthetic problems still get the full
# matrix.
_REAL_TASK_PROBLEMS = ("r1_cifar10_resnet18", "r2_charlm_shakespeare", "r3_nanogpt_wikitext2")
_REAL_TASK_LR_OVERRIDE = {
    "adam":    [1e-3],
    "adamw":   [1e-3],
    "yogi":    [1e-3],
    "lion":    [3e-4],
    "liger":   [3e-4],
    "muogi":   [3e-4],
    "ramuogi": [3e-4],
    "racaso":  [3e-4],
}
_REAL_TASK_SEEDS = (0, 1)


def run_sweep(
    out_path: Path,
    problems: Tuple[str, ...] = _DEFAULT_SWEEP_PROBLEMS,
    optimizers: Tuple[str, ...] = _DEFAULT_SWEEP_OPTIMIZERS,
    seeds: Tuple[int, ...] = _DEFAULT_SWEEP_SEEDS,
    device: str = "cpu",
    skip_existing: bool = False,
) -> None:
    """Run the full sweep matrix and write rows to CSV.

    For real-task problems (R1/R2/R3), the harness uses a reduced
    cardinality (one LR per optimizer family, 2 seeds) because each run
    is too expensive for a full 4-LR × 3-seed grid.
    """
    def lr_grid(problem: str, opt: str) -> List[float]:
        if problem in _REAL_TASK_PROBLEMS:
            return _REAL_TASK_LR_OVERRIDE.get(opt, [3e-4])
        return _LR_SWEEP[opt]

    def seed_grid(problem: str) -> Tuple[int, ...]:
        return _REAL_TASK_SEEDS if problem in _REAL_TASK_PROBLEMS else seeds

    total = 0
    for problem in problems:
        for opt in optimizers:
            total += len(lr_grid(problem, opt)) * len(seed_grid(problem))
    print(f"sweep: {total} runs total → {out_path} [device={device}]")
    n = 0
    for problem in problems:
        for opt in optimizers:
            for lr in lr_grid(problem, opt):
                for seed in seed_grid(problem):
                    n += 1
                    print(
                        f"[{n}/{total}] {problem} × {opt} × lr={lr} × seed={seed}",
                        flush=True,
                    )
                    try:
                        row = run_one(problem, opt, lr, seed, device=device)
                    except NotImplementedError as exc:
                        print(f"  SKIP: {exc}")
                        continue
                    except Exception as exc:
                        print(f"  ERROR: {type(exc).__name__}: {exc}")
                        continue
                    _write_row(out_path, row, append=True)


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", choices=list(_PROBLEMS), help="problem id")
    ap.add_argument("--optimizer", choices=list(KNOWN_OPTIMIZERS), help="optimizer name")
    ap.add_argument("--lr", type=float, help="learning rate")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--max-steps", type=int, help="override problem.max_steps")
    ap.add_argument("--output", type=Path, help="CSV output path")
    ap.add_argument("--sweep", action="store_true", help="run the full sweep matrix")
    ap.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for tensors (default: cpu)",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but torch.cuda.is_available() is False")

    if args.sweep:
        if args.output is None:
            args.output = Path("results.csv")
        run_sweep(args.output, device=args.device)
        return

    if not (args.problem and args.optimizer and args.lr):
        ap.error("--problem, --optimizer, and --lr are required (unless --sweep)")

    row = run_one(
        args.problem,
        args.optimizer,
        args.lr,
        args.seed,
        args.max_steps,
        device=args.device,
    )
    _write_row(args.output, row, append=False)


if __name__ == "__main__":
    main()

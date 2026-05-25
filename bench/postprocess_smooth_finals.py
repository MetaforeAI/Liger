"""Post-process ``bench/results.csv`` to recompute ``final_loss`` using
the smoothed-final-loss policy (mean of last 50 steps) for R1/R2/R3 rows.

The single-noisy-minibatch ``final_loss`` recorded by an earlier run of
``run_bench.py`` is methodologically unfair: a single trailing batch's
loss is a high-variance estimator of true final-loss, and optimizers
whose last batch happened to be easy were privileged in the head-to-head.

This script reads ``bench/results.csv``, re-derives ``final_loss`` from
the per-step ``loss_trajectory`` column using the smoothed policy in
``bench/run_bench.py::_smoothed_final_loss``, and rewrites the CSV in
place. The unsmoothed copy is preserved at
``bench/results_unsmoothed.csv``.

Only R1/R2/R3 rows are touched. Synthetic problems P1-P5 already have
deterministic-enough trajectories that the single-step final is the same
as a smoothed final.

Usage:
    python bench/postprocess_smooth_finals.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.csv"
_UNSMOOTHED_BACKUP = _HERE / "results_unsmoothed.csv"

_REAL_TASKS = {
    "r1_cifar10_resnet18",
    "r2_charlm_shakespeare",
    "r3_nanogpt_wikitext2",
}

_SMOOTH_WINDOW = 50


def _smoothed_final(trajectory: list[float]) -> float:
    if not trajectory:
        return float("nan")
    window = trajectory[-_SMOOTH_WINDOW:] if len(trajectory) >= _SMOOTH_WINDOW else trajectory
    clean = [x for x in window if x >= 0.0 and x == x and x != float("inf")]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _parse_trajectory(s: str) -> list[float]:
    if not s:
        return []
    out: list[float] = []
    for tok in s.split(";"):
        try:
            out.append(float(tok))
        except ValueError:
            out.append(float("nan"))
    return out


def main() -> None:
    if not _RESULTS.exists():
        raise SystemExit(f"missing {_RESULTS}")
    if not _UNSMOOTHED_BACKUP.exists():
        print(
            f"[warn] {_UNSMOOTHED_BACKUP.name} does not exist; "
            "copying current results.csv before rewriting."
        )
        _UNSMOOTHED_BACKUP.write_bytes(_RESULTS.read_bytes())

    with _RESULTS.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        raise SystemExit("CSV has no header")

    touched = 0
    deltas: list[tuple[str, str, str, str, float, float]] = []
    for row in rows:
        prob = row["problem"]
        if prob not in _REAL_TASKS:
            continue
        traj = _parse_trajectory(row.get("loss_trajectory", ""))
        new_final = _smoothed_final(traj)
        try:
            old_final = float(row["final_loss"])
        except (TypeError, ValueError):
            old_final = float("nan")
        if new_final != new_final:
            continue
        if abs(new_final - old_final) > 1e-12 or old_final != old_final:
            deltas.append((
                prob, row["optimizer"], row["lr"], row["seed"],
                old_final, new_final,
            ))
            row["final_loss"] = repr(new_final)
            touched += 1

    with _RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] rewrote {_RESULTS} ({touched} rows updated)")
    if deltas:
        print()
        print(f"{'problem':<28} {'opt':<10} {'lr':<8} {'seed':<4} "
              f"{'before':>10} {'after':>10}")
        print("-" * 76)
        for prob, opt, lr, seed, old, new in deltas:
            print(f"{prob:<28} {opt:<10} {lr:<8} {seed:<4} "
                  f"{old:>10.4f} {new:>10.4f}")


if __name__ == "__main__":
    main()

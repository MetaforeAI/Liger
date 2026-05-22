"""Extract Liger telemetry from a Morpheus Neo training log.

The Morpheus stack emits per-step Liger telemetry lines of the form:

    liger[rah]   step=70  n_2d=20  n_1d=29  mom_n=8.1e-03  upd_l1=1.5e+05
                 v_max=3.5e-03  v_min=1.5e-05

This script parses those lines (and the adjacent loss line that follows),
emits a tidy CSV, and renders a matplotlib figure for the paper's §9
empirical-results section.

Usage:
    python extract_morpheus_telemetry.py <log_path> [<csv_out>] [<png_out>]

Defaults:
    csv_out = morpheus_v2.2_liger_telemetry.csv
    png_out = morpheus_v2.2_liger_trajectory.png
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

LIGER_RE = re.compile(
    r"liger\[(?P<organ>\w+)\]\s+"
    r"step=(?P<step>\d+)\s+"
    r"n_2d=(?P<n_2d>\d+)\s+"
    r"n_1d=(?P<n_1d>\d+)\s+"
    r"mom_n=(?P<mom_n>[+\-0-9.eE]+)\s+"
    r"upd_l1=(?P<upd_l1>[+\-0-9.eE]+)\s+"
    r"v_max=(?P<v_max>[+\-0-9.eE]+)\s+"
    r"v_min=(?P<v_min>[+\-0-9.eE]+)"
)
LOSS_RE = re.compile(
    r"loss\s+M=(?P<loss>[+\-0-9.]+)\s+lo/hi\s+(?P<lo>[+\-0-9.]+)\s+/\s+(?P<hi>[+\-0-9.]+)"
)


def parse_log(log_path: Path) -> list[dict]:
    """Walk the log line by line, pairing each Liger telemetry line with
    the next loss line. Returns a list of dicts, one per step."""
    rows: list[dict] = []
    pending: Optional[dict] = None
    with log_path.open() as f:
        for line in f:
            m = LIGER_RE.search(line)
            if m:
                pending = m.groupdict()
                continue
            m = LOSS_RE.search(line)
            if m and pending is not None:
                pending.update(m.groupdict())
                rows.append(pending)
                pending = None
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit("no Liger telemetry rows found")
    cols = [
        "step",
        "organ",
        "n_2d",
        "n_1d",
        "mom_n",
        "upd_l1",
        "v_max",
        "v_min",
        "loss",
        "lo",
        "hi",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def render_figure(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure render")
        return

    steps = [int(r["step"]) for r in rows]
    mom = [float(r["mom_n"]) for r in rows]
    v_max = [float(r["v_max"]) for r in rows]
    v_min = [float(r["v_min"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.plot(steps, mom, marker="o", color="C0", label="Lion-path ||m_t||₂ (max across params)")
    ax.set_ylabel("momentum L2-norm")
    ax.set_title("Liger on Morpheus Neo v2.2 RAH organ — live telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(steps, v_max, marker="o", color="C1", label="Yogi-path v_hat max")
    ax.plot(steps, v_min, marker="o", color="C3", label="Yogi-path v_hat min")
    ax.set_yscale("log")
    ax.set_ylabel("v_hat (log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(steps, loss, marker="o", color="C2", label="training loss (mean)")
    ax.set_ylabel("loss")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    log_path = Path(sys.argv[1])
    csv_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("morpheus_v2.2_liger_telemetry.csv")
    png_out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("morpheus_v2.2_liger_trajectory.png")

    rows = parse_log(log_path)
    print(f"parsed {len(rows)} telemetry/loss pairs from {log_path}")
    write_csv(rows, csv_out)
    print(f"wrote {csv_out}")
    render_figure(rows, png_out)


if __name__ == "__main__":
    main()

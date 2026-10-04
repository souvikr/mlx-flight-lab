#!/usr/bin/env python3
"""
plot_losses.py - turn mlx_lm.lora logs into a loss-curve PNG.

Save the log while training:
  mlx_lm.lora ... 2>&1 | tee runs/flight-lora.log

Then:
  python labs/plot_losses.py runs/flight-lora.log                     # -> runs/flight-lora.png
  python labs/plot_losses.py runs/a.log runs/b.log -o runs/compare.png
  open runs/compare.png
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# mlx-lm 0.32 table rows:  "    25    0.142 ▼      206      4.2k"   and   "    50    val 0.037    5.16s"
TRAIN = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s*[▼▲]?\s+[0-9.]+\s+[0-9.]+k?\s*$")
VAL = re.compile(r"^\s*(\d+)\s+val\s+([0-9.]+)")
# older mlx-lm style: "Iter 10: Train loss 1.234, ..." / "Iter 50: Val loss 1.234, ..."
OLD_TRAIN = re.compile(r"Iter (\d+): Train loss ([0-9.]+)")
OLD_VAL = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")


def read_log(path):
    train, val = [], []
    for line in open(path, errors="ignore"):
        line = ANSI.sub("", line)
        if m := VAL.match(line) or OLD_VAL.search(line):
            val.append((int(m.group(1)), float(m.group(2))))
        elif m := TRAIN.match(line) or OLD_TRAIN.search(line):
            train.append((int(m.group(1)), float(m.group(2))))
    return train, val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--log-scale", action="store_true", help="log y-axis (good when loss gets tiny)")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, path in enumerate(args.logs):
        train, val = read_log(path)
        if not train and not val:
            print(f"no loss lines found in {path}")
            continue
        name, color = Path(path).stem, f"C{i}"
        if train:
            ax.plot(*zip(*train), color=color, alpha=0.5, label=f"{name} train")
        if val:
            ax.plot(*zip(*val), color=color, marker="o", lw=2, label=f"{name} val")
        last_val = f"{val[-1][1]:.3f}" if val else "-"
        print(f"{name}: {len(train)} train points, {len(val)} val points, last val loss {last_val}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    if args.log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Training (thin) vs validation (dots) loss")
    out = args.out or str(Path(args.logs[0]).with_suffix(".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}   (view it: open {out})")


if __name__ == "__main__":
    main()

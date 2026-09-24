#!/usr/bin/env python
"""
Regenerate every paper figure from the locked result tables.

Each figure script runs in its own interpreter, so matplotlib state cannot leak
between figures and a failure in one does not hide the others. Outputs land in
figures/<name>/ (PNG, SVG, PDF, one CSV per plotted panel, and <name>_data.md).

Usage
-----
  python scripts/figures/make_figures.py                 # main and supplementary figures
  python scripts/figures/make_figures.py Fig2 FigS3      # a subset
  python scripts/figures/make_figures.py --list

Commit the figure code before a final run: each data dictionary records the commit
and is marked '-dirty' if scripts/figures or the style module had uncommitted edits.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MAIN = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5"]
SUPPLEMENT = ["FigS1", "FigS2", "FigS3", "FigS4", "FigS5", "FigS6", "FigS7", "FigS8"]
PINNED_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS")


def run(name: str) -> tuple[bool, float, str]:
    script = HERE / f"fig_{name[3:].lower()}.py"
    if not script.exists():
        return False, 0.0, f"no script {script.relative_to(ROOT)}"
    env = dict(os.environ, MPLBACKEND="Agg", **{k: "1" for k in PINNED_THREADS})
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, time.time() - t0, (proc.stdout.strip().splitlines() or [""])[-1]
    err = proc.stderr.strip().splitlines()
    # the exception message (for a canvas overflow, its list of offending panels) is what needs fixing
    start = max((i for i, line in enumerate(err) if not line.startswith((" ", "Traceback"))), default=0)
    return False, time.time() - t0, "\n       ".join(err[start:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help="figure names (default: all)")
    ap.add_argument("--list", action="store_true", help="list figure names and exit")
    args = ap.parse_args()
    names = args.figures or MAIN + SUPPLEMENT
    if args.list:
        print("\n".join(MAIN + SUPPLEMENT))
        return 0
    unknown = sorted(set(names) - set(MAIN + SUPPLEMENT))
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    failures = 0
    for name in names:
        ok, secs, msg = run(name)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:<6} {secs:6.1f}s  {msg}", flush=True)
    print(f"\n{len(names) - failures}/{len(names)} figures built")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

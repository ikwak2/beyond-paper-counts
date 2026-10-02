#!/usr/bin/env python3
"""Run offline aggregate reproduction and fail on any numerical mismatch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check frozen assets without regenerating figures")
    args = parser.parse_args()
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    steps = ["scripts/validate_release.py"]
    if not args.check_only:
        steps += ["scripts/build_pri.py", "scripts/build_restricted_bounds.py",
                  "scripts/build_figure1.py", "scripts/build_figure2.py",
                  "scripts/build_distance_figures.py"]
    for step in steps:
        print(f"\n>>> {step}", flush=True)
        command = [sys.executable, str(ROOT / step)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.check_only:
        subprocess.run([sys.executable, str(ROOT / "scripts/validate_release.py"), "--generated"],
                       cwd=ROOT, env=env, check=True)
    print("\nPASS: frozen-asset checks" if args.check_only else "\nPASS: aggregate reproduction; see outputs/")


if __name__ == "__main__":
    main()

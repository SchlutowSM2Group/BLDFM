"""Regenerate all manuscript figures.

Run from the repository root:

    python runs/manuscript/generate_all.py
    python runs/manuscript/generate_all.py --tier interface
    python runs/manuscript/generate_all.py --tier low_level

Outputs are saved to plots/.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "comparison_analytic",
    "comparison_footprint_neutral",
    "comparison_footprint_stable",
    "comparison_footprint_unstable",
    "analytic_convergence_test",
    "numeric_convergence_test",
]

TIERS = ["interface", "low_level"]


def run_tier(tier, scripts):
    failed = []
    tier_dir = SCRIPT_DIR / tier

    for name in scripts:
        script_path = tier_dir / f"{name}.py"
        label = f"{tier}/{name}"
        print(f"[manuscript] Running {label} ...")
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script_path)])
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[manuscript] FAILED: {label} (exit {result.returncode})")
            failed.append(label)
        else:
            print(f"[manuscript] Done: {label} ({elapsed:.1f}s)")

    return failed


def main():
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures.")
    parser.add_argument(
        "--tier", choices=TIERS, default=None,
        help="Run only a specific tier (default: run both).",
    )
    args = parser.parse_args()

    tiers = [args.tier] if args.tier else TIERS

    failed = []
    t_total = time.time()

    for tier in tiers:
        print(f"\n[manuscript] === {tier} ===")
        failed.extend(run_tier(tier, SCRIPTS))

    total = time.time() - t_total
    n_total = len(SCRIPTS) * len(tiers)
    print(f"\n[manuscript] Total time: {total:.1f}s")

    if failed:
        print(f"[manuscript] {len(failed)} script(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"[manuscript] All {n_total} scripts completed in plots/")


if __name__ == "__main__":
    main()

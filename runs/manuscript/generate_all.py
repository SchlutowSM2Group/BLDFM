"""Regenerate all manuscript figures.

Run from the repository root:

    python -m runs.manuscript.generate_all

Outputs are saved to plots/.
"""

import subprocess
import sys
import time

SCRIPTS = [
    "comparison_analytic",
    "comparison_footprint_neutral",
    "comparison_footprint_stable",
    "comparison_footprint_unstable",
    "analytic_convergence_test",
    "numeric_convergence_test",
]


def main():
    failed = []
    t_total = time.time()

    for name in SCRIPTS:
        module = f"runs.manuscript.{name}"
        print(f"[manuscript] Running {module} ...")
        t0 = time.time()
        result = subprocess.run([sys.executable, "-m", module])
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[manuscript] FAILED: {module} (exit {result.returncode})")
            failed.append(name)
        else:
            print(f"[manuscript] Done: {module} ({elapsed:.1f}s)")

    total = time.time() - t_total
    print(f"\n[manuscript] Total time: {total:.1f}s")

    if failed:
        print(f"[manuscript] {len(failed)} script(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"[manuscript] All {len(SCRIPTS)} figures generated in plots/")


if __name__ == "__main__":
    main()

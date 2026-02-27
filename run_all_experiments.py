#!/usr/bin/env python3
"""
Run all blueprint experiments and save results.

Outputs:
- experiment_results/run_YYYYMMDD_HHMMSS/
  - sweep_<family>_N<n>.csv   (full sweep data for each family × N)
  - graphs/
    - heatmap_<family>_N<n>.png
    - scaling_<family>.png
    - phase_portraits_<family>_N500.png
  - summary.csv   (scaling slopes: family, N, selection_sharpness)

Usage:
  python run_all_experiments.py [--output-dir DIR] [--t-max T] [--seed S]
"""

import argparse
from pathlib import Path

from experiments import run_all_blueprint_experiments, N_VALUES


def main():
    parser = argparse.ArgumentParser(
        description="Run all blueprint experiments (Marschall et al. 2025) and save results."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiment_results/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=100.0,
        help="Simulation time per run (default: 100)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="RK4 timestep (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("Running all blueprint experiments...")
    print(f"  N values: {N_VALUES}")
    print(f"  t_max: {args.t_max}, dt: {args.dt}, seed: {args.seed}")
    print()

    out = run_all_blueprint_experiments(
        output_dir=args.output_dir,
        dt=args.dt,
        t_max=args.t_max,
        seed=args.seed,
    )

    print(f"Done. Results saved to: {out.resolve()}")
    print("\nGenerated files:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}")


if __name__ == "__main__":
    main()

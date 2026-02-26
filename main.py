"""
CTRNN Multi-Task Simulation - Main entry point.
Based on "A theory of multi-task computation and task selection" by Marschall et al.

Run scenarios A (limit cycle), B (bistable), C (winner-take-all).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from task_component import TaskComponent
from multi_task_network import MultiTaskNetwork
from simulation import Simulation
from visualizer import Visualizer


def run_scenario_a(
    N: int,
    R: int,
    dt: float,
    t_max: float,
    show_plots: bool,
    output_dir: Path,
    seed: int,
) -> None:
    """Scenario A: Single-task limit cycle (Fig 2A)."""
    tc = TaskComponent(D=2.2, A=[[0.8, 0.4], [-0.4, 0.8]], N=N, R=R, seed=seed)
    network = MultiTaskNetwork([tc], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    rng = np.random.default_rng(123)
    x0 = rng.standard_normal(N) * 2.0
    t, _, z_traj = sim.run(x0)
    z = z_traj[0]

    lim = max(np.abs(z).max() * 1.15, 10)
    z_range = (-lim, lim)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    viz = Visualizer(network, sim)
    viz.plot_trajectory(0, t, z, ax, color="purple")
    ax.set_xlabel(r"$z_1(t)$")
    ax.set_ylabel(r"$z_2(t)$")
    ax.set_title("A: Single-Task Limit Cycle  (P = 1, R = 2)")
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    out_path = output_dir / "figure_2a_limit_cycle.png"
    plt.savefig(out_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()


def run_scenario_b(
    N: int,
    R: int,
    dt: float,
    t_max: float,
    show_plots: bool,
    output_dir: Path,
    seed: int,
) -> None:
    """Scenario B: Single-task bistable (Fig 2B)."""
    tc = TaskComponent(D=2.0, A=[[0.5, 0.3], [0.3, 0.5]], N=N, R=R, seed=seed)
    network = MultiTaskNetwork([tc], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    rng_a = np.random.default_rng(1001)
    rng_b = np.random.default_rng(1002)
    x0a = rng_a.standard_normal(N) * 2.0
    x0b = rng_b.standard_normal(N) * 2.0

    t, _, z_traj_a = sim.run(x0a)
    _, _, z_traj_b = sim.run(x0b)

    za, zb = z_traj_a[0], z_traj_b[0]
    z_flat = np.concatenate([za[:, 0], za[:, 1], zb[:, 0], zb[:, 1]])
    lim = max(np.abs(z_flat).max() * 1.15, 10)
    z_range = (-lim, lim)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    viz = Visualizer(network, sim)
    viz.plot_trajectory(0, t, za, ax, color="red")
    viz.plot_latent_trajectory(t, zb, ax, color="red", mark_start=True)
    ax.set_xlabel(r"$z_1(t)$")
    ax.set_ylabel(r"$z_2(t)$")
    ax.set_title("B: Single-Task Stable Fixed Points  (P = 1, R = 2)")
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    out_path = output_dir / "figure_2b_bistable.png"
    plt.savefig(out_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()


def run_scenario_c(
    N: int,
    R: int,
    dt: float,
    t_max: float,
    show_plots: bool,
    output_dir: Path,
    seed: int,
) -> None:
    """Scenario C: Two-task winner-take-all (Fig 2C)."""
    tc1 = TaskComponent(D=2.2, A=[[0.8, 0.4], [-0.4, 0.8]], N=N, R=R, seed=seed)
    tc2 = TaskComponent(D=2.0, A=[[0.5, 0.3], [0.3, 0.5]], N=N, R=R, seed=seed + 1)
    network = MultiTaskNetwork([tc1, tc2], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    rng = np.random.default_rng(456)
    x0 = rng.standard_normal(N) * 2.0
    t, _, z_traj = sim.run(x0)

    z1, z2 = z_traj[0], z_traj[1]
    lim1 = max(np.abs(np.concatenate([z1[:, 0], z1[:, 1]])).max() * 1.15, 10)
    lim2 = max(np.abs(np.concatenate([z2[:, 0], z2[:, 1]])).max() * 1.15, 10)
    range1, range2 = (-lim1, lim1), (-lim2, lim2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    viz = Visualizer(network, sim)

    viz.plot_trajectory(0, t, z1, axes[0], color="purple")
    axes[0].set_xlabel(r"$z_1^{(1)}(t)$")
    axes[0].set_ylabel(r"$z_2^{(1)}(t)$")
    axes[0].set_title("C (left): Task 1 dominant  (P = 2, R = 2)")
    axes[0].set_xlim(range1)
    axes[0].set_ylim(range1)
    axes[0].set_aspect("equal")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)

    viz.plot_trajectory(1, t, z2, axes[1], color="red")
    axes[1].set_xlabel(r"$z_1^{(2)}(t)$")
    axes[1].set_ylabel(r"$z_2^{(2)}(t)$")
    axes[1].set_title("C (right): Task 2 decays to origin  (P = 2, R = 2)")
    axes[1].set_xlim(range2)
    axes[1].set_ylim(range2)
    axes[1].set_aspect("equal")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)

    plt.tight_layout()
    out_path = output_dir / "figure_2c_winner_take_all.png"
    plt.savefig(out_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTRNN Multi-Task Simulation (Marschall et al.)"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="all",
        choices=["a", "b", "c", "all"],
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save figures only, do not display",
    )
    parser.add_argument(
        "-N", "--neurons",
        type=int,
        default=2000,
        help="Number of neurons (default: 2000)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=100.0,
        help="Simulation time (default: 100)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="RK4 timestep (default: 0.05)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for figures (default: current)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task components (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    show_plots = not args.no_display
    N = args.neurons
    R = 2
    dt = float(args.dt)
    t_max = float(args.t_max)
    seed = args.seed

    config = {
        "N": N,
        "R": R,
        "dt": dt,
        "t_max": t_max,
        "show_plots": show_plots,
        "output_dir": output_dir,
        "seed": seed,
    }

    scenarios = []
    if args.scenario in ("a", "all"):
        scenarios.append(("A", run_scenario_a))
    if args.scenario in ("b", "all"):
        scenarios.append(("B", run_scenario_b))
    if args.scenario in ("c", "all"):
        scenarios.append(("C", run_scenario_c))

    for name, run_fn in scenarios:
        print(f"Running Scenario {name}...")
        run_fn(**config)

    if len(scenarios) > 0:
        out_path = output_dir.resolve()
        files = [
            "figure_2a_limit_cycle.png" if args.scenario in ("a", "all") else None,
            "figure_2b_bistable.png" if args.scenario in ("b", "all") else None,
            "figure_2c_winner_take_all.png" if args.scenario in ("c", "all") else None,
        ]
        files = [f for f in files if f]
        print(f"Done. Figures saved to: {out_path}")
        for f in files:
            print(f"  - {out_path / f}")


if __name__ == "__main__":
    main()

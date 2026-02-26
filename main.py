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
    show_flow: bool,
    show_plots: bool,
    output_dir: Path,
    seed: int,
    flow_n_ics: int,
    flow_t_max: float,
    flow_x0_scale: float,
) -> None:
    """Scenario A: Single-task limit cycle (Fig 2A)."""
    tc = TaskComponent(D=2.2, A=[[0.8, 0.4], [-0.4, 0.8]], N=N, R=R, seed=seed)
    network = MultiTaskNetwork([tc], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    # Flow field: many short trajectories from random ICs
    if show_flow:
        _, flow_z_by_task = sim.run_flow_ensemble(
            n_ics=flow_n_ics,
            t_max_flow=flow_t_max,
            x0_scale=flow_x0_scale,
            seed=seed + 100,
        )
        flow_z = flow_z_by_task[0]
    else:
        flow_z = None

    # Main trajectory: one long run (t_max = 100), random IC with larger scale
    rng = np.random.default_rng(123)
    x0 = rng.standard_normal(N) * 2.0
    t, _, z_traj = sim.run(x0)
    z = z_traj[0]

    z_flat = np.concatenate([z[:, 0], z[:, 1]])
    if flow_z is not None:
        for fz in flow_z:
            z_flat = np.concatenate([z_flat, fz[:, 0], fz[:, 1]])
    lim = max(np.abs(z_flat).max() * 1.15, 10)
    z_range = (-lim, lim)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    viz = Visualizer(network, sim)
    viz.plot_flow_and_trajectory(
        0, t, z, ax,
        flow_z_trajectories=flow_z,
        color="purple",
        show_flow=show_flow,
    )
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
    show_flow: bool,
    show_plots: bool,
    output_dir: Path,
    seed: int,
    flow_n_ics: int,
    flow_t_max: float,
    flow_x0_scale: float,
) -> None:
    """Scenario B: Single-task bistable (Fig 2B)."""
    tc = TaskComponent(D=2.0, A=[[0.5, 0.3], [0.3, 0.5]], N=N, R=R, seed=seed)
    network = MultiTaskNetwork([tc], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    if show_flow:
        _, flow_z_by_task = sim.run_flow_ensemble(
            n_ics=flow_n_ics,
            t_max_flow=flow_t_max,
            x0_scale=flow_x0_scale,
            seed=seed + 200,
        )
        flow_z = flow_z_by_task[0]
    else:
        flow_z = None

    # Two distinct random initial states (different seeds) so trajectories start in different places
    rng_a = np.random.default_rng(1001)
    rng_b = np.random.default_rng(1002)
    x0a = rng_a.standard_normal(N) * 2.0
    x0b = rng_b.standard_normal(N) * 2.0

    t, _, z_traj_a = sim.run(x0a)
    _, _, z_traj_b = sim.run(x0b)

    za, zb = z_traj_a[0], z_traj_b[0]
    z_flat = np.concatenate([za[:, 0], za[:, 1], zb[:, 0], zb[:, 1]])
    if flow_z is not None:
        for fz in flow_z:
            z_flat = np.concatenate([z_flat, fz[:, 0], fz[:, 1]])
    lim = max(np.abs(z_flat).max() * 1.15, 10)
    z_range = (-lim, lim)
    fixed_pts = [tuple(za[-1]), tuple(zb[-1])]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    viz = Visualizer(network, sim)
    viz.plot_flow_and_trajectory(
        0, t, za, ax,
        flow_z_trajectories=flow_z,
        color="red",
        show_flow=show_flow,
        fixed_points=fixed_pts,
    )
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
    show_flow: bool,
    show_plots: bool,
    output_dir: Path,
    seed: int,
    flow_n_ics: int,
    flow_t_max: float,
    flow_x0_scale: float,
) -> None:
    """Scenario C: Two-task winner-take-all (Fig 2C)."""
    tc1 = TaskComponent(D=2.2, A=[[0.8, 0.4], [-0.4, 0.8]], N=N, R=R, seed=seed)
    tc2 = TaskComponent(D=2.0, A=[[0.5, 0.3], [0.3, 0.5]], N=N, R=R, seed=seed + 1)
    network = MultiTaskNetwork([tc1, tc2], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    if show_flow:
        _, flow_z_by_task = sim.run_flow_ensemble(
            n_ics=flow_n_ics,
            t_max_flow=flow_t_max,
            x0_scale=flow_x0_scale,
            seed=seed + 300,
        )
        flow_z1, flow_z2 = flow_z_by_task[0], flow_z_by_task[1]
    else:
        flow_z1, flow_z2 = None, None

    # Main trajectory: random IC with larger scale (so not starting at origin in latent space)
    rng = np.random.default_rng(456)
    x0 = rng.standard_normal(N) * 2.0
    t, _, z_traj = sim.run(x0)

    z1, z2 = z_traj[0], z_traj[1]
    lim1 = max(np.abs(np.concatenate([z1[:, 0], z1[:, 1]])).max() * 1.15, 10)
    lim2 = max(np.abs(np.concatenate([z2[:, 0], z2[:, 1]])).max() * 1.15, 10)
    if flow_z1 is not None:
        for fz in flow_z1:
            lim1 = max(lim1, np.abs(fz).max() * 1.15)
        for fz in flow_z2:
            lim2 = max(lim2, np.abs(fz).max() * 1.15)
    range1, range2 = (-lim1, lim1), (-lim2, lim2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    viz = Visualizer(network, sim)

    viz.plot_flow_and_trajectory(
        0, t, z1, axes[0],
        flow_z_trajectories=flow_z1,
        color="purple",
        show_flow=show_flow,
    )
    axes[0].set_xlabel(r"$z_1^{(1)}(t)$")
    axes[0].set_ylabel(r"$z_2^{(1)}(t)$")
    axes[0].set_title("C (left): Task 1 dominant  (P = 2, R = 2)")
    axes[0].set_xlim(range1)
    axes[0].set_ylim(range1)
    axes[0].set_aspect("equal")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)

    viz.plot_flow_and_trajectory(
        1, t, z2, axes[1],
        flow_z_trajectories=flow_z2,
        color="red",
        show_flow=show_flow,
        fixed_points=[(0.0, 0.0)],
    )
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
        "--no-flow",
        action="store_true",
        help="Skip flow field in plots (faster)",
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
    parser.add_argument(
        "--flow-n-ics",
        type=int,
        default=25,
        help="Number of short trajectories for flow field (default: 25)",
    )
    parser.add_argument(
        "--flow-t-max",
        type=float,
        default=20.0,
        help="Duration of each flow-field trajectory (default: 20)",
    )
    parser.add_argument(
        "--flow-x0-scale",
        type=float,
        default=15.0,
        help="Scale (std) of random initial conditions for flow field (default: 15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    show_flow = not args.no_flow
    show_plots = not args.no_display
    N = args.neurons
    R = 2
    dt = float(args.dt)
    t_max = float(args.t_max)
    seed = args.seed
    # Numerical integration: dt = 0.05, t_max = 100 (paper)
    if args.dt == 0.05 and args.t_max == 100.0:
        pass  # use as-is
    flow_n_ics = args.flow_n_ics
    flow_t_max = args.flow_t_max
    flow_x0_scale = args.flow_x0_scale

    config = {
        "N": N,
        "R": R,
        "dt": dt,
        "t_max": t_max,
        "show_flow": show_flow,
        "show_plots": show_plots,
        "output_dir": output_dir,
        "seed": seed,
        "flow_n_ics": flow_n_ics,
        "flow_t_max": flow_t_max,
        "flow_x0_scale": flow_x0_scale,
    }

    scenarios = []
    # if args.scenario in ("a", "all"):
    #     scenarios.append(("A", run_scenario_a))
    if args.scenario in ("b", "all"):
        scenarios.append(("B", run_scenario_b))
    # if args.scenario in ("c", "all"):
    #     scenarios.append(("C", run_scenario_c))

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

"""
Visualizer: flow fields and trajectory plotting.
Paper-style: dense short trajectories as thin black lines with black X at endpoints;
main long trajectory on top in bold color.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

from multi_task_network import MultiTaskNetwork
from simulation import Simulation
from core import firing_rate


class Visualizer:
    """
    Plots flow fields (short trajectories) and long trajectories.
    """

    def __init__(self, network: MultiTaskNetwork, simulation: Simulation):
        self.network = network
        self.simulation = simulation

    def plot_flow_trajectories(
        self,
        flow_z_list: List[np.ndarray],
        ax: plt.Axes,
        color: str = "black",
        alpha: float = 0.5,
        linewidth: float = 0.6,
        mark_ends: bool = True,
    ):
        """
        Plot many short flow-field trajectories as thin semi-transparent lines.
        Mark each trajectory's final point with a black 'X'.
        """
        for z_traj in flow_z_list:
            if len(z_traj) < 2:
                continue
            z1, z2 = z_traj[:, 0], z_traj[:, 1]
            ax.plot(z1, z2, color=color, alpha=alpha, linewidth=linewidth, zorder=1)
            if mark_ends:
                ax.scatter(
                    z1[-1], z2[-1],
                    color="black",
                    s=50,
                    zorder=2,
                    marker="x",
                    linewidths=1.2,
                )

    def plot_latent_trajectory(
        self,
        t: np.ndarray,
        z_traj: np.ndarray,
        ax: plt.Axes,
        task_label: str = "",
        color: str = "C0",
        alpha: float = 0.9,
        linewidth: float = 2.5,
        mark_start: bool = True,
        fixed_points: Optional[list] = None,
    ):
        """Plot z1 vs z2 trajectory on given axes. Optionally mark fixed points with black X."""
        z1, z2 = z_traj[:, 0], z_traj[:, 1]
        ax.plot(z1, z2, color=color, alpha=alpha, linewidth=linewidth, zorder=5)
        if mark_start:
            ax.scatter(z1[0], z2[0], color=color, s=40, zorder=6, marker="o")
        if fixed_points:
            for (x, y) in fixed_points:
                ax.scatter(x, y, color="black", s=80, zorder=6, marker="x", linewidths=2)

    def plot_flow_and_trajectory(
        self,
        task_idx: int,
        t: np.ndarray,
        z_traj: np.ndarray,
        ax: plt.Axes,
        flow_z_trajectories: Optional[List[np.ndarray]] = None,
        color: str = "C0",
        show_flow: bool = True,
        linewidth: float = 2.5,
        fixed_points: Optional[list] = None,
    ):
        """
        Plot flow field then main trajectory (paper-style).
        If flow_z_trajectories is provided, plot them as thin black lines with X at ends;
        then plot the long trajectory on top in bold color.
        """
        if show_flow and flow_z_trajectories is not None:
            self.plot_flow_trajectories(
                flow_z_trajectories,
                ax,
                color="black",
                alpha=0.5,
                linewidth=0.6,
                mark_ends=True,
            )
        self.plot_latent_trajectory(
            t, z_traj, ax, color=color, linewidth=linewidth, fixed_points=fixed_points
        )

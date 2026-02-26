"""
Visualizer: trajectory plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from multi_task_network import MultiTaskNetwork
from simulation import Simulation


class Visualizer:
    """Plots latent-space trajectories."""

    def __init__(self, network: MultiTaskNetwork, simulation: Simulation):
        self.network = network
        self.simulation = simulation

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

    def plot_trajectory(
        self,
        task_idx: int,
        t: np.ndarray,
        z_traj: np.ndarray,
        ax: plt.Axes,
        color: str = "C0",
        linewidth: float = 2.5,
    ):
        """Plot z1 vs z2 trajectory on given axes."""
        self.plot_latent_trajectory(t, z_traj, ax, color=color, linewidth=linewidth)

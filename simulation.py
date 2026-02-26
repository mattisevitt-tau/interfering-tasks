"""
Simulation: runs network over time and records trajectories.
"""

import numpy as np
from typing import Optional, List, Tuple

from multi_task_network import MultiTaskNetwork
from core import firing_rate


class Simulation:
    """
    Runs the network over a time window and records latent variable trajectories.
    """

    def __init__(
        self,
        network: MultiTaskNetwork,
        dt: float = 0.05,
        t_max: float = 100.0,
    ):
        self.network = network
        self.dt = dt
        self.t_max = t_max

    def run_flow_ensemble(
        self,
        n_ics: int = 25,
        t_max_flow: float = 20.0,
        x0_scale: float = 15.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """
        Run many short trajectories from random initial conditions for flow-field visualization.
        Returns (t_flow, flow_z_by_task) where flow_z_by_task[task_idx] is a list of
        z-trajectory arrays, each shape (n_steps, R).
        """
        rng = np.random.default_rng(seed)
        flow_z_by_task = [[] for _ in self.network.tasks]

        for _ in range(n_ics):
            x0 = x0_scale * rng.standard_normal(self.network.N)
            t_flow, _, z_traj_list = self.run(x0, t_max=t_max_flow)
            for task_idx, z_traj in enumerate(z_traj_list):
                flow_z_by_task[task_idx].append(z_traj)

        return t_flow, flow_z_by_task

    def run(
        self,
        x0: np.ndarray,
        t_max: Optional[float] = None,
        record_interval: int = 1,
    ) -> tuple:
        """
        Run simulation from initial condition x0.

        Returns
        -------
        t : np.ndarray
        x_traj : np.ndarray, shape (n_steps, N)
        z_traj : list of np.ndarray, each shape (n_steps, R)
        """
        t_max = t_max or self.t_max
        n_steps = int(t_max / self.dt)
        t = np.arange(n_steps + 1) * self.dt

        x = x0.copy()
        x_traj = [x.copy()]
        phi = firing_rate(x)
        z_list = self.network.compute_latent(phi)
        z_traj = [[z.copy()] for z in z_list]

        for step in range(n_steps):
            x = self.network.step_rk4(x, self.dt)
            if step % record_interval == 0 or step == n_steps - 1:
                x_traj.append(x.copy())
                phi = firing_rate(x)
                z_list = self.network.compute_latent(phi)
                for i, z in enumerate(z_list):
                    z_traj[i].append(z.copy())

        x_traj = np.array(x_traj)
        z_traj = [np.array(z) for z in z_traj]
        n_rec = len(x_traj)
        t = t[:n_rec]

        return t, x_traj, z_traj

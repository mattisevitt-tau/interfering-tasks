"""
MultiTaskNetwork: builds weight matrix and provides RK4 integration.
"""

import numpy as np
from typing import List

from task_component import TaskComponent
from core import firing_rate


class MultiTaskNetwork:
    """
    Builds the full weight matrix J from TaskComponents and provides
    RK4 integration for the network dynamics.
    """

    def __init__(self, task_components: List[TaskComponent], N: int):
        """
        Parameters
        ----------
        task_components : list of TaskComponent
        N : int
            Number of neurons.
        """
        self.tasks = task_components
        self.N = N

        for tc in self.tasks:
            tc.generate_vectors()

        self.J = np.zeros((N, N))
        for tc in self.tasks:
            self.J += tc.weight_contribution()

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side of dynamics: ẋ_i = -x_i + Σ_j J_ij φ_j"""
        phi = firing_rate(x)
        return -x + self.J @ phi

    def step_rk4(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Single RK4 step."""
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * dt * k1)
        k3 = self.rhs(x + 0.5 * dt * k2)
        k4 = self.rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def compute_latent(self, phi: np.ndarray) -> List[np.ndarray]:
        """
        Latent variables: z_r^(μ) = D^(μ) * Σ_i n_i^(μ,r) φ_i (paper formula).
        """
        z_list = []
        for tc in self.tasks:
            z = tc.D * (tc.n.T @ phi)  # D * sum_i(n_i * phi_i) per component r
            z_list.append(z)
        return z_list

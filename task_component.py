"""
TaskComponent: stores task parameters and generates loading vectors.
"""

import numpy as np
from typing import Optional


class TaskComponent:
    """
    Stores parameters for a single task (D^(μ), A^(μ)) and generates
    the task-specific m and n loading vectors via multivariate normal sampling.
    """

    def __init__(self, D: float, A: np.ndarray, N: int, R: int = 2, seed: Optional[int] = None):
        """
        Parameters
        ----------
        D : float
            Connection strength for this task.
        A : np.ndarray
            R x R overlap matrix for this task.
        N : int
            Number of neurons.
        R : int
            Within-task dimension (number of left/right loading vectors).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.D = D
        self.A = np.asarray(A)
        self.N = N
        self.R = R
        self.seed = seed

        # Build 2R x 2R covariance matrix: Σ = (1/N) * [[I, A], [A^T, I]]
        I_R = np.eye(R)
        Sigma = (1.0 / N) * np.block([
            [I_R, self.A],
            [self.A.T, I_R]
        ])

        self.Sigma = Sigma
        self._m = None  # shape (N, R)
        self._n = None  # shape (N, R)

    def generate_vectors(self) -> tuple:
        """
        Generate m and n loading vectors. For each neuron i, draw
        v_i = [m_i^(1), ..., m_i^(R), n_i^(1), ..., n_i^(R)] from N(0, Σ).

        Returns
        -------
        m : np.ndarray, shape (N, R)
        n : np.ndarray, shape (N, R)
        """
        rng = np.random.default_rng(self.seed)
        v = rng.multivariate_normal(
            mean=np.zeros(2 * self.R),
            cov=self.Sigma,
            size=self.N
        )
        self._m = v[:, :self.R]
        self._n = v[:, self.R:]
        return self._m, self._n

    @property
    def m(self) -> np.ndarray:
        """Left loading vectors, shape (N, R)."""
        if self._m is None:
            self.generate_vectors()
        return self._m

    @property
    def n(self) -> np.ndarray:
        """Right loading vectors, shape (N, R)."""
        if self._n is None:
            self.generate_vectors()
        return self._n

    def weight_contribution(self) -> np.ndarray:
        """J^(μ)_ij = D * Σ_r m_i^(r) n_j^(r). Returns shape (N, N)."""
        return self.D * (self.m @ self.n.T)

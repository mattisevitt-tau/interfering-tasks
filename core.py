"""
Core utilities for CTRNN simulation.
Based on "A theory of multi-task computation and task selection" by Marschall et al.
"""

import numpy as np
from scipy.special import erf


def firing_rate(x: np.ndarray) -> np.ndarray:
    """Neuron activation: φ(x) = erf(√π · x / 2), exactly as in the paper."""
    return erf(np.sqrt(np.pi) * x / 2)

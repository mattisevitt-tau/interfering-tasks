"""
Experiments module - Blueprint-aligned implementation.

Based on Marschall et al. (2025): Similarity of dynamical motifs in low-rank RNNs.
- Network: τẋ_i = -x_i + Σ_j J_ij φ(x_j), τ=1
- φ(x) = erf(√π/2 · x)
- J_ij = Σ_μ D^(μ) Σ_r m_i^(μ,r) n_j^(μ,r), R=2, P=2
- Overlap: ⟨n^(μ,r)·m^(μ,r')⟩ = A_rr'^(μ)/N
"""

import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt

from task_component import TaskComponent
from multi_task_network import MultiTaskNetwork
from simulation import Simulation
from core import firing_rate


# --- Blueprint constants ---
TAU = 1.0
R_RANK = 2
P_TASKS = 2
D_MEAN = 2.5
N_VALUES = [200, 500, 1000]
BURNIN_FRACTION = 0.2  # discard first 20% of simulation time


# --- Feature family A matrices (exact blueprint formulas) ---

def A_frequency(omega: float) -> np.ndarray:
    """Family 1: Frequency. A(ω) = 0.8 [[1, ω], [-ω, 1]]. Note: ω = tan θ."""
    return 0.8 * np.array([[1, omega], [-omega, 1]], dtype=float)


def A_amplitude(gamma: float) -> np.ndarray:
    """Family 2: Amplitude/Radius. A(γ) = γ [[1, 0.5], [-0.5, 1]]."""
    return gamma * np.array([[1, 0.5], [-0.5, 1]], dtype=float)


def A_shape(epsilon: float) -> np.ndarray:
    """Family 3: Shape/Eccentricity. A(ε) = 0.8 [[ε, 0.5], [-0.5, 1/ε]]."""
    return 0.8 * np.array([[epsilon, 0.5], [-0.5, 1.0 / epsilon]], dtype=float)


# --- Sweep parameter grids (blueprint) ---

FAMILY_1_DELTA_OMEGA = [0.0, 0.2, 0.4, 0.6]  # Δω
FAMILY_2_DELTA_GAMMA = [0.0, 0.2, 0.4, 0.6]  # Δγ
FAMILY_3_DELTA_EPSILON = [0.0, 0.2, 0.5, 1.0]  # Δε

DELTA_D_VALUES = np.linspace(-2.0, 2.0, 9).tolist()  # 9 steps


def delta_D_to_D1_D2(delta_D: float) -> Tuple[float, float]:
    """D1 = D_mean + ΔD/2, D2 = D_mean - ΔD/2."""
    D1 = D_MEAN + delta_D / 2
    D2 = D_MEAN - delta_D / 2
    return D1, D2


# --- Post-processing metrics (blueprint) ---

def latent_norm_task_survival(z_traj: np.ndarray) -> float:
    """S_μ = ⟨√(Σ_r z_r^(μ)(t)^2)⟩_t. Task survival: O(√N) vs O(1)."""
    norms = np.sqrt(np.sum(z_traj ** 2, axis=1))
    return float(np.mean(norms))


def normalized_power(z_traj: np.ndarray, D: float) -> float:
    """P_μ = mean_t(Σ_r (z_r^(μ)(t)/D^(μ))^2). Task dominance."""
    scaled = z_traj / D
    return float(np.mean(np.sum(scaled ** 2, axis=1)))


def dominance_index(P1: float, P2: float) -> float:
    """S = (P_1 - P_2)/(P_1 + P_2). Maps selection to [-1, 1]. """
    denom = P1 + P2
    if denom < 1e-12:
        return 0.0
    return (P1 - P2) / denom


def compute_metrics_from_z(z1_traj: np.ndarray, z2_traj: np.ndarray, D1: float, D2: float) -> dict:
    """
    Compute blueprint metrics from latent trajectories.
    Expects z trajectories as (n_steps, R) arrays.
    Discards first 20% (caller should pass post-burnin data if needed).
    """
    P1 = normalized_power(z1_traj, D1)
    P2 = normalized_power(z2_traj, D2)
    S1 = latent_norm_task_survival(z1_traj)
    S2 = latent_norm_task_survival(z2_traj)
    S_idx = dominance_index(P1, P2)
    return {"P1": P1, "P2": P2, "S1": S1, "S2": S2, "S": S_idx}


# --- Single run ---

def run_single_experiment(
    N: int,
    A1: np.ndarray,
    A2: np.ndarray,
    D1: float,
    D2: float,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    x0_seed: int = 456,
    x0_scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run one two-task simulation and return t, z1, z2, metrics.
    Metrics are computed on steady-state (after discarding first 20%).
    """
    tc1 = TaskComponent(D=D1, A=A1.tolist(), N=N, R=R_RANK, seed=seed)
    tc2 = TaskComponent(D=D2, A=A2.tolist(), N=N, R=R_RANK, seed=seed + 1)
    network = MultiTaskNetwork([tc1, tc2], N)
    sim = Simulation(network, dt=dt, t_max=t_max)

    rng = np.random.default_rng(x0_seed)
    x0 = rng.standard_normal(N) * x0_scale
    t, _, z_traj = sim.run(x0)

    z1 = z_traj[0]  # (n_steps, R)
    z2 = z_traj[1]

    # Discard first 20% (blueprint)
    n_burnin = int(len(t) * BURNIN_FRACTION)
    z1_ss = z1[n_burnin:]
    z2_ss = z2[n_burnin:]

    metrics = compute_metrics_from_z(z1_ss, z2_ss, D1, D2)
    return t, z1, z2, metrics


# --- Experiment sweeps ---

@dataclass
class ExperimentConfig:
    """Configuration for one experiment family."""

    family_name: str
    A1: np.ndarray
    delta_values: List[float]
    A_func: callable  # A_func(base + delta) -> A matrix
    base_value: float


def get_experiment_configs() -> List[ExperimentConfig]:
    """Return the three blueprint experiment families."""
    return [
        ExperimentConfig(
            family_name="Frequency (ω)",
            A1=A_frequency(0.5),
            delta_values=FAMILY_1_DELTA_OMEGA,
            A_func=A_frequency,
            base_value=0.5,
        ),
        ExperimentConfig(
            family_name="Amplitude (γ)",
            A1=A_amplitude(0.8),
            delta_values=FAMILY_2_DELTA_GAMMA,
            A_func=A_amplitude,
            base_value=0.8,
        ),
        ExperimentConfig(
            family_name="Shape (ε)",
            A1=A_shape(1.0),
            delta_values=FAMILY_3_DELTA_EPSILON,
            A_func=A_shape,
            base_value=1.0,
        ),
    ]


def run_experiment_sweep(
    config: ExperimentConfig,
    N: int,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full sweep: (ΔFeature × ΔD) -> S.
    Returns: delta_feature_grid, delta_D_grid, S_matrix
    """
    n_delta = len(config.delta_values)
    n_dD = len(DELTA_D_VALUES)
    S_matrix = np.zeros((n_delta, n_dD))
    # Optional: store raw metrics for debugging
    # metrics_grid = [[None] * n_dD for _ in range(n_delta)]

    for i, delta_f in enumerate(config.delta_values):
        A2 = config.A_func(config.base_value + delta_f)
        for j, delta_D in enumerate(DELTA_D_VALUES):
            D1, D2 = delta_D_to_D1_D2(delta_D)
            _, _, _, metrics = run_single_experiment(
                N=N, A1=config.A1, A2=A2, D1=D1, D2=D2,
                dt=dt, t_max=t_max, seed=seed,
                x0_seed=seed + i * 100 + j,
            )
            S_matrix[i, j] = metrics["S"]

    return np.array(config.delta_values), np.array(DELTA_D_VALUES), S_matrix


def run_scaling_experiment(
    config: ExperimentConfig,
    N_values: Optional[List[int]] = None,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selection sharpness (slope of S at ΔD=0) vs N.
    Uses finite difference around ΔD=0.
    """
    N_values = N_values or N_VALUES
    slopes = []
    for N in N_values:
        _, delta_D_arr, S_matrix = run_experiment_sweep(config, N, dt, t_max, seed)
        # S_matrix: rows = delta_feature, cols = delta_D
        # For scaling we typically use delta_feature=0 (identical tasks) and slope at ΔD=0
        # Row 0 = delta_feature=0 (identical A1, A2)
        S_at_zero = S_matrix[0, :]  # S as function of ΔD when tasks are identical
        # Slope at ΔD=0: center index is 4 (of 9: -2,-1.5,-1,-0.5,0,0.5,1,1.5,2)
        idx_center = len(delta_D_arr) // 2
        if idx_center > 0 and idx_center < len(delta_D_arr) - 1:
            slope = (S_at_zero[idx_center + 1] - S_at_zero[idx_center - 1]) / (
                delta_D_arr[idx_center + 1] - delta_D_arr[idx_center - 1]
            )
        else:
            slope = 0.0
        slopes.append(slope)
    return np.array(N_values, dtype=float), np.array(slopes)


# --- Visualizations (blueprint) ---

def plot_heatmap(
    delta_feature: np.ndarray,
    delta_D: np.ndarray,
    S_matrix: np.ndarray,
    family_name: str,
    delta_label: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Heatmap: X = ΔFeature, Y = ΔD, Color = S (dominance index). Blueprint axis order."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    # S_matrix[i,j] = S at (delta_feature[i], delta_D[j]). Rows=ΔFeature, Cols=ΔD.
    # Blueprint: X = ΔFeature, Y = ΔD. imshow: cols→x, rows→y. So transpose.
    S_plot = S_matrix.T  # Now rows=ΔD, cols=ΔFeature
    im = ax.imshow(
        S_plot,
        aspect="auto",
        origin="lower",
        extent=[delta_feature[0], delta_feature[-1], delta_D[0], delta_D[-1]],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax.set_xlabel(delta_label)   # X = ΔFeature
    ax.set_ylabel(r"$\Delta D$")  # Y = ΔD
    ax.set_title(f"Heatmap: {family_name} — Dominance index S")
    plt.colorbar(im, ax=ax, label="S (dominance)")
    return fig


def plot_phase_portraits(
    config: ExperimentConfig,
    N: int,
    delta_D_show: List[float],
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
) -> plt.Figure:
    """2D phase portraits of z^(1) and z^(2) for selected ΔD values."""
    n_show = len(delta_D_show)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    A2 = config.A_func(config.base_value)  # use base (delta=0) for simplicity

    for idx, delta_D in enumerate(delta_D_show):
        D1, D2 = delta_D_to_D1_D2(delta_D)
        _, z1, z2, _ = run_single_experiment(
            N=N, A1=config.A1, A2=A2, D1=D1, D2=D2,
            dt=dt, t_max=t_max, seed=seed, x0_seed=seed + idx,
        )
        ax = axes[idx]
        ax.plot(z1[:, 0], z1[:, 1], color="purple", alpha=0.8, label=r"$z^{(1)}$")
        ax.plot(z2[:, 0], z2[:, 1], color="red", alpha=0.8, label=r"$z^{(2)}$")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_title(f"ΔD = {delta_D:.1f}")
        ax.legend()
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
    plt.suptitle(f"Phase portraits: {config.family_name}")
    plt.tight_layout()
    return fig


def plot_scaling(
    N_arr: np.ndarray,
    slopes: np.ndarray,
    family_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scaling plot: Selection sharpness (slope of S at ΔD=0) vs N."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.plot(N_arr, slopes, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Selection sharpness (dS/dΔD at ΔD=0)")
    ax.set_title(f"Scaling: {family_name}")
    return fig


def _family_to_slug(name: str) -> str:
    """Convert family name to filename-safe slug."""
    s = name.lower().replace(" (ω)", "_omega").replace(" (γ)", "_gamma").replace(" (ε)", "_epsilon")
    return s.replace(" ", "_")


def _get_delta_label(family_name: str) -> str:
    """Get LaTeX delta label for family."""
    labels = {"Frequency (ω)": r"$\Delta\omega$", "Amplitude (γ)": r"$\Delta\gamma$", "Shape (ε)": r"$\Delta\varepsilon$"}
    return labels.get(family_name, r"$\Delta$")


def run_all_blueprint_experiments(
    output_dir: Optional[Path] = None,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
) -> Path:
    """
    Run all blueprint experiments and save results.

    Creates:
    - CSV files: sweep data (family, N, delta_feature, delta_D, S) for each family×N
    - PNG graphs: heatmaps, phase portraits, scaling plots
    - summary.csv: scaling slopes (family, N, selection_sharpness)

    Returns the output directory path.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiment_results") / f"run_{timestamp}"
    output_dir = Path(output_dir)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    configs = get_experiment_configs()
    delta_D_show = [-1.0, 0.0, 1.0]
    scaling_rows = []

    for config in configs:
        slug = _family_to_slug(config.family_name)
        delta_label = _get_delta_label(config.family_name)

        for N in N_VALUES:
            delta_f, delta_D, S_mat = run_experiment_sweep(config, N, dt=dt, t_max=t_max, seed=seed)
            csv_path = output_dir / f"sweep_{slug}_N{N}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["family", "N", "delta_feature", "delta_D", "S"])
                for i, df in enumerate(delta_f):
                    for j, dD in enumerate(delta_D):
                        w.writerow([config.family_name, N, df, dD, S_mat[i, j]])

            fig = plot_heatmap(delta_f, delta_D, S_mat, config.family_name, delta_label)
            fig.savefig(graphs_dir / f"heatmap_{slug}_N{N}.png", dpi=150)
            plt.close(fig)

        N_arr, slopes = run_scaling_experiment(config, N_values=N_VALUES, dt=dt, t_max=t_max, seed=seed)
        for n, sl in zip(N_arr, slopes):
            scaling_rows.append({"family": config.family_name, "N": int(n), "selection_sharpness": sl})
        fig = plot_scaling(N_arr, slopes, config.family_name)
        fig.savefig(graphs_dir / f"scaling_{slug}.png", dpi=150)
        plt.close(fig)

        fig = plot_phase_portraits(config, N=500, delta_D_show=delta_D_show, dt=dt, t_max=t_max, seed=seed)
        fig.savefig(graphs_dir / f"phase_portraits_{slug}_N500.png", dpi=150)
        plt.close(fig)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "N", "selection_sharpness"])
        w.writeheader()
        w.writerows(scaling_rows)

    return output_dir

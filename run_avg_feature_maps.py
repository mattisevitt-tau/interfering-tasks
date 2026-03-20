#!/usr/bin/env python3
"""
Averaged feature-vs-strength experiments for two-task competition.

Generates (averaged across seeds):
1) Heatmaps: S(Δfeature, ΔD) for
   - frequency difference (Δω)
   - coupling difference (Δb)
2) Curves of S at ΔD=0:
   - S vs Δω
   - S vs Δb
3) Slopes at zero feature gap for different ΔD:
   - dS/d(Δω) at Δω=0 vs ΔD
   - dS/d(Δb) at Δb=0 vs ΔD

Usage:
  /opt/miniconda3/envs/interfering-tasks/bin/python run_avg_feature_maps.py \
      --n-seeds 5 -N 500 -o comparison_results
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import run_single_experiment, D_MEAN


def A_limit_cycle(omega: float, mu: float = 0.6) -> np.ndarray:
    return mu * np.array([[1.0, omega], [-omega, 1.0]], dtype=float)


def A_fixed_point(b: float, diag: float = 0.5) -> np.ndarray:
    return np.array([[diag, b], [b, diag]], dtype=float)


def finite_diff_at_zero(x: np.ndarray, y: np.ndarray) -> float:
    idx0 = np.argmin(np.abs(x))
    if idx0 == 0 or idx0 == len(x) - 1:
        return 0.0
    return float((y[idx0 + 1] - y[idx0 - 1]) / (x[idx0 + 1] - x[idx0 - 1]))


def run_mean_std_S(
    A1: np.ndarray,
    A2: np.ndarray,
    D1: float,
    D2: float,
    N: int,
    dt: float,
    t_max: float,
    base_seed: int,
    n_seeds: int,
    offset: int,
) -> Tuple[float, float]:
    vals = []
    for k in range(n_seeds):
        seed_k = base_seed + 1000 * k + offset
        _, _, _, metrics = run_single_experiment(
            N=N,
            A1=A1,
            A2=A2,
            D1=D1,
            D2=D2,
            dt=dt,
            t_max=t_max,
            seed=seed_k,
            x0_seed=seed_k + 500,
        )
        vals.append(metrics["S"])
    arr = np.array(vals)
    return float(np.mean(arr)), float(np.std(arr))


def sweep_feature_vs_deltaD(
    feature_kind: str,
    feature_grid: np.ndarray,
    delta_D_grid: np.ndarray,
    N: int,
    dt: float,
    t_max: float,
    n_seeds: int,
    base_seed: int,
    D_mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    S_mean = np.zeros((len(feature_grid), len(delta_D_grid)))
    S_std = np.zeros_like(S_mean)

    if feature_kind == "omega":
        omega_base = 0.6
        A1 = A_limit_cycle(omega_base)
    elif feature_kind == "b":
        b_base = 0.0
        A1 = A_fixed_point(b_base)
    else:
        raise ValueError("feature_kind must be 'omega' or 'b'")

    for i, feat in enumerate(feature_grid):
        if feature_kind == "omega":
            A2 = A_limit_cycle(omega_base + feat)
        else:
            A2 = A_fixed_point(feat)

        for j, dD in enumerate(delta_D_grid):
            D1 = D_mean + dD / 2.0
            D2 = D_mean - dD / 2.0
            m, s = run_mean_std_S(
                A1, A2, D1, D2,
                N=N, dt=dt, t_max=t_max,
                base_seed=base_seed, n_seeds=n_seeds,
                offset=10000 * i + j,
            )
            S_mean[i, j] = m
            S_std[i, j] = s

    return S_mean, S_std


def save_matrix_csv(path: Path, feature_name: str, feature_grid: np.ndarray, delta_D_grid: np.ndarray, S_mean: np.ndarray, S_std: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([feature_name, "delta_D", "S_mean", "S_std"])
        for i, fv in enumerate(feature_grid):
            for j, dD in enumerate(delta_D_grid):
                w.writerow([float(fv), float(dD), float(S_mean[i, j]), float(S_std[i, j])])


def plot_heatmap(path: Path, x: np.ndarray, y: np.ndarray, Z: np.ndarray, xlabel: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        Z.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta D$")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="S")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_S_at_deltaD0(path: Path, omega_grid: np.ndarray, S_omega: np.ndarray, E_omega: np.ndarray,
                      b_grid: np.ndarray, S_b: np.ndarray, E_b: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.errorbar(omega_grid, S_omega, yerr=E_omega, fmt="o-", capsize=3, color="black")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta\omega$")
    ax.set_ylabel("S")
    ax.set_title(r"$S$ at $\Delta D=0$ vs $\Delta\omega$")
    ax.set_ylim(-1.05, 1.05)

    ax = axes[1]
    ax.errorbar(b_grid, S_b, yerr=E_b, fmt="o-", capsize=3, color="darkgreen")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta b$")
    ax.set_ylabel("S")
    ax.set_title(r"$S$ at $\Delta D=0$ vs $\Delta b$")
    ax.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_slopes_vs_deltaD(path: Path, delta_D_grid: np.ndarray, slope_omega: np.ndarray, slope_b: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(delta_D_grid, slope_omega, "o-", color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel(r"$dS/d(\Delta\omega)$ at $\Delta\omega=0$")
    ax.set_title("Frequency sensitivity vs ΔD")

    ax = axes[1]
    ax.plot(delta_D_grid, slope_b, "o-", color="teal")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel(r"$dS/d(\Delta b)$ at $\Delta b=0$")
    ax.set_title("Coupling sensitivity vs ΔD")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_all(N=500, n_seeds=5, dt=0.05, t_max=100.0, base_seed=42,
            D_mean=D_MEAN, output_dir="comparison_results"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omega_grid = np.linspace(-0.5, 0.5, 11)
    b_grid = np.linspace(-0.45, 0.45, 19)
    delta_D_grid = np.linspace(-2.0, 2.0, 17)

    print("Running ω heatmap sweep...")
    S_omega, E_omega = sweep_feature_vs_deltaD(
        "omega", omega_grid, delta_D_grid,
        N=N, dt=dt, t_max=t_max,
        n_seeds=n_seeds, base_seed=base_seed,
        D_mean=D_mean,
    )
    save_matrix_csv(out_dir / "avg_heatmap_omega.csv", "delta_omega", omega_grid, delta_D_grid, S_omega, E_omega)
    plot_heatmap(
        out_dir / "avg_heatmap_omega.png",
        omega_grid,
        delta_D_grid,
        S_omega,
        xlabel=r"$\Delta\omega$",
        title=f"Average S heatmap: frequency difference (N={N}, seeds={n_seeds})",
    )

    print("Running b heatmap sweep...")
    S_b, E_b = sweep_feature_vs_deltaD(
        "b", b_grid, delta_D_grid,
        N=N, dt=dt, t_max=t_max,
        n_seeds=n_seeds, base_seed=base_seed + 200000,
        D_mean=D_mean,
    )
    save_matrix_csv(out_dir / "avg_heatmap_b.csv", "delta_b", b_grid, delta_D_grid, S_b, E_b)
    plot_heatmap(
        out_dir / "avg_heatmap_b.png",
        b_grid,
        delta_D_grid,
        S_b,
        xlabel=r"$\Delta b$",
        title=f"Average S heatmap: coupling difference (N={N}, seeds={n_seeds})",
    )

    idx_dD0 = int(np.argmin(np.abs(delta_D_grid)))
    S_omega_dD0 = S_omega[:, idx_dD0]
    E_omega_dD0 = E_omega[:, idx_dD0]
    S_b_dD0 = S_b[:, idx_dD0]
    E_b_dD0 = E_b[:, idx_dD0]

    plot_S_at_deltaD0(
        out_dir / "avg_S_at_deltaD0.png",
        omega_grid, S_omega_dD0, E_omega_dD0,
        b_grid, S_b_dD0, E_b_dD0,
    )

    slope_omega = np.zeros(len(delta_D_grid))
    for j in range(len(delta_D_grid)):
        slope_omega[j] = finite_diff_at_zero(omega_grid, S_omega[:, j])

    slope_b = np.zeros(len(delta_D_grid))
    for j in range(len(delta_D_grid)):
        slope_b[j] = finite_diff_at_zero(b_grid, S_b[:, j])

    with open(out_dir / "avg_slopes_vs_deltaD.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_D", "dS_dDeltaOmega_at0", "dS_dDeltaB_at0"])
        for j, dD in enumerate(delta_D_grid):
            w.writerow([float(dD), float(slope_omega[j]), float(slope_b[j])])

    plot_slopes_vs_deltaD(
        out_dir / "avg_slopes_vs_deltaD.png",
        delta_D_grid,
        slope_omega,
        slope_b,
    )

    print("Done. Files written to:", out_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="Averaged heatmaps and sensitivity curves")
    parser.add_argument("-o", "--output-dir", default="comparison_results")
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=100.0)
    parser.add_argument("--D-mean", type=float, default=D_MEAN)
    args = parser.parse_args()

    run_all(N=args.N, n_seeds=args.n_seeds, dt=args.dt, t_max=args.t_max,
            base_seed=args.seed, D_mean=args.D_mean, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

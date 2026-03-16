#!/usr/bin/env python3
"""
Comparison experiments: Two-task competition under different dynamical regimes.

Experiments:
  1. Limit Cycle vs Limit Cycle — varying frequency difference Δω
  2a. Fixed Point vs Fixed Point — varying rotation angle θ
  2b. Fixed Point vs Fixed Point — varying off-diagonal coupling Δb
  3. Cross-comparison: LC-vs-LC, FP-vs-FP, Mixed (LC-vs-FP)

All outputs (CSVs + PNGs) go to comparison_results/.

Usage:
  conda activate interfering-tasks
  python run_comparison_experiments.py [--seed 42] [--t-max 100] [-o comparison_results]
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from experiments import (
    run_single_experiment,
    D_MEAN,
    R_RANK,
    BURNIN_FRACTION,
    delta_D_to_D1_D2,
)


# ──────────────────────────────────────────────────────────
#  A-matrix builders
# ──────────────────────────────────────────────────────────

def A_limit_cycle(omega: float, mu: float = 0.8) -> np.ndarray:
    """Limit-cycle A: mu * [[1, ω], [-ω, 1]].  Antisymmetric off-diag → oscillation."""
    return mu * np.array([[1, omega], [-omega, 1]], dtype=float)


def A_fixed_point(b: float = 0.3, diag: float = 0.5) -> np.ndarray:
    """Stable fixed-point A: [[diag, b], [b, diag]].  Symmetric → no rotation."""
    return np.array([[diag, b], [b, diag]], dtype=float)


def A_fixed_point_rotated(A_base: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate A: A' = R(θ) A R(θ)ᵀ.  Same eigenvalues, different eigenvectors."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ A_base @ R.T


# ──────────────────────────────────────────────────────────
#  Sweep helper
# ──────────────────────────────────────────────────────────

DELTA_D_GRID = np.linspace(-2.0, 2.0, 21)  # 21-point grid for smooth curves


def sweep_delta_D(
    A1: np.ndarray,
    A2: np.ndarray,
    N: int,
    delta_D_values: np.ndarray = DELTA_D_GRID,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep ΔD for a fixed pair of A matrices.  Returns (delta_D_arr, S_arr)."""
    S_arr = np.zeros(len(delta_D_values))
    for j, dD in enumerate(delta_D_values):
        D1, D2 = delta_D_to_D1_D2(dD)
        _, _, _, metrics = run_single_experiment(
            N=N, A1=A1, A2=A2, D1=D1, D2=D2,
            dt=dt, t_max=t_max, seed=seed, x0_seed=seed + j,
        )
        S_arr[j] = metrics["S"]
    return delta_D_values, S_arr


def selection_sharpness(delta_D: np.ndarray, S: np.ndarray) -> float:
    """dS/dΔD at ΔD≈0 via center finite difference."""
    idx = np.argmin(np.abs(delta_D))
    if 0 < idx < len(delta_D) - 1:
        return float(
            (S[idx + 1] - S[idx - 1]) / (delta_D[idx + 1] - delta_D[idx - 1])
        )
    return 0.0


# ──────────────────────────────────────────────────────────
#  Experiment 1: Limit-cycle frequency competition
# ──────────────────────────────────────────────────────────

def run_lc_freq_experiment(
    N: int = 500,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    out_dir: Path = Path("comparison_results"),
):
    """Sweep Δω (frequency difference) × ΔD for two limit-cycle tasks."""
    print("── Experiment 1: Limit-cycle frequency competition ──")
    omega_base = 0.5
    delta_omegas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    A1 = A_limit_cycle(omega_base)

    all_S = {}     # delta_omega -> S_arr
    csv_rows = []

    for d_omega in delta_omegas:
        omega2 = omega_base + d_omega
        A2 = A_limit_cycle(omega2)
        dD_arr, S_arr = sweep_delta_D(A1, A2, N, DELTA_D_GRID, dt, t_max, seed)
        all_S[d_omega] = S_arr
        for j, dD in enumerate(dD_arr):
            csv_rows.append([d_omega, dD, S_arr[j]])
        print(f"  Δω={d_omega:.1f}  done  (sharpness={selection_sharpness(dD_arr, S_arr):.4f})")

    # ── CSV ──
    csv_path = out_dir / "lc_freq_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_omega", "delta_D", "S"])
        w.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    # ── S vs ΔD curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    for i, d_omega in enumerate(delta_omegas):
        color = cmap(i / max(len(delta_omegas) - 1, 1))
        ax.plot(DELTA_D_GRID, all_S[d_omega], "o-", ms=4, color=color,
                label=rf"$\Delta\omega={d_omega:.1f}$")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("Dominance index  $S$")
    ax.set_title(f"Limit-cycle freq. competition  (N={N})")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "lc_freq_S_vs_deltaD.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lc_freq_S_vs_deltaD.png")

    # ── Heatmap ──
    S_mat = np.array([all_S[d] for d in delta_omegas])  # shape (n_omega, n_dD)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        S_mat.T, aspect="auto", origin="lower",
        extent=[delta_omegas[0], delta_omegas[-1], DELTA_D_GRID[0], DELTA_D_GRID[-1]],
        cmap="RdBu_r", vmin=-1, vmax=1,
    )
    ax.set_xlabel(r"$\Delta\omega$  (frequency difference)")
    ax.set_ylabel(r"$\Delta D$")
    ax.set_title(f"Heatmap: LC freq. competition  (N={N})")
    plt.colorbar(im, ax=ax, label="S")
    fig.tight_layout()
    fig.savefig(out_dir / "lc_freq_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lc_freq_heatmap.png")

    # ── Sharpness vs Δω ──
    sharpness_vals = [selection_sharpness(DELTA_D_GRID, all_S[d]) for d in delta_omegas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delta_omegas, sharpness_vals, "s-", ms=7, color="teal")
    ax.set_xlabel(r"$\Delta\omega$")
    ax.set_ylabel("Selection sharpness  (dS/dΔD)")
    ax.set_title(f"Sharpness vs frequency difference  (N={N})")
    fig.tight_layout()
    fig.savefig(out_dir / "lc_freq_sharpness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lc_freq_sharpness.png")

    # ── Phase portraits (2×3) ──
    delta_D_show = [-1.0, 0.0, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, d_omega in enumerate([0.0, 0.5]):
        omega2 = omega_base + d_omega
        A2 = A_limit_cycle(omega2)
        for col, dD in enumerate(delta_D_show):
            ax = axes[row, col]
            D1, D2 = delta_D_to_D1_D2(dD)
            _, z1, z2, metrics = run_single_experiment(
                N=N, A1=A1, A2=A2, D1=D1, D2=D2,
                dt=dt, t_max=t_max, seed=seed, x0_seed=seed + row * 10 + col,
            )
            ax.plot(z1[:, 0], z1[:, 1], color="purple", alpha=0.8, lw=1.2, label=r"Task 1 ($z^{(1)}$)")
            ax.plot(z2[:, 0], z2[:, 1], color="red", alpha=0.8, lw=1.2, label=r"Task 2 ($z^{(2)}$)")
            ax.set_aspect("equal")
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_title(rf"$\Delta\omega={d_omega:.1f}$,  $\Delta D={dD:.0f}$   S={metrics['S']:.2f}",
                         fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$z_1$")
            if col == 0:
                ax.set_ylabel(r"$z_2$")
            if row == 0 and col == 2:
                ax.legend(fontsize=7)
    fig.suptitle(f"Phase portraits — LC freq. competition  (N={N})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "lc_freq_phase_portraits.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lc_freq_phase_portraits.png")

    return all_S, sharpness_vals


# ──────────────────────────────────────────────────────────
#  Experiment 2a: Fixed-point competition — rotation angle θ
# ──────────────────────────────────────────────────────────

def run_fp_rotation_experiment(
    N: int = 500,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    out_dir: Path = Path("comparison_results"),
):
    """Sweep rotation angle θ × ΔD for two stable-fixed-point tasks."""
    print("── Experiment 2a: Fixed-point rotation competition ──")
    A_base = A_fixed_point(b=0.3, diag=0.5)
    thetas = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]

    all_S = {}
    csv_rows = []

    for theta in thetas:
        A2 = A_fixed_point_rotated(A_base, theta)
        dD_arr, S_arr = sweep_delta_D(A_base, A2, N, DELTA_D_GRID, dt, t_max, seed)
        all_S[theta] = S_arr
        for j, dD in enumerate(dD_arr):
            csv_rows.append([theta, dD, S_arr[j]])
        print(f"  θ={theta:5.1f}°  done  (sharpness={selection_sharpness(dD_arr, S_arr):.4f})")

    csv_path = out_dir / "fp_rotation_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_deg", "delta_D", "S"])
        w.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    # ── S vs ΔD ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.plasma
    for i, theta in enumerate(thetas):
        color = cmap(i / max(len(thetas) - 1, 1))
        ax.plot(DELTA_D_GRID, all_S[theta], "o-", ms=4, color=color,
                label=rf"$\theta={theta:.0f}°$")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("Dominance index  $S$")
    ax.set_title(f"Fixed-point rotation competition  (N={N})")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fp_rotation_S_vs_deltaD.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_rotation_S_vs_deltaD.png")

    # ── Heatmap ──
    S_mat = np.array([all_S[t] for t in thetas])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        S_mat.T, aspect="auto", origin="lower",
        extent=[thetas[0], thetas[-1], DELTA_D_GRID[0], DELTA_D_GRID[-1]],
        cmap="RdBu_r", vmin=-1, vmax=1,
    )
    ax.set_xlabel(r"$\theta$  (rotation angle, °)")
    ax.set_ylabel(r"$\Delta D$")
    ax.set_title(f"Heatmap: FP rotation competition  (N={N})")
    plt.colorbar(im, ax=ax, label="S")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_rotation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_rotation_heatmap.png")

    # ── Sharpness vs θ ──
    sharpness_vals = [selection_sharpness(DELTA_D_GRID, all_S[t]) for t in thetas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thetas, sharpness_vals, "s-", ms=7, color="darkorange")
    ax.set_xlabel(r"$\theta$  (°)")
    ax.set_ylabel("Selection sharpness  (dS/dΔD)")
    ax.set_title(f"Sharpness vs rotation angle  (N={N})")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_rotation_sharpness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_rotation_sharpness.png")

    # ── Phase portraits (2×3) ──
    delta_D_show = [-1.0, 0.0, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, theta in enumerate([0.0, 90.0]):
        A2 = A_fixed_point_rotated(A_base, theta)
        for col, dD in enumerate(delta_D_show):
            ax = axes[row, col]
            D1, D2 = delta_D_to_D1_D2(dD)
            _, z1, z2, metrics = run_single_experiment(
                N=N, A1=A_base, A2=A2, D1=D1, D2=D2,
                dt=dt, t_max=t_max, seed=seed, x0_seed=seed + 100 + row * 10 + col,
            )
            ax.plot(z1[:, 0], z1[:, 1], color="purple", alpha=0.8, lw=1.2, label=r"Task 1 ($z^{(1)}$)")
            ax.plot(z2[:, 0], z2[:, 1], color="red", alpha=0.8, lw=1.2, label=r"Task 2 ($z^{(2)}$)")
            ax.set_aspect("equal")
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_title(rf"$\theta={theta:.0f}°$,  $\Delta D={dD:.0f}$   S={metrics['S']:.2f}",
                         fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$z_1$")
            if col == 0:
                ax.set_ylabel(r"$z_2$")
            if row == 0 and col == 2:
                ax.legend(fontsize=7)
    fig.suptitle(f"Phase portraits — FP rotation competition  (N={N})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fp_rotation_phase_portraits.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_rotation_phase_portraits.png")

    return all_S, sharpness_vals


# ──────────────────────────────────────────────────────────
#  Experiment 2b: Fixed-point competition — off-diagonal Δb
# ──────────────────────────────────────────────────────────

def run_fp_coupling_experiment(
    N: int = 500,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    out_dir: Path = Path("comparison_results"),
):
    """Sweep off-diagonal coupling difference Δb × ΔD for two FP tasks."""
    print("── Experiment 2b: Fixed-point coupling competition ──")
    b_base = 0.3
    diag = 0.5
    # Δb range: keep b₂ = b_base + Δb < diag to stay positive-definite
    delta_bs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    A1 = A_fixed_point(b=b_base, diag=diag)

    all_S = {}
    csv_rows = []

    for db in delta_bs:
        b2 = b_base + db
        # Warn if eigenvalue goes non-positive (diag - b < 0)
        if b2 >= diag:
            print(f"  WARNING: b₂={b2:.2f} ≥ diag={diag}, A₂ eigenvalue ≤ 0. Skipping.")
            continue
        A2 = A_fixed_point(b=b2, diag=diag)
        dD_arr, S_arr = sweep_delta_D(A1, A2, N, DELTA_D_GRID, dt, t_max, seed)
        all_S[db] = S_arr
        for j, dD in enumerate(dD_arr):
            csv_rows.append([db, dD, S_arr[j]])
        print(f"  Δb={db:.2f}  done  (sharpness={selection_sharpness(dD_arr, S_arr):.4f})")

    csv_path = out_dir / "fp_coupling_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_b", "delta_D", "S"])
        w.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    valid_dbs = [db for db in delta_bs if db in all_S]

    # ── S vs ΔD ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.cool
    for i, db in enumerate(valid_dbs):
        color = cmap(i / max(len(valid_dbs) - 1, 1))
        ax.plot(DELTA_D_GRID, all_S[db], "o-", ms=4, color=color,
                label=rf"$\Delta b={db:.2f}$")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("Dominance index  $S$")
    ax.set_title(f"Fixed-point coupling competition  (N={N})")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fp_coupling_S_vs_deltaD.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_coupling_S_vs_deltaD.png")

    # ── Heatmap ──
    S_mat = np.array([all_S[db] for db in valid_dbs])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        S_mat.T, aspect="auto", origin="lower",
        extent=[valid_dbs[0], valid_dbs[-1], DELTA_D_GRID[0], DELTA_D_GRID[-1]],
        cmap="RdBu_r", vmin=-1, vmax=1,
    )
    ax.set_xlabel(r"$\Delta b$  (coupling difference)")
    ax.set_ylabel(r"$\Delta D$")
    ax.set_title(f"Heatmap: FP coupling competition  (N={N})")
    plt.colorbar(im, ax=ax, label="S")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_coupling_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_coupling_heatmap.png")

    # ── Sharpness vs Δb ──
    sharpness_vals = [selection_sharpness(DELTA_D_GRID, all_S[db]) for db in valid_dbs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(valid_dbs, sharpness_vals, "s-", ms=7, color="steelblue")
    ax.set_xlabel(r"$\Delta b$")
    ax.set_ylabel("Selection sharpness  (dS/dΔD)")
    ax.set_title(f"Sharpness vs coupling difference  (N={N})")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_coupling_sharpness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fp_coupling_sharpness.png")

    return all_S, sharpness_vals


# ──────────────────────────────────────────────────────────
#  Experiment 3: Cross-comparison (LC vs FP vs Mixed)
# ──────────────────────────────────────────────────────────

def run_cross_comparison(
    N: int = 500,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    out_dir: Path = Path("comparison_results"),
):
    """Compare WTA transition for LC-vs-LC, FP-vs-FP, and mixed LC-vs-FP."""
    print("── Experiment 3: Cross-comparison ──")

    A_lc = A_limit_cycle(0.5)
    A_fp = A_fixed_point(0.3, 0.5)

    cases = {
        "Both LC":  (A_lc, A_lc),
        "Both FP":  (A_fp, A_fp),
        "Mixed (LC vs FP)": (A_lc, A_fp),
    }

    results = {}
    csv_rows = []

    for label, (A1, A2) in cases.items():
        dD_arr, S_arr = sweep_delta_D(A1, A2, N, DELTA_D_GRID, dt, t_max, seed)
        results[label] = S_arr
        for j, dD in enumerate(dD_arr):
            csv_rows.append([label, dD, S_arr[j]])
        sharp = selection_sharpness(dD_arr, S_arr)
        print(f"  {label:20s}  sharpness={sharp:.4f}")

    csv_path = out_dir / "cross_comparison_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "delta_D", "S"])
        w.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    # ── S vs ΔD overlay ──
    colors = {"Both LC": "purple", "Both FP": "teal", "Mixed (LC vs FP)": "darkorange"}
    styles = {"Both LC": "o-", "Both FP": "s-", "Mixed (LC vs FP)": "^-"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in cases:
        ax.plot(DELTA_D_GRID, results[label], styles[label], ms=5,
                color=colors[label], label=label, lw=1.8)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("Dominance index  $S$")
    ax.set_title(f"Cross-comparison: WTA transition  (N={N})")
    ax.legend(fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_comparison_S_vs_deltaD.png", dpi=150)
    plt.close(fig)
    print(f"  Saved cross_comparison_S_vs_deltaD.png")

    # ── Sharpness bar chart ──
    labels_list = list(cases.keys())
    sharp_vals = [selection_sharpness(DELTA_D_GRID, results[l]) for l in labels_list]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels_list, sharp_vals, color=[colors[l] for l in labels_list], edgecolor="k")
    ax.set_ylabel("Selection sharpness  (dS/dΔD)")
    ax.set_title(f"Sharpness comparison  (N={N})")
    for bar, v in zip(bars, sharp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_comparison_sharpness_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved cross_comparison_sharpness_bar.png")

    # ── Phase portraits (3×3): rows = case, cols = ΔD ∈ {-1, 0, 1} ──
    delta_D_show = [-1.0, 0.0, 1.0]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for row, (label, (A1, A2)) in enumerate(cases.items()):
        for col, dD in enumerate(delta_D_show):
            ax = axes[row, col]
            D1, D2 = delta_D_to_D1_D2(dD)
            _, z1, z2, metrics = run_single_experiment(
                N=N, A1=A1, A2=A2, D1=D1, D2=D2,
                dt=dt, t_max=t_max, seed=seed, x0_seed=seed + 200 + row * 10 + col,
            )
            ax.plot(z1[:, 0], z1[:, 1], color="purple", alpha=0.8, lw=1.0, label="Task 1")
            ax.plot(z2[:, 0], z2[:, 1], color="red", alpha=0.8, lw=1.0, label="Task 2")
            ax.set_aspect("equal")
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_title(f"{label}\n$\\Delta D={dD:.0f}$   S={metrics['S']:.2f}", fontsize=8)
            if row == 2:
                ax.set_xlabel(r"$z_1$")
            if col == 0:
                ax.set_ylabel(r"$z_2$")
            if row == 0 and col == 2:
                ax.legend(fontsize=7)
    fig.suptitle(f"Phase portraits — Cross-comparison  (N={N})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_comparison_phase_portraits.png", dpi=150)
    plt.close(fig)
    print(f"  Saved cross_comparison_phase_portraits.png")

    return results


# ──────────────────────────────────────────────────────────
#  Summary figure
# ──────────────────────────────────────────────────────────

def make_summary_figure(
    lc_S: dict, fp_rot_S: dict, fp_coup_S: dict, cross_S: dict,
    N: int, out_dir: Path,
):
    """4-panel summary combining key curves from all experiments."""
    print("── Summary figure ──")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: LC freq
    ax = axes[0, 0]
    cmap = plt.cm.viridis
    keys = sorted(lc_S.keys())
    for i, k in enumerate(keys):
        ax.plot(DELTA_D_GRID, lc_S[k], "-", lw=1.5,
                color=cmap(i / max(len(keys) - 1, 1)),
                label=rf"$\Delta\omega={k:.1f}$")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("S")
    ax.set_title("A) Limit-cycle freq. difference")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-1.05, 1.05)

    # Panel B: FP rotation
    ax = axes[0, 1]
    cmap = plt.cm.plasma
    keys = sorted(fp_rot_S.keys())
    for i, k in enumerate(keys):
        ax.plot(DELTA_D_GRID, fp_rot_S[k], "-", lw=1.5,
                color=cmap(i / max(len(keys) - 1, 1)),
                label=rf"$\theta={k:.0f}°$")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("S")
    ax.set_title("B) Fixed-point rotation angle")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-1.05, 1.05)

    # Panel C: FP coupling
    ax = axes[1, 0]
    cmap = plt.cm.cool
    keys = sorted(fp_coup_S.keys())
    for i, k in enumerate(keys):
        ax.plot(DELTA_D_GRID, fp_coup_S[k], "-", lw=1.5,
                color=cmap(i / max(len(keys) - 1, 1)),
                label=rf"$\Delta b={k:.2f}$")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("S")
    ax.set_title("C) Fixed-point coupling difference")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-1.05, 1.05)

    # Panel D: Cross-comparison
    ax = axes[1, 1]
    colors = {"Both LC": "purple", "Both FP": "teal", "Mixed (LC vs FP)": "darkorange"}
    for label, S_arr in cross_S.items():
        ax.plot(DELTA_D_GRID, S_arr, "o-", ms=4, lw=1.8,
                color=colors[label], label=label)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\Delta D$")
    ax.set_ylabel("S")
    ax.set_title("D) Cross-comparison: LC vs FP vs Mixed")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.05, 1.05)

    fig.suptitle(f"Two-task competition comparison  (N={N}, P=2, R=2)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary_comparison.png")


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def run_all(
    N: int = 500,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    output_dir: str = "comparison_results",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Two-task competition comparison experiments")
    print(f"  N={N}  |  t_max={t_max}  |  seed={seed}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*60}\n")

    lc_S, _ = run_lc_freq_experiment(N, dt, t_max, seed, out_dir)
    print()
    fp_rot_S, _ = run_fp_rotation_experiment(N, dt, t_max, seed, out_dir)
    print()
    fp_coup_S, _ = run_fp_coupling_experiment(N, dt, t_max, seed, out_dir)
    print()
    cross_S = run_cross_comparison(N, dt, t_max, seed, out_dir)
    print()
    make_summary_figure(lc_S, fp_rot_S, fp_coup_S, cross_S, N, out_dir)

    print(f"\n{'='*60}")
    print(f"  All done!  Results in: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-task competition comparison experiments")
    parser.add_argument("-o", "--output-dir", default="comparison_results")
    parser.add_argument("--t-max", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-N", type=int, default=500)
    args = parser.parse_args()

    run_all(
        N=args.N,
        dt=args.dt,
        t_max=args.t_max,
        seed=args.seed,
        output_dir=args.output_dir,
    )

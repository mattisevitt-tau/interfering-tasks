#!/usr/bin/env python3
"""
Experiment 2 (improved): Amplitude/radius competition.

Family 2:  A(γ) = γ [[1, 0.5], [-0.5, 1]]
Reference: A⁽¹⁾ = A(0.8)
Sweep:     A⁽²⁾ = A(0.8 + Δγ)

Generates:
  1. S vs Δγ at several fixed ΔD values  (left panel)
  2. S vs Δγ at ΔD=0 — does amplitude diff alone drive WTA?  (right panel)
  3. dS/dΔγ at Δγ=0 for different D_mean values

Multi-seed averaging (default 5 seeds) to reduce finite-N noise.

Usage:
    python run_exp2_amplitude.py [-N 500] [--n-seeds 5] [-o comparison_results]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import (
    run_single_experiment,
    D_MEAN,
    A_amplitude,
)


# ---------- Multi-seed single run ----------

def run_single_averaged(N, A1, A2, D1, D2, n_seeds=5, base_seed=42, dt=0.05, t_max=100.0):
    """Run same config with multiple random seeds. Returns (mean_S, std_S)."""
    S_vals = []
    for k in range(n_seeds):
        seed_k = base_seed + k * 1000
        _, _, _, metrics = run_single_experiment(
            N=N, A1=A1, A2=A2, D1=D1, D2=D2,
            dt=dt, t_max=t_max,
            seed=seed_k, x0_seed=seed_k + 500,
        )
        S_vals.append(metrics["S"])
    return float(np.mean(S_vals)), float(np.std(S_vals))


# ---------- Core sweep: S vs Δγ at a given ΔD ----------

def sweep_delta_gamma(delta_gammas, delta_D, N, D_mean=D_MEAN, gamma_base=0.8,
                      n_seeds=5, base_seed=42, dt=0.05, t_max=100.0):
    D1 = D_mean + delta_D / 2.0
    D2 = D_mean - delta_D / 2.0
    A1 = A_amplitude(gamma_base)
    S_mean = np.zeros(len(delta_gammas))
    S_std = np.zeros(len(delta_gammas))
    for i, dg in enumerate(delta_gammas):
        A2 = A_amplitude(gamma_base + dg)
        m, s = run_single_averaged(N, A1, A2, D1, D2, n_seeds, base_seed, dt, t_max)
        S_mean[i] = m
        S_std[i] = s
    return delta_gammas, S_mean, S_std


# ---------- Plot: S vs Δγ (two-panel) ----------

def plot_S_vs_delta_gamma(delta_gammas, delta_D_values, results, N, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    colors = {-1.0: "#1f77b4", -0.5: "#7eb8da", 0.0: "#2ca02c",
              0.5: "#f5a623", 1.0: "#d62728"}
    for dD in delta_D_values:
        S_mean, S_std = results[dD]
        c = colors.get(dD, "gray")
        ax.plot(delta_gammas, S_mean, "o-", ms=6, lw=1.8, color=c,
                label=rf"$\Delta D = {dD:+.1f}$")
        ax.fill_between(delta_gammas, S_mean - S_std, S_mean + S_std,
                        alpha=0.15, color=c)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta\gamma$  (amplitude difference)", fontsize=11)
    ax.set_ylabel("Dominance index  $S$", fontsize=11)
    ax.set_title("Amplitude/radius\n"
                 r"$S$ vs $\Delta\gamma$, at several $\Delta D$ values", fontsize=11)
    ax.legend(title=r"$\Delta D$ (strength gap)", fontsize=8, title_fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(delta_gammas[0] - 0.02, delta_gammas[-1] + 0.02)

    ax = axes[1]
    S_mean_0, S_std_0 = results[0.0]
    ax.plot(delta_gammas, S_mean_0, "o-", ms=7, lw=2, color="k")
    ax.fill_between(delta_gammas, S_mean_0 - S_std_0, S_mean_0 + S_std_0,
                    alpha=0.2, color="gray")
    ax.fill_between(delta_gammas, 0, S_mean_0,
                    where=(S_mean_0 > 0), alpha=0.25, color="mediumpurple")
    ax.fill_between(delta_gammas, 0, S_mean_0,
                    where=(S_mean_0 < 0), alpha=0.25, color="lightsalmon")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$\Delta\gamma$  (amplitude difference)", fontsize=11)
    ax.set_ylabel("Dominance index  $S$", fontsize=11)
    ax.set_title(r"Amplitude — $\Delta D = 0$ (equal strength)" "\n"
                 r"Does amplitude difference alone drive winner-take-all?", fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(delta_gammas[0] - 0.02, delta_gammas[-1] + 0.02)
    ax.text(0.02, 0.97,
            "$S > 0$: Task 1 (lower $\\gamma$) wins\n"
            "$S < 0$: Task 2 (higher $\\gamma$) wins\n"
            "$S \\approx 0$: no winner",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(f"Exp 2 — Effect of amplitude difference on winner-take-all  (N={N})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------- dS/dΔγ helpers ----------

def compute_dS_dGamma_at_zero(delta_gammas, S_mean):
    """Finite-difference derivative at Δγ≈0."""
    idx = np.argmin(np.abs(delta_gammas))
    if 0 < idx < len(delta_gammas) - 1:
        return float((S_mean[idx + 1] - S_mean[idx - 1])
                     / (delta_gammas[idx + 1] - delta_gammas[idx - 1]))
    if idx < len(delta_gammas) - 1:
        return float((S_mean[idx + 1] - S_mean[idx])
                     / (delta_gammas[idx + 1] - delta_gammas[idx]))
    return 0.0


def run_dS_dGamma_vs_D(D_mean_values, delta_D_values, N, gamma_base=0.8,
                        n_seeds=5, base_seed=42, dt=0.05, t_max=100.0):
    """For each D_mean × ΔD, compute dS/dΔγ at Δγ=0 with error bars."""
    d_gammas_fine = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    results = {}
    for D_m in D_mean_values:
        results[D_m] = {}
        for dD in delta_D_values:
            D1 = D_m + dD / 2.0
            D2 = D_m - dD / 2.0
            A1 = A_amplitude(gamma_base)
            slopes_per_seed = []
            for k in range(n_seeds):
                seed_k = base_seed + k * 1000
                S_vals = []
                for dg in d_gammas_fine:
                    A2 = A_amplitude(gamma_base + dg)
                    _, _, _, metrics = run_single_experiment(
                        N=N, A1=A1, A2=A2, D1=D1, D2=D2,
                        dt=dt, t_max=t_max,
                        seed=seed_k, x0_seed=seed_k + 500,
                    )
                    S_vals.append(metrics["S"])
                slope = compute_dS_dGamma_at_zero(d_gammas_fine, np.array(S_vals))
                slopes_per_seed.append(slope)
            results[D_m][dD] = (float(np.mean(slopes_per_seed)),
                                float(np.std(slopes_per_seed)))
        print(f"  D_mean={D_m:.1f} done")
    return results


def plot_dS_dGamma_vs_D(D_mean_values, delta_D_values, slope_results, N, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.coolwarm
    n_dD = len(delta_D_values)
    for i, dD in enumerate(delta_D_values):
        color = cmap(i / max(n_dD - 1, 1))
        means = [slope_results[D_m][dD][0] for D_m in D_mean_values]
        stds = [slope_results[D_m][dD][1] for D_m in D_mean_values]
        ax.errorbar(D_mean_values, means, yerr=stds, fmt="o-", ms=6, lw=1.8,
                    color=color, capsize=3,
                    label=rf"$\Delta D = {dD:+.1f}$")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel(r"$D_{\mathrm{mean}}$  (connection strength)", fontsize=11)
    ax.set_ylabel(r"$dS / d\Delta\gamma$  at $\Delta\gamma = 0$", fontsize=11)
    ax.set_title(f"Amplitude sensitivity vs connection strength  (N={N})", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------- Main ----------

def run_all(N=500, n_seeds=5, dt=0.05, t_max=100.0, base_seed=42,
            output_dir="comparison_results"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Exp 2: Amplitude competition (multi-seed avg)")
    print(f"  N={N}  |  n_seeds={n_seeds}  |  t_max={t_max}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*60}\n")

    delta_gammas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    delta_D_show = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Part 1: S vs Δγ
    print("-- Part 1: S vs delta_gamma at fixed delta_D (multi-seed) --")
    results = {}
    csv_rows = []
    for dD in delta_D_show:
        _, S_mean, S_std = sweep_delta_gamma(
            delta_gammas, dD, N, D_mean=D_MEAN, gamma_base=0.8,
            n_seeds=n_seeds, base_seed=base_seed, dt=dt, t_max=t_max,
        )
        results[dD] = (S_mean, S_std)
        for i, dg in enumerate(delta_gammas):
            csv_rows.append([dD, dg, S_mean[i], S_std[i]])
        print(f"  delta_D={dD:+.1f}  done")

    csv_path = out_dir / "exp2_S_vs_delta_gamma.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_D", "delta_gamma", "S_mean", "S_std"])
        w.writerows(csv_rows)
    print(f"  Saved {csv_path}")

    plot_S_vs_delta_gamma(delta_gammas, delta_D_show, results, N,
                          out_dir / "exp2_S_vs_delta_gamma.png")

    # Part 2: dS/dΔγ vs D_mean
    print("\n-- Part 2: dS/d(delta_gamma) at delta_gamma=0 vs D_mean --")
    D_mean_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    dD_for_slope = [0.0, 0.5, 1.0]

    slope_results = run_dS_dGamma_vs_D(
        D_mean_values, dD_for_slope, N,
        gamma_base=0.8, n_seeds=n_seeds, base_seed=base_seed,
        dt=dt, t_max=t_max,
    )

    csv_path2 = out_dir / "exp2_dS_dGamma_vs_D.csv"
    with open(csv_path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D_mean", "delta_D", "dS_dGamma_mean", "dS_dGamma_std"])
        for D_m in D_mean_values:
            for dD in dD_for_slope:
                m, s = slope_results[D_m][dD]
                w.writerow([D_m, dD, m, s])
    print(f"  Saved {csv_path2}")

    plot_dS_dGamma_vs_D(D_mean_values, dD_for_slope, slope_results, N,
                        out_dir / "exp2_dS_dGamma_vs_D.png")

    print(f"\n{'='*60}")
    print(f"  All done!  Results in: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 2: Amplitude competition (multi-seed)")
    parser.add_argument("-o", "--output-dir", default="comparison_results")
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random seeds to average over (default: 5)")
    parser.add_argument("--t-max", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all(N=args.N, n_seeds=args.n_seeds, dt=args.dt, t_max=args.t_max,
            base_seed=args.seed, output_dir=args.output_dir)

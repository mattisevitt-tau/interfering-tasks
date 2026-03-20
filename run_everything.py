#!/usr/bin/env python3
"""
Unified runner: execute all experiments and compile output figures
into a single PDF report.

Calls the existing entry points:
  1. run_exp1_frequency.run_all()      — Family 1: Frequency (ω)
  2. run_exp2_amplitude.run_all()      — Family 2: Amplitude (γ)
  3. run_exp3_shape.run_all()          — Family 3: Shape (ε)
  4. run_comparison_experiments.run_all() — Cross-regime comparisons
  5. run_avg_feature_maps.run_all()    — Seed-averaged heatmaps

Then collects every PNG produced and assembles them into one PDF
with section dividers.

Usage:
    python run_everything.py [-N 500] [--n-seeds 5] [--t-max 100] [--seed 42] [-o results]
"""

import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from experiments import D_MEAN
import run_exp1_frequency
import run_exp2_amplitude
import run_exp3_shape
import run_comparison_experiments
import run_avg_feature_maps


# ── PDF helpers ──────────────────────────────────────────────

def add_text_page(pdf: PdfPages, title: str, body: str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.92, title, ha="center", va="top",
             fontsize=16, fontweight="bold")
    wrapped = "\n".join(
        "\n".join(textwrap.wrap(line, width=100)) if line.strip() else ""
        for line in body.strip().splitlines()
    )
    fig.text(0.06, 0.84, wrapped, ha="left", va="top", fontsize=11,
             family="monospace", linespacing=1.45)
    pdf.savefig(fig)
    plt.close(fig)


def add_images(pdf: PdfPages, image_dir: Path):
    paths = sorted(image_dir.rglob("*.png"))
    for p in paths:
        img = Image.open(p)
        w, h = img.size
        dpi = 150
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.figimage(img, 0, 0)
        fig.suptitle(p.name, fontsize=9, y=0.99, va="top", color="gray")
        pdf.savefig(fig)
        plt.close(fig)
    return paths


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ALL experiments and produce a combined PDF report.")
    parser.add_argument("-o", "--output-dir", default="results")
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--t-max", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--D-mean", type=float, default=D_MEAN)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families_dir = out_dir / "families"
    comparison_dir = out_dir / "comparison"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"full_report_{timestamp}.pdf"

    common = dict(N=args.N, n_seeds=args.n_seeds, dt=args.dt,
                  t_max=args.t_max, base_seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  UNIFIED EXPERIMENT RUNNER")
    print(f"  N={args.N}  |  n_seeds={args.n_seeds}  |  t_max={args.t_max}")
    print(f"  Output: {pdf_path}")
    print(f"{'='*60}\n")

    # ── 1–3. Three blueprint families (multi-seed deep-dives) ──
    print("Family 1: Frequency (ω)")
    run_exp1_frequency.run_all(**common, output_dir=str(families_dir))

    print("\nFamily 2: Amplitude (γ)")
    run_exp2_amplitude.run_all(**common, output_dir=str(families_dir))

    print("\nFamily 3: Shape (ε)")
    run_exp3_shape.run_all(**common, output_dir=str(families_dir))

    # ── 4. Comparison experiments ──
    print("\nComparison experiments")
    run_comparison_experiments.run_all(
        N=args.N, dt=args.dt, t_max=args.t_max,
        seed=args.seed, output_dir=str(comparison_dir),
    )

    # ── 5. Averaged feature heatmaps ──
    print("\nAveraged feature heatmaps")
    run_avg_feature_maps.run_all(
        N=args.N, n_seeds=args.n_seeds, dt=args.dt,
        t_max=args.t_max, base_seed=args.seed,
        D_mean=args.D_mean, output_dir=str(comparison_dir),
    )

    # ── Assemble PDF ──
    print(f"\nAssembling PDF report → {pdf_path}")
    with PdfPages(str(pdf_path)) as pdf:

        add_text_page(pdf, "Two-Task Competition in Low-Rank RNNs", f"""
Full Experiment Report
Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Parameters:
  N = {args.N}   |   n_seeds = {args.n_seeds}   |   t_max = {args.t_max}
  dt = {args.dt}  |   seed = {args.seed}          |   D_mean = {args.D_mean}

Contents:
  Families  — Multi-seed deep-dives for all three blueprint families
    1. Frequency (ω):  S vs Δω, sensitivity dS/dΔω vs D_mean
    2. Amplitude (γ):  S vs Δγ, sensitivity dS/dΔγ vs D_mean
    3. Shape (ε):      S vs Δε, sensitivity dS/dΔε vs D_mean

  Comparison — Cross-regime experiments
    LC freq, FP rotation, FP coupling, cross-comparison
    Seed-averaged heatmaps for Δω and Δb
""")

        add_text_page(pdf, "Families 1–3: Multi-Seed Deep-Dives", """
For each of the three blueprint families:

  Family 1 — Frequency:   A(ω) = 0.8 [[1, ω], [-ω, 1]],       base ω = 0.5
  Family 2 — Amplitude:   A(γ) = γ   [[1, 0.5], [-0.5, 1]],    base γ = 0.8
  Family 3 — Shape:       A(ε) = 0.8 [[ε, 0.5], [-0.5, 1/ε]],  base ε = 1.0

Two analyses per family (multi-seed averaged):
  1. S vs Δ(feature) at several fixed ΔD values, with ±1σ bands
     Right panel: ΔD = 0 only — does feature difference alone
     drive winner-take-all?
  2. Sensitivity dS/d(Δfeature) at Δfeature = 0 as a function
     of mean connection strength D_mean
""")
        fam_pngs = add_images(pdf, families_dir)
        print(f"  Added {len(fam_pngs)} family figures")

        add_text_page(pdf, "Comparison Experiments", """
Cross-regime experiments and averaged heatmaps:

  Exp 1  — LC frequency competition (varying Δω)
  Exp 2a — FP rotation (varying eigenvector angle θ)
  Exp 2b — FP coupling (varying off-diagonal Δb)
  Exp 3  — Cross-comparison (LC-vs-LC, FP-vs-FP, Mixed)

  Averaged heatmaps of S(Δfeature, ΔD) for Δω and Δb
  S at ΔD = 0, sensitivity slopes at zero feature gap
""")
        comp_pngs = add_images(pdf, comparison_dir)
        print(f"  Added {len(comp_pngs)} comparison figures")

    print(f"\n{'='*60}")
    print(f"  Done!  Report: {pdf_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

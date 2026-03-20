# Interfering Tasks: Low-Rank RNN Simulation

Implementation of two-task competition in low-rank recurrent neural networks, based on Marschall et al. (2025). Studies similarity of dynamical motifs across feature families (frequency, amplitude, shape).

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `streamlit`.

## Quick Start

### Interactive UI

Run the Streamlit app for interactive simulations and blueprint experiments:

```bash
streamlit run run_interactive.py
```

- **Interactive Simulation** tab: explore single- and two-task limit cycles with custom A matrices, D values, N, etc.
- **Blueprint Experiments** tab: run heatmaps, phase portraits, scaling plots from the Marschall et al. blueprint.
- **Run all experiments**: button to execute the complete blueprint suite (see below).

### Command-Line Figures (Marschall Scenarios)

Generate the original Marschall scenario figures (Fig 2A–C):

```bash
python main.py [a|b|c|all] [options]
```

Examples:

```bash
python main.py all
python main.py a --no-display -N 2000 -o ./figures
```

Options: `--no-display`, `-N`, `--t-max`, `--dt`, `-o`, `--seed`.

## Run All Blueprint Experiments

Execute the full experiment suite from the blueprint: 3 feature families × 3 network sizes (N=200, 500, 1000), heatmaps, phase portraits, scaling plots.

### Option 1: From the UI

1. `streamlit run run_interactive.py`
2. Open the **Blueprint Experiments** tab.
3. Click **Run all experiments**.
4. Results are written to `experiment_results/run_YYYYMMDD_HHMMSS/`.

### Option 2: From the Command Line

```bash
python run_all_experiments.py [options]
```

Examples:

```bash
python run_all_experiments.py
python run_all_experiments.py -o ./my_results --t-max 100 --seed 123
```

Options:

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output-dir` | Output directory | `experiment_results/run_YYYYMMDD_HHMMSS` |
| `--t-max` | Simulation time per run | 100 |
| `--dt` | RK4 timestep | 0.05 |
| `--seed` | Random seed | 42 |

## Output Structure

After running all experiments:

```
experiment_results/run_YYYYMMDD_HHMMSS/
├── sweep_frequency_omega_N200.csv
├── sweep_frequency_omega_N500.csv
├── sweep_frequency_omega_N1000.csv
├── sweep_amplitude_gamma_N200.csv
├── sweep_amplitude_gamma_N500.csv
├── sweep_amplitude_gamma_N1000.csv
├── sweep_shape_epsilon_N200.csv
├── sweep_shape_epsilon_N500.csv
├── sweep_shape_epsilon_N1000.csv
├── summary.csv
└── graphs/
    ├── heatmap_frequency_omega_N200.png
    ├── heatmap_frequency_omega_N500.png
    ├── ...
    ├── scaling_frequency_omega.png
    ├── scaling_amplitude_gamma.png
    ├── scaling_shape_epsilon.png
    ├── phase_portraits_frequency_omega_N500.png
    ├── phase_portraits_amplitude_gamma_N500.png
    └── phase_portraits_shape_epsilon_N500.png
```

### CSV Files

- **sweep_&lt;family&gt;_N&lt;n&gt;.csv**: Full sweep over ΔFeature and ΔD. Columns: `family`, `N`, `delta_feature`, `delta_D`, `S` (dominance index).
- **summary.csv**: Scaling slopes. Columns: `family`, `N`, `selection_sharpness` (dS/dΔD at ΔD=0).

### Graphs

- **heatmap_***: X = ΔFeature, Y = ΔD, color = S.
- **scaling_***: Selection sharpness vs N.
- **phase_portraits_***: z⁽¹⁾ and z⁽²⁾ for ΔD ∈ {-1, 0, 1}.

## Multi-Seed Family Deep-Dives

Three scripts provide a deeper investigation of each blueprint family with multi-seed averaging (default 5 seeds) to reduce finite-N noise. Each produces:

- **S vs Δ(feature)** at several fixed ΔD values, with ±1σ error bands
- **S vs Δ(feature) at ΔD=0** — does the feature difference alone drive winner-take-all?
- **Sensitivity dS/d(Δfeature) at Δfeature=0** as a function of mean connection strength D

| Script | Family | A-matrix | Base value |
|--------|--------|----------|------------|
| `run_exp1_frequency.py` | Frequency (ω) | A(ω) = 0.8 [[1, ω], [-ω, 1]] | ω = 0.5 |
| `run_exp2_amplitude.py` | Amplitude (γ) | A(γ) = γ [[1, 0.5], [-0.5, 1]] | γ = 0.8 |
| `run_exp3_shape.py` | Shape (ε) | A(ε) = 0.8 [[ε, 0.5], [-0.5, 1/ε]] | ε = 1.0 |

Run individually:

```bash
python run_exp1_frequency.py [-N 500] [--n-seeds 5] [-o comparison_results]
python run_exp2_amplitude.py [-N 500] [--n-seeds 5] [-o comparison_results]
python run_exp3_shape.py [-N 500] [--n-seeds 5] [-o comparison_results]
```

## Comparison Experiments

Additional scripts explore two-task competition across different dynamical regimes, going beyond the original blueprint scenarios.

### `run_comparison_experiments.py`

Runs four experiments in a single pass and produces CSVs, heatmaps, S-vs-ΔD curves, sharpness plots, phase portraits, and a 4-panel summary figure:

| Experiment | Description |
|------------|-------------|
| 1 — LC frequency | Two limit-cycle tasks with varying frequency difference Δω |
| 2a — FP rotation | Two fixed-point tasks with varying eigenvector rotation angle θ |
| 2b — FP coupling | Two fixed-point tasks with varying off-diagonal coupling Δb |
| 3 — Cross-comparison | LC-vs-LC, FP-vs-FP, and mixed LC-vs-FP side by side |

```bash
python run_comparison_experiments.py [-N 500] [--t-max 100] [--seed 42] [-o comparison_results]
```

### `run_avg_feature_maps.py`

Generates seed-averaged heatmaps of S(Δfeature, ΔD) for frequency (Δω) and coupling (Δb) differences, plus sensitivity slopes at zero feature gap:

```bash
python run_avg_feature_maps.py [--n-seeds 5] [-N 500] [-o comparison_results]
```

## Run Everything

A single unified runner that executes all experiments and compiles every output figure into one PDF report:

```bash
python run_everything.py [-N 500] [--n-seeds 5] [--t-max 100] [--seed 42] [-o results]
```

This runs, in order:

1. **Family deep-dives** — `run_exp1_frequency`, `run_exp2_amplitude`, `run_exp3_shape`
2. **Comparison experiments** — `run_comparison_experiments`
3. **Averaged feature heatmaps** — `run_avg_feature_maps`

All PNGs are collected into `results/full_report_<timestamp>.pdf` with section dividers and explanations.

| Option | Description | Default |
|--------|-------------|---------|
| `-o` | Output directory | `results` |
| `-N` | Network size | 500 |
| `--n-seeds` | Seeds for multi-seed averaging | 5 |
| `--t-max` | Simulation time per run | 100 |
| `--dt` | RK4 timestep | 0.05 |
| `--seed` | Base random seed | 42 |
| `--D-mean` | Mean connection strength | 2.5 |

## Project Structure

```
├── core.py                      # φ(x) = erf(√π/2 · x)
├── task_component.py            # Task (D, A), loading vectors m, n
├── multi_task_network.py        # J matrix, RK4 dynamics
├── simulation.py                # Time integration, latent recording
├── visualizer.py                # Trajectory plots
├── experiments.py               # Blueprint experiments, metrics, sweeps
├── run_interactive.py           # Streamlit UI
├── run_all_experiments.py       # CLI for full experiment suite
├── main.py                      # Marschall scenarios (Fig 2A–C)
├── run_exp1_frequency.py        # Family 1 deep-dive (multi-seed)
├── run_exp2_amplitude.py        # Family 2 deep-dive (multi-seed)
├── run_exp3_shape.py            # Family 3 deep-dive (multi-seed)
├── run_comparison_experiments.py # Comparison across dynamical regimes
├── run_avg_feature_maps.py      # Averaged heatmaps and sensitivity
├── run_everything.py            # Unified runner → PDF report
└── README.md
```

## Blueprint Summary (Marschall et al. 2025)

- **Network**: τẋᵢ = -xᵢ + Σⱼ Jᵢⱼ φ(xⱼ), τ=1, φ(x)=erf(√π/2·x).
- **Connectivity**: Jᵢⱼ = Σ_μ D⁽μ⁾ Σ_r mᵢ⁽μ,r⁾ nⱼ⁽μ,r⁾, R=2, P=2.
- **Feature families**: Frequency (ω), Amplitude (γ), Shape (ε).
- **Competition**: D₁ = D_mean + ΔD/2, D₂ = D_mean - ΔD/2.
- **Metrics**: Dominance index S = (P₁ - P₂)/(P₁ + P₂), post–20% burn-in.

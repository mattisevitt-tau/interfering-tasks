"""
Interactive two-task CTRNN runner with Streamlit UI.
Run: streamlit run run_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from task_component import TaskComponent
from multi_task_network import MultiTaskNetwork
from simulation import Simulation
from visualizer import Visualizer


def limit_cycle_A(mu_axis1: float, mu_axis2: float, omega: float, theta_deg: float = 0.0) -> np.ndarray:
    """
    Build 2x2 A for limit cycle. A0 = [[mu_axis1, omega], [-omega, mu_axis2]];
    first arg = z1 diagonal, second = z2 diagonal. When equal circle, else ellipse.
    """
    theta = np.deg2rad(theta_deg)
    A0 = np.array([[mu_axis1, omega], [-omega, mu_axis2]], dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ A0 @ R.T


def _plot_flow_field(ax, A, lim, center=(0.0, 0.0), n_grid=20, use_streamplot=True):
    A = np.asarray(A)[:2, :2]
    cen = np.array(center)
    x = np.linspace(-lim, lim, n_grid)
    y = np.linspace(-lim, lim, n_grid)
    X, Y = np.meshgrid(x, y)
    z_display = np.stack([X.ravel(), Y.ravel()], axis=1)
    z = z_display - cen
    dz = (A @ z.T).T
    dZ1 = dz[:, 0].reshape(X.shape)
    dZ2 = dz[:, 1].reshape(Y.shape)
    if use_streamplot:
        ax.streamplot(X, Y, dZ1, dZ2, color=(0.5, 0.5, 0.5, 0.4), density=1.2, linewidth=0.6, zorder=0)
    else:
        scale = lim / (np.hypot(dZ1, dZ2).max() + 1e-12) * 0.15
        ax.quiver(X, Y, dZ1, dZ2, color="gray", alpha=0.4, scale=scale, zorder=0)


def _run_and_plot_single(
    ax, tc, N, R, dt, t_max, seed, x0_seed, x0_scale, color, title,
    radius_scale=1.0, center=(0.0, 0.0),
):
    """Run single-task network and plot its latent trajectory on ax."""
    network = MultiTaskNetwork([tc], N)
    sim = Simulation(network, dt=dt, t_max=t_max)
    rng = np.random.default_rng(x0_seed)
    x0 = rng.standard_normal(N) * x0_scale
    t, _, z_traj = sim.run(x0)
    z = radius_scale * z_traj[0] + np.array(center)
    viz = Visualizer(network, sim)
    lim = max(np.abs(z).max() * 1.15, 10)
    _plot_flow_field(ax, tc.A, lim, center=center)
    viz.plot_trajectory(0, t, z, ax, color=color)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$z_1(t)$")
    ax.set_ylabel(r"$z_2(t)$")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)


def run_four_panel_simulation(
    N: int,
    R: int,
    D1: float,
    D2: float,
    A1: list,
    A2: list,
    dt: float = 0.05,
    t_max: float = 100.0,
    seed: int = 42,
    x0_scale: float = 2.0,
    x0_seed: int = 456,
    radius_scale: float = 1.0,
    center: tuple = (0.0, 0.0),
):
    """Run single-task (top row) and two-task (bottom row) simulations; return 2x2 figure."""
    tc1 = TaskComponent(D=D1, A=A1, N=N, R=R, seed=seed)
    tc2 = TaskComponent(D=D2, A=A2, N=N, R=R, seed=seed + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cen = np.array(center)

    # Top row: single-task networks
    _run_and_plot_single(
        axes[0, 0], tc1, N, R, dt, t_max, seed, x0_seed, x0_scale,
        color="purple",
        title="Task 1 only (single-task network)",
        radius_scale=radius_scale,
        center=center,
    )
    _run_and_plot_single(
        axes[0, 1], tc2, N, R, dt, t_max, seed, x0_seed + 1, x0_scale,
        color="red",
        title="Task 2 only (single-task network)",
        radius_scale=radius_scale,
        center=center,
    )

    # Bottom row: two-task network
    network = MultiTaskNetwork([tc1, tc2], N)
    sim = Simulation(network, dt=dt, t_max=t_max)
    rng = np.random.default_rng(x0_seed + 2)
    x0 = rng.standard_normal(N) * x0_scale
    t, _, z_traj = sim.run(x0)
    z1 = radius_scale * z_traj[0] + cen
    z2 = radius_scale * z_traj[1] + cen

    lim1 = max(np.abs(z1).max() * 1.15, 10)
    lim2 = max(np.abs(z2).max() * 1.15, 10)
    viz = Visualizer(network, sim)

    _plot_flow_field(axes[1, 0], A1, lim1, center=center)
    viz.plot_trajectory(0, t, z1, axes[1, 0], color="purple")
    axes[1, 0].set_xlabel(r"$z_1^{(1)}(t)$")
    axes[1, 0].set_ylabel(r"$z_2^{(1)}(t)$")
    axes[1, 0].set_title("Task 1 (two-task network)")
    axes[1, 0].set_xlim(-lim1, lim1)
    axes[1, 0].set_ylim(-lim1, lim1)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].axvline(0, color="k", lw=0.5)

    _plot_flow_field(axes[1, 1], A2, lim2, center=center)
    viz.plot_trajectory(1, t, z2, axes[1, 1], color="red")
    axes[1, 1].set_xlabel(r"$z_1^{(2)}(t)$")
    axes[1, 1].set_ylabel(r"$z_2^{(2)}(t)$")
    axes[1, 1].set_title("Task 2 (two-task network)")
    axes[1, 1].set_xlim(-lim2, lim2)
    axes[1, 1].set_ylim(-lim2, lim2)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].axvline(0, color="k", lw=0.5)

    plt.tight_layout()
    return fig


# --- Streamlit UI ---

st.set_page_config(page_title="CTRNN Two-Task Runner", layout="wide")
st.title("CTRNN Two-Task Simulation")
st.markdown(
    "**Top row:** single-task networks (Task 1 only, Task 2 only). "
    "**Bottom row:** two-task network (Task 1 and Task 2 latent spaces). "
    "Each task is a limit cycle: set μ (growth), ω (frequency), θ (rotation) and D, N, R in the sidebar."
)

with st.sidebar:
    st.subheader("Network")
    N = st.number_input("N (neurons)", min_value=100, max_value=10000, value=2000, step=100)
    R = st.number_input("R (within-task dimension)", min_value=2, max_value=4, value=2, step=1)

    st.subheader("Task 1 — limit cycle")
    D1 = st.number_input("D₁ (task 1 strength)", value=2.2, step=0.1, format="%.1f", key="D1")
    st.caption("Different mu for z1 vs z2 gives ellipse; same gives circle. Need omega^2 > ((mu1-mu2)/2)^2")
    mu1_a = st.number_input("Task 1: μ for z₁ (axis 1)", value=0.8, step=0.05, format="%.2f", key="mu1_a")
    mu1_b = st.number_input("Task 1: μ for z₂ (axis 2)", value=0.8, step=0.05, format="%.2f", key="mu1_b")
    omega1 = st.number_input("ω₁ (angular freq, rad/s)", value=0.4, step=0.05, format="%.2f", key="omega1")
    theta1_deg = st.number_input("θ₁ (rotation, °)", value=0.0, step=5.0, format="%.1f", key="theta1")

    st.subheader("Task 2 — limit cycle")
    D2 = st.number_input("D₂ (task 2 strength)", value=2.0, step=0.1, format="%.1f", key="D2")
    mu2_a = st.number_input("Task 2: μ for z₁ (axis 1)", value=0.8, step=0.05, format="%.2f", key="mu2_a")
    mu2_b = st.number_input("Task 2: μ for z₂ (axis 2)", value=0.8, step=0.05, format="%.2f", key="mu2_b")
    omega2 = st.number_input("ω₂ (angular freq, rad/s)", value=0.4, step=0.05, format="%.2f", key="omega2")
    theta2_deg = st.number_input("θ₂ (rotation, °)", value=45.0, step=5.0, format="%.1f", key="theta2")

    st.subheader("Orbit (x, y, radius)")
    x0_scale = st.number_input("Initial scale (orbit size)", value=2.0, step=0.5, format="%.1f", key="x0_scale",
        help="Larger → larger limit cycle. D also affects size.")
    radius_scale = st.number_input("Radius scale (display)", value=1.0, step=0.1, format="%.1f", key="radius_scale",
        help="Scale factor applied to latent trajectory (z) for display.")
    z1_center = st.number_input("Center z₁ (x)", value=0.0, step=0.5, format="%.1f", key="z1_center")
    z2_center = st.number_input("Center z₂ (y)", value=0.0, step=0.5, format="%.1f", key="z2_center")

    st.subheader("Simulation")
    t_max = st.number_input("t_max", value=100.0, step=10.0, format="%.0f")
    seed = st.number_input("Seed", value=42, step=1)

if st.button("Run simulation", type="primary"):
    A1 = limit_cycle_A(float(mu1_a), float(mu1_b), float(omega1), float(theta1_deg))
    A2 = limit_cycle_A(float(mu2_a), float(mu2_b), float(omega2), float(theta2_deg))
    A1_list = A1.tolist()
    A2_list = A2.tolist()
    center = (float(z1_center), float(z2_center))

    with st.spinner("Running simulation…"):
        fig = run_four_panel_simulation(
            N=int(N),
            R=int(R),
            D1=D1,
            D2=D2,
            A1=A1_list,
            A2=A2_list,
            t_max=t_max,
            seed=int(seed),
            x0_scale=x0_scale,
            radius_scale=radius_scale,
            center=center,
        )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Computed A matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Task 1 — A₁**")
        st.latex(r"A_1 = \begin{pmatrix} %.3f & %.3f \\ %.3f & %.3f \end{pmatrix}" % (A1[0,0], A1[0,1], A1[1,0], A1[1,1]))
        st.caption("Task 1: μ(z₁)=%s, μ(z₂)=%s, ω=%s, θ=%s° → period ≈ %.2f s" % (mu1_a, mu1_b, omega1, theta1_deg, 2*np.pi/float(omega1)))
    with col2:
        st.markdown("**Task 2 — A₂**")
        st.latex(r"A_2 = \begin{pmatrix} %.3f & %.3f \\ %.3f & %.3f \end{pmatrix}" % (A2[0,0], A2[0,1], A2[1,0], A2[1,1]))
        st.caption("Task 2: μ(z₁)=%s, μ(z₂)=%s, ω=%s, θ=%s° → period ≈ %.2f s" % (mu2_a, mu2_b, omega2, theta2_deg, 2*np.pi/float(omega2)))
else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# Reference scales and dimensionless model parameters
# ============================================================

alpha = 0.157
beta = 0.954

# Use dimensional reference values only for the same conversions as in the existing dimensionless code.
V0 = 44900e-9               # [m^3]
s = 71e-6                  # [m^2]
r = 3.5e-3                 # [m]
I = 5.7e-5                 # [kg m^2]
P0 = 101300.0               # [Pa]
n = 0.00185                # [mol]
R = 8.314                   # [J/(mol K)]
Gamma = I / 20.0            # [kg m^2/s]
T_top = 24.0 + 273.15       # [K]
tau = 0.0137               # [s]

s_displacer = V0 / (2.0 * r)
time_scale = math.sqrt(n * R * T_top / I)  # [1/s]

# t_tilde = t * time_scale, omega_tilde = omega / time_scale
# DeltaT_tilde = DeltaT / T_top, V_tilde = V / (s_displacer * r)
# p_tilde = p * s_displacer * r / (n * R * T_top)
sigma_tilde = s / s_displacer
p_air_tilde = s_displacer * P0 * r / (n * R * T_top)
Gamma_tilde = Gamma / math.sqrt(n * R * T_top * I)
tau_tilde = tau * time_scale


# ============================================================
# User settings (all dimensionless)
# ============================================================

# Dimensionless temperature difference to plot. Same conditions as the previous setting of 0.167 K.
# Specify dimensionless values directly from now on (e.g., DeltaT_tilde_plot = 0.0176).
DeltaT_tilde_plot = 7 / T_top

# Whether to also plot the limit cycle.
PLOT_LIMIT_CYCLE = True

# Angular displacement from the saddle along an eigenvector with a theta component of 1.
MANIFOLD_DELTA_THETA = 1.0e-7

# Dimensionless integration time (corresponding to the previous setting of 7 s).
MANIFOLD_T_END = 7.0 * time_scale

# Number of points to save and plot, not the internal step size of solve_ivp.
MANIFOLD_N_EVAL = 20000

# Dimensionless times for the long integration and plotting window (previously 4000 s and 20 s).
LIMIT_CYCLE_T_END = 4000.0 * time_scale
LIMIT_CYCLE_PLOT_WINDOW = 20.0 * time_scale
LIMIT_CYCLE_N_EVAL = 20000

# Dimensionless initial angular velocity (corresponding to the previous setting of 20 rad/s).
INITIAL_OMEGA_MAGNITUDE = 20.0 / time_scale

# Display range of dimensionless angular velocity (automatic if both are None).
OMEGA_MIN = -15.0 / time_scale
OMEGA_MAX = 15.0 / time_scale

# Numerical integration settings. Also nondimensionalize the absolute tolerance for omega_tilde.
METHOD = "DOP853"
RTOL = 1.0e-9
ATOL_THETA = 1.0e-11
ATOL_OMEGA = 1.0e-10 / time_scale


# ============================================================
# Output
# ============================================================

output_dir = (
    Path(__file__).resolve().parent
    / "results_manifold_nondimensional"
    / f"sigma_{sigma_tilde:.6f}_p_air_{p_air_tilde:.6f}_Gamma_{Gamma_tilde:.6e}"
    / f"alpha_{alpha:.3f}_beta_{beta:.3f}_tau_tilde_{tau_tilde:.6f}"
)

pdf_path = output_dir / f"manifolds_DeltaT_tilde_{DeltaT_tilde_plot:+.8f}.pdf"


# ============================================================
# Dimensionless model equations
# ============================================================

def volume_tilde(theta):
    """V_tilde(theta) = V(theta) / (s_displacer * r)."""
    return 2.0 + sigma_tilde * (1.0 - np.cos(theta))


def gas_temperature_tilde(theta, omega_tilde, DeltaT_tilde):
    """T_tilde = T / T_top; the delay phase is omega_tilde * tau_tilde."""
    return (
        1.0
        + 0.5
        * (1.0 + alpha * np.sin(theta - omega_tilde * tau_tilde))
        * DeltaT_tilde
    )


def pressure_tilde(theta, omega_tilde, DeltaT_tilde):
    """p_tilde = p * s_displacer * r / (n * R * T_top)."""
    return (
        beta
        * gas_temperature_tilde(theta, omega_tilde, DeltaT_tilde)
        / volume_tilde(theta)
    )


def rhs(t_tilde, y, DeltaT_tilde):
    """Derivatives with respect to t_tilde for y = [theta, omega_tilde]."""
    theta, omega_tilde = y
    dtheta = omega_tilde
    domega_tilde = (
        sigma_tilde
        * (pressure_tilde(theta, omega_tilde, DeltaT_tilde) - p_air_tilde)
        * np.sin(theta)
        - Gamma_tilde * omega_tilde
    )
    return np.array([dtheta, domega_tilde], dtype=float)


# ============================================================
# Numerical Jacobian
# ============================================================

def numerical_jacobian(
    y_fixed,
    DeltaT_tilde,
    h_theta=1.0e-6,
    h_omega_tilde=1.0e-8 / time_scale,
):
    """
    Compute the Jacobian of rhs using central differences.

    J_ij = partial f_i / partial y_j
    """
    y_fixed = np.asarray(y_fixed, dtype=float)
    steps = np.array([h_theta, h_omega_tilde], dtype=float)

    J = np.zeros((2, 2), dtype=float)

    for j in range(2):
        dy = np.zeros(2, dtype=float)
        dy[j] = steps[j]

        f_plus = rhs(
            0.0, # The system is autonomous, so the value of t does not matter.
            y_fixed + dy,
            DeltaT_tilde,
        )

        f_minus = rhs(
            0.0,
            y_fixed - dy,
            DeltaT_tilde,
        )

        J[:, j] = (
            f_plus - f_minus
        ) / (2.0 * steps[j])

    return J


def normalize_eigenvector_by_theta(eigenvector):
    """
    Normalize the eigenvector so that its theta component is 1.

    Normalize by the theta component even after nondimensionalization
    so that the displacement in the theta direction can be specified directly.
    Discard any imaginary components caused by numerical error.
    """
    print(eigenvector, "eigenvector")
    eigenvector = np.real(
        np.asarray(eigenvector)
    )

    if abs(eigenvector[0]) < 1.0e-14:
        raise ValueError(
            "The theta component of the eigenvector is too small "
            "to normalize by the theta component."
        )

    return eigenvector / eigenvector[0]


# ============================================================
# Angle processing for plotting
# ============================================================

def wrap_theta(theta):
    """
    Wrap theta into [-pi, pi).
    """
    theta = np.asarray(theta, dtype=float)

    return (
        (theta + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


def break_at_periodic_boundary(theta_wrapped, omega_tilde):
    """
    Insert NaN where the trajectory crosses the boundary between theta = pi and theta = -pi.

    This prevents Matplotlib from drawing spurious horizontal lines
    from the right edge to the left edge.
    """
    theta_wrapped = np.asarray(
        theta_wrapped,
        dtype=float,
    ).copy()

    omega_tilde = np.asarray(
        omega_tilde,
        dtype=float,
    ).copy()

    jumps = np.where(
        np.abs(np.diff(theta_wrapped)) > np.pi
    )[0]

    theta_wrapped[jumps + 1] = np.nan
    omega_tilde[jumps + 1] = np.nan

    return theta_wrapped, omega_tilde


# ============================================================
# Stable and unstable manifolds
# ============================================================

def calculate_manifolds(
    DeltaT_tilde,
    saddle_point=np.array([np.pi, 0.0]),
    delta_theta=1.0e-7,
    t_end=MANIFOLD_T_END,
    n_eval=20000,
):
    """
    Compute the stable and unstable manifolds of a saddle in the dimensionless system.
    t_end is dimensionless time; the rows of each branch are theta and omega_tilde.

    Unstable manifold:
        Integrate the original system forward from a displacement along the unstable eigenvector.

    Stable manifold:
        Integrate the time-reversed system forward from a displacement along the stable eigenvector.
    """
    saddle_point = np.asarray(
        saddle_point, #(pi, 0)
        dtype=float,
    )

    J = numerical_jacobian(
        saddle_point,
        DeltaT_tilde,
    )

    eigenvalues, eigenvectors = np.linalg.eig(J)

    real_parts = np.real(eigenvalues) # Take the real parts, since numerical calculations may produce tiny imaginary components.

    unstable_indices = np.where(
        real_parts > 0.0
    )[0]

    stable_indices = np.where(
        real_parts < 0.0
    )[0]

    if (
        len(unstable_indices) != 1
        or len(stable_indices) != 1
    ):
        raise ValueError(
            "The specified fixed point is not a saddle.\n"
            f"DeltaT_tilde = {DeltaT_tilde}\n"
            f"fixed point = {saddle_point}\n"
            f"eigenvalues = {eigenvalues}"
        )

    unstable_index = unstable_indices[0]
    stable_index = stable_indices[0]

    lambda_u = float(
        np.real(eigenvalues[unstable_index])
    )

    lambda_s = float(
        np.real(eigenvalues[stable_index])
    )

    e_u = normalize_eigenvector_by_theta(
        eigenvectors[:, unstable_index]
    )

    e_s = normalize_eigenvector_by_theta(
        eigenvectors[:, stable_index]
    )

    print("\n========================================")
    print("Saddle-point analysis")
    print("========================================")
    print(f"DeltaT_tilde = {DeltaT_tilde:+.8f}")
    print(f"saddle point = {saddle_point}")

    print("\nJacobian:")
    print(J)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nUnstable eigenvalue (dimensionless):")
    print(lambda_u)

    print("Stable eigenvalue (dimensionless):")
    print(lambda_s)

    print("\nUnstable eigenvector (theta component = 1):")
    print(e_u)

    print("Stable eigenvector (theta component = 1):")
    print(e_s)

    t_eval = np.linspace(
        0.0,
        t_end,
        n_eval,
    )

    stable_branches = []
    unstable_branches = []

    # Use an array to specify separate error tolerances for theta and omega_tilde.
    atol = np.array( 
        [ATOL_THETA, ATOL_OMEGA],
        dtype=float,
    )

    for sign in (-1.0, 1.0):

        # ----------------------------------------
        # Unstable manifold:
        # original system, forward integration
        # ----------------------------------------
        # Apply a small displacement in the theta direction and integrate trajectories away from the saddle.
        y0_u = ( 
            saddle_point
            + sign * delta_theta * e_u
        )

        sol_u = solve_ivp(
            fun=lambda t, y: rhs(
                t,
                y,
                DeltaT_tilde,
            ),
            t_span=(0.0, t_end),
            y0=y0_u,
            t_eval=t_eval,
            method=METHOD,
            rtol=RTOL,
            atol=atol,
        )

        if not sol_u.success:
            print(
                "[WARN] Integration of the unstable manifold failed:",
                sol_u.message,
            )

        unstable_branches.append(sol_u.y)

        # ----------------------------------------
        # Stable manifold:
        # reverse-time system, forward integration
        # ----------------------------------------
        y0_s = (
            saddle_point
            + sign * delta_theta * e_s
        )
        # Compute the stable manifold in reverse time.
        sol_s = solve_ivp(
            fun=lambda t, y: -rhs(
                t,
                y,
                DeltaT_tilde,
            ),
            t_span=(0.0, t_end),
            y0=y0_s,
            t_eval=t_eval,
            method=METHOD,
            rtol=RTOL,
            atol=atol,
        )

        if not sol_s.success:
            print(
                "[WARN] Integration of the stable manifold failed:",
                sol_s.message,
            )

        stable_branches.append(sol_s.y)

    return {
        "stable_branches": stable_branches,
        "unstable_branches": unstable_branches,
        "jacobian": J,
        "stable_eigenvalue": lambda_s,
        "unstable_eigenvalue": lambda_u,
        "stable_eigenvector": e_s,
        "unstable_eigenvector": e_u,
    }


# ============================================================
# Long-time orbit / limit cycle
# ============================================================

def calculate_limit_cycle(
    DeltaT_tilde,
    t_end=LIMIT_CYCLE_T_END,
    plot_window=LIMIT_CYCLE_PLOT_WINDOW,
    n_eval=20000,
):
    """
    Integrate over a long time from an initial angular velocity chosen
    for the rotation direction, and return the final segment of the trajectory.
    t_end, plot_window, and the returned time are dimensionless times.
    """
    if DeltaT_tilde > 0.0:
        omega0_tilde = INITIAL_OMEGA_MAGNITUDE
    elif DeltaT_tilde < 0.0:
        omega0_tilde = -INITIAL_OMEGA_MAGNITUDE
    else:
        omega0_tilde = INITIAL_OMEGA_MAGNITUDE

    y0 = np.array(
        [0.0, omega0_tilde],
        dtype=float,
    )

    atol = np.array(
        [ATOL_THETA, ATOL_OMEGA],
        dtype=float,
    )

    sol = solve_ivp(
        fun=lambda t, y: rhs(
            t,
            y,
            DeltaT_tilde,
        ),
        t_span=(0.0, t_end),
        y0=y0,
        method=METHOD,
        dense_output=True,
        rtol=RTOL,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(
            "Integration of the long-time trajectory failed: "
            + sol.message
        )

    t_start = max(
        0.0,
        t_end - plot_window,
    )

    t_plot = np.linspace(
        t_start,
        t_end,
        n_eval,
    )

    y_plot = sol.sol(t_plot)

    theta = y_plot[0]
    omega_tilde = y_plot[1]

    mean_omega_tilde = np.mean(omega_tilde)

    # Divide the change in theta by 2π to calculate the number of revolutions.
    rotation_count = (
        theta[-1] - theta[0]
    ) / (
        2.0 * np.pi
    )

    print("\n========================================")
    print("Long-time orbit")
    print("========================================")
    print(f"mean omega_tilde = {mean_omega_tilde:+.8f}")
    print(
        "phase advance in plotting window = "
        f"{rotation_count:+.6f} rotations"
    )
    print(
        "omega_tilde range = "
        f"[{np.min(omega_tilde):+.8f}, {np.max(omega_tilde):+.8f}]"
    )

    if abs(rotation_count) < 0.5:
        print(
            "The trajectory completed less than one revolution near the end of integration. "
            "It may be converging to a fixed point."
        )
    else:
        print(
            "A rotating long-time trajectory was obtained."
        )

    return {
        "time": t_plot,
        "theta": theta,
        "omega_tilde": omega_tilde,
        "mean_omega_tilde": mean_omega_tilde,
        "rotation_count": rotation_count,
    }

# ============================================================
# Plot helpers
# ============================================================

def plot_phase_curve(
    ax,
    theta,
    omega_tilde,
    **plot_kwargs,
):
    """
    Wrap theta into [-pi, pi) and plot in the theta/pi - omega_tilde plane,
    breaking the line at periodic boundaries.

    Returns
    -------
    line : matplotlib.lines.Line2D
        Handle of the plotted line.
    """
    theta_wrapped = wrap_theta(theta)

    theta_plot, omega_plot_tilde = break_at_periodic_boundary(
        theta_wrapped,
        omega_tilde,
    )

    line, = ax.plot(
        theta_plot / np.pi,
        omega_plot_tilde,
        **plot_kwargs,
    )

    return line


# ============================================================
# Main
# ============================================================

def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    manifold_result = calculate_manifolds(
        DeltaT_tilde=DeltaT_tilde_plot,
        saddle_point=np.array(
            [np.pi, 0.0],
            dtype=float,
        ),
        delta_theta=MANIFOLD_DELTA_THETA,
        t_end=MANIFOLD_T_END,
        n_eval=MANIFOLD_N_EVAL,
    )

    limit_cycle_result = None

    if PLOT_LIMIT_CYCLE:
        limit_cycle_result = calculate_limit_cycle(
            DeltaT_tilde=DeltaT_tilde_plot,
            t_end=LIMIT_CYCLE_T_END,
            plot_window=LIMIT_CYCLE_PLOT_WINDOW,
            n_eval=LIMIT_CYCLE_N_EVAL,
        )

    fig, ax = plt.subplots(
        figsize=(5.0, 5.0)
    )

    # Line handles for the legend.
    unstable_line = None
    stable_line = None
    limit_cycle_line = None

    # ----------------------------------------
    # Unstable manifolds
    # ----------------------------------------
    for i, branch in enumerate(
        manifold_result["unstable_branches"] # There are two branches: a trajectory approaching the saddle and one departing from it.
    ):
        line = plot_phase_curve(
            ax=ax,
            theta=branch[0],
            omega_tilde=branch[1],
            color="tab:red",
            linewidth=2.0,
            linestyle=(0, (6, 3)),
            zorder=2,
        )

        # Use only the first branch in the legend.
        if i == 0:
            unstable_line = line

    # ----------------------------------------
    # Stable manifolds
    # ----------------------------------------
    for i, branch in enumerate(
        manifold_result["stable_branches"]
    ):
        line = plot_phase_curve(
            ax=ax,
            theta=branch[0],
            omega_tilde=branch[1],
            color="tab:blue",
            linewidth=2.0,
            linestyle="-",
            zorder=1,
        )

        # Use only the first branch in the legend.
        if i == 0:
            stable_line = line

    # ----------------------------------------
    # Limit cycle / long-time orbit
    # Plot last so that the manifolds do not obscure it.
    # ----------------------------------------
    if limit_cycle_result is not None:
        theta_lc = limit_cycle_result["theta"]
        omega_lc_tilde = limit_cycle_result["omega_tilde"]

        limit_cycle_line = plot_phase_curve(
            ax=ax,
            theta=theta_lc,
            omega_tilde=omega_lc_tilde,
            color="black",
            linewidth=2.0,
            linestyle=(0, (1, 2)),
            zorder=3,
        )

    # ----------------------------------------
    # Fixed points
    # ----------------------------------------

    # theta = pi and theta = -pi represent
    # the same saddle point due to periodicity.
    saddle_point = ax.scatter(
        [-1.0, 1.0],
        [0.0, 0.0],
        s=100,
        facecolors="white",
        edgecolors="black",
        zorder=10,
    )

    fixed_point = ax.scatter(
        [0.0],
        [0.0],
        s=100,
        facecolors="black",
        edgecolors="black",
        zorder=10,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.7,
        alpha=0.45,
        zorder=0,
    )

    # ----------------------------------------
    # Axis settings
    # ----------------------------------------
    ax.set_xlim(
        -1.0,
        1.0,
    )

    ax.set_ylim(
        OMEGA_MIN,
        OMEGA_MAX,
    )

    ax.set_xticks(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    ax.set_xlabel(
        r"$\theta/\pi$",
        fontsize=20,
    )

    ax.set_ylabel(
        r"$\tilde{\omega}$",
        fontsize=20,
    )

    ax.set_title(
        rf"$\Delta\tilde{{T}}={DeltaT_tilde_plot:+.6f}$",
        fontsize=24,
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        labelsize=16,
    )

    # ----------------------------------------
    # Legends
    # ----------------------------------------

    # Left: trajectory legend.
    line_handles = [
        unstable_line,
        stable_line,
    ]

    line_labels = [
        "unstable manifold",
        "stable manifold",
    ]

    # Add only if the limit cycle was plotted.
    if limit_cycle_line is not None:
        line_handles.append(limit_cycle_line)
        line_labels.append("limit cycle")

    # legend_lines = ax.legend(
    #     handles=line_handles,
    #     labels=line_labels,
    #     loc="lower left",
    #     bbox_to_anchor=(0.015, 0.0),
    #     frameon=False,
    #     fontsize=10,
    #     handlelength=2.8,
    #     handletextpad=0.7,
    #     labelspacing=0.1,
    #     borderaxespad=0.0,
    # )

    # # Keep the first legend so that the second does not replace it.
    # ax.add_artist(legend_lines)

    # # Right: equilibrium point legend.
    # legend_points = ax.legend(
    #     handles=[
    #         saddle_point,
    #         fixed_point,
    #     ],
    #     labels=[
    #         "saddle",
    #         "fixed point",
    #     ],
    #     loc="lower right",
    #     bbox_to_anchor=(0.985, 0.0),
    #     frameon=False,
    #     fontsize=10,
    #     handletextpad=0.6,
    #     labelspacing=0.1,
    #     borderaxespad=0.0,
    #     scatterpoints=1,
    # )

    fig.tight_layout()

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("\n========================================")
    print("Output")
    print("========================================")
    print(f"PDF saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

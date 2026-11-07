import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Parameters
# -----------------------------
V0 = 44900e-9
s = 71e-6
r = 3.5e-3
I = 5.7e-5
P0 = 101300
n = 0.00185
R = 8.314
Gamma = I / 20
T_top = 24 + 273.15

s_displacer = V0 / (2 * r)
sigma_tilde = s/s_displacer
p_air_tilde = s_displacer * P0 * r / (n * R * T_top)

gamma_min=0.00005
gamma_max=0.0005
gamma_step=0.00005

alpha = 0.157
beta = 0.954

gamma_vals = np.arange(gamma_min, gamma_max + gamma_step / 2, gamma_step)

overlay_csv_path = Path("compare_gamma_sigma0.011069_p0.497586_PO101300.0_alpha0.157_beta0.954_tau0.0137.csv") 

# Read the CSV beside this script, independent of the working directory.
experiment_csv_path = Path(__file__).resolve().with_name("experiment_plot_include_L.csv")
experiment_df = pd.read_csv(experiment_csv_path)
no_load_experiment_ids = [
    "20200310-2",
    "20200310-3",
    "20200310-5",
    "20200311-2",
    "20200325-1",
    "20200325-2",
]


def last_delta_t(df, experiment_id):
    """Return the last non-missing dT in CSV row order, regardless of w."""
    column = f"{experiment_id}-dT"
    if column not in df.columns:
        raise ValueError(f"Required column is missing from {experiment_csv_path}: {column}")
    values = pd.to_numeric(df[column], errors="raise").dropna()
    if values.empty:
        raise ValueError(f"No dT measurements in column {column}.")
    value = float(values.iloc[-1])
    if not math.isfinite(value):
        raise ValueError(f"The last dT measurement in column {column} must be finite.")
    return value


# DeltaT_c [K]: average the final dT of the six positive no-load trials;
# use the final dT of each loaded trial for L1, L2, and L4.
# Relaxation times tau [s] remain explicit inputs, with Gamma = I / tau.
def first_stopped_delta_t(df, experiment_id, w_tol=1e-5):
    """Return the temperature difference at the first stopped point in CSV measurement order."""
    dT = pd.to_numeric(df[f"{experiment_id}-dT"], errors="raise")
    w = pd.to_numeric(df[f"{experiment_id}-w"], errors="raise")

    mask = dT.notna() & w.notna() & (w.abs() <= w_tol)
    stopped_dT = dT.loc[mask]

    if stopped_dT.empty:
        raise ValueError(f"{experiment_id}: 停止相当の点がありません")

    return float(stopped_dT.iloc[0])


experimental_data = [
    {
        "condition": "No added load",
        "DeltaT_c": float(np.mean([
            first_stopped_delta_t(experiment_df, experiment_id)
            for experiment_id in no_load_experiment_ids
        ])),
        "Gamma": I / 20.0,
    },
    {"condition": "L1", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded1"), "Gamma": I / 13.2},
    {"condition": "L2", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded2"), "Gamma": I / 9.1},
    {"condition": "L4", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded4"), "Gamma": I / 6.9},
]

# Nondimensionalization: Gamma_tilde = Gamma / sqrt(n R T_top I),
#                       DeltaT_c_tilde = DeltaT_c / T_top (T_top is the absolute temperature [K]).
gamma_scale = math.sqrt(n * R * T_top * I)
experimental_points = [
    (row["condition"], row["Gamma"] / gamma_scale, row["DeltaT_c"] / T_top)
    for row in experimental_data
]
print("Experimental nondimensional values:")
print(pd.DataFrame(
    experimental_points,
    columns=["condition", "Gamma_tilde", "DeltaT_c_tilde"],
).to_string(index=False, float_format=lambda value: f"{value:.8g}"))

# -----------------------------
# Output
# -----------------------------
output_dir = Path("results_pi")
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / (
    f"compare_gamma_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"gamma_{gamma_min:.5f}-{gamma_max:.5f}.png"
)

pdf_path= output_dir / (
    f"compare_gamma_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"gamma_{gamma_min:.5f}-{gamma_max:.5f}.pdf"
)

csv_path = output_dir / (
    f"compare_gamma_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"gamma_{gamma_min:.5f}-{gamma_max:.5f}.csv"
)

# ============================================================
# Numerical solver for one gamma
# ============================================================
def solve_for_gamma(gamma, DeltaT_min=-1.0, DeltaT_max=1.0, n_grid=3001):
    """
    For a fixed gamma, solve
        C |DeltaT| - Gamma_tilde * int omega_h dtheta = 0
    on positive and negative DeltaT sides.
    """

    C = (
        math.pi
        * (math.sqrt(1.0 + sigma_tilde) - 1.0) ** 2
        / sigma_tilde   
        * alpha
        * beta
    )

    def A(DeltaT):
        return beta * (1.0 + DeltaT / 2.0)

    def energy_inside(theta, DeltaT):
        return (
                A(DeltaT)*math.log((1+sigma_tilde*math.sin(theta/2)**2)/(1+sigma_tilde)) + sigma_tilde * p_air_tilde * (1.0 + math.cos(theta))
        )

    def orbit_is_real(DeltaT, n_check=1001):
        if A(DeltaT) <= 0:
            return False

        xs = np.linspace(-math.pi, math.pi, n_check)
        vals = np.array([2.0 * energy_inside(x, DeltaT) for x in xs])

        return np.min(vals) >= 0

    def omega_h(theta, DeltaT):
        val = 2.0 * energy_inside(theta, DeltaT)

        if val < 0:
            return np.nan

        return math.sqrt(val)

    def damping_integral(DeltaT):
        if not orbit_is_real(DeltaT):
            return np.nan

        val, err = quad(
            lambda th: omega_h(th, DeltaT),
            -math.pi,
            math.pi,
            epsabs=1e-10,
            epsrel=1e-10,
        )

        return val

    def rhs(DeltaT):
        integral = damping_integral(DeltaT)
        return gamma * integral

    def equation_abs(DeltaT):
        right = rhs(DeltaT)
        return C * abs(DeltaT) - right

    def find_sign_changes(func, xs):
        brackets = []

        x_prev = xs[0]
        f_prev = func(x_prev)

        for x in xs[1:]:
            f = func(x)

            if np.isfinite(f_prev) and np.isfinite(f):
                if f_prev * f < 0:
                    brackets.append((x_prev, x))

            x_prev = x
            f_prev = f

        return brackets

    def solve_roots(xmin, xmax):
        xs = np.linspace(xmin, xmax, n_grid)
        brackets = find_sign_changes(equation_abs, xs)

        roots = []

        for a, b in brackets:
            root = brentq(
                equation_abs,
                a,
                b,
                xtol=1e-12,
                rtol=1e-12,
            )

            roots.append(root)

        return brackets, roots

    brackets_plus, roots_plus = solve_roots(1e-12, DeltaT_max)
    brackets_minus, roots_minus = solve_roots(DeltaT_min, -1e-12)

    return {
        "gamma": gamma,
        "C": C,
        "brackets_plus": brackets_plus,
        "roots_plus": roots_plus,
        "brackets_minus": brackets_minus,
        "roots_minus": roots_minus,
    }


def select_root(roots, side):
    """
    Select a representative root when multiple candidates are found.
    Positive side: the positive root closest to zero.
    Negative side: the negative root closest to zero.
    """
    if len(roots) == 0:
        return np.nan

    if side == "+":
        candidates = [x for x in roots if x > 0]
        if len(candidates) > 1:
            print("Warning: multiple positive roots found. Selecting the smallest one.")
        return min(candidates) if candidates else np.nan

    if side == "-":
        candidates = [x for x in roots if x < 0]
        if len(candidates) > 1:
            print("Warning: multiple negative roots found. Selecting the largest one.")
        return max(candidates) if candidates else np.nan

    raise ValueError("side must be '+' or '-'.")


# ============================================================
# Numerical solutions for all gamma
# ============================================================
roots_plus_num = []
roots_minus_num = []

for gamma in gamma_vals:
    result = solve_for_gamma(gamma)

    root_plus = select_root(result["roots_plus"], "+")
    root_minus = select_root(result["roots_minus"], "-")

    roots_plus_num.append(root_plus)
    roots_minus_num.append(root_minus)

roots_plus_num = np.array(roots_plus_num)
roots_minus_num = np.array(roots_minus_num)

# ============================================================
# Analytical approximation
# ============================================================
radicand = (
    16 * gamma_vals**2
    + math.pi**2 * sigma_tilde * alpha**2 * (p_air_tilde - beta / 2)
)

valid= radicand >= 0

sqrt_valid = np.sqrt(radicand[valid])
gamma_valid = gamma_vals[valid]

denom_valid = (
    math.pi**2
    * sigma_tilde
    * alpha**2
    * beta
)

deltaT_plus_approx = (
    32
    * gamma_vals
    / denom_valid
    * (-4 * gamma_valid + sqrt_valid)
)

deltaT_minus_approx = (
    32
    * gamma_vals
    / denom_valid
    * (-4 * gamma_valid - sqrt_valid)
)

# ============================================================
# Save csv
# ============================================================

data = np.column_stack(
    [
        gamma_vals,
        roots_plus_num,
        roots_minus_num,
        deltaT_plus_approx,
        deltaT_minus_approx,
    ]
)

np.savetxt(
    csv_path,
    data,
    delimiter=",",
    header=(
        "gamma,"
        "DeltaT_c_plus_numerical,DeltaT_c_minus_numerical,"
        "DeltaT_c_plus_approx,DeltaT_c_minus_approx"
    ),
    comments="",
)

# ============================================================
# Plot
# ============================================================
x_num = 1e4 * gamma_vals
x_apx = 1e4 * gamma_valid

plt.figure(figsize=(8, 6))

# numerical
plt.plot(
    x_num,
    roots_plus_num,
    "-o",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,+}$",
    color="red"
)

plt.plot(
    x_num,
    roots_minus_num,
    "-s",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,-}$",
    color="blue"
)

# analytical approximation
plt.plot(
    x_apx,
    deltaT_plus_approx,
    "--o",
    markersize=3,
    label=r"Approx. $\Delta \tilde{T}_{c,+}$",
    color="orange"
)

plt.plot(
    x_apx,
    deltaT_minus_approx,
    "--s",
    markersize=3,
    label=r"Approx. $\Delta \tilde{T}_{c,-}$",
    color="green"
)


# --- overlay csv results ---
if overlay_csv_path.exists():
    df = pd.read_csv(overlay_csv_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    required_cols = {"gamma", "T_minus", "T_plus"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing from the CSV: {missing}")

    gamma_csv = df["gamma"].to_numpy(dtype=float)
    T_minus_csv = df["T_minus"].to_numpy(dtype=float)
    T_plus_csv = df["T_plus"].to_numpy(dtype=float)

    gamma_csv = 1e4*gamma_csv

    plt.plot(
        gamma_csv,
        T_plus_csv,
        marker="x",
        linestyle="None",
        label=r"Determined $\Delta\tilde{T}_{c,+}$",
        markersize=10,
        color="purple"
    )

    plt.plot(
        gamma_csv,
        T_minus_csv,
        marker="^",
        linestyle="None",
        label=r"Determined $\Delta\tilde{T}_{c,-}$",
        markersize=10,
        color="black"
    )

else:
    print("overlay csv not found:", overlay_csv_path)

# Overlay the nondimensional experimental values calculated from the inputs,
# scaling them to match the horizontal axis, 10^4 Gamma_tilde.
plt.scatter(
    [1e4 * gamma for _, gamma, _ in experimental_points],
    [deltaT for _, _, deltaT in experimental_points],
    marker="D",
    s=65,
    color="deepskyblue",
    edgecolors="black",
    linewidths=0.8,
    zorder=5,
    label=r"Experimental $\Delta\tilde{T}_c$",
)
plt.margins(x=0.08, y=0.12)


plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel(
    r"$10^4 \tilde{\Gamma}$",fontsize=40
)
plt.ylabel(r"$\Delta \tilde{T}_c$",fontsize=40)
plt.tick_params(axis="both", labelsize=20)
# plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path, dpi=300)
plt.show()

print("saved:", png_path)
print("saved:", csv_path)

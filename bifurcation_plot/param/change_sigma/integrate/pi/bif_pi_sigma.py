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

p_air_tilde = s_displacer * P0 * r / (n * R * T_top)
Gamma_tilde = Gamma / math.sqrt(n * R * T_top * I)

alpha = 0.157
beta = 0.954

overlay_csv_path = Path("../../compare_sigma_p0.497586_Gamma0.000177_PO101300.0_alpha0.157_beta0.954_tau0.0137.csv") 

# -----------------------------
# sigma range
# -----------------------------
sigma_min = 0.01
sigma_max = 0.05
sigma_step = 0.005

sigma_vals = np.arange(sigma_min, sigma_max + sigma_step/2, sigma_step)

# -----------------------------
# Output
# -----------------------------
output_dir = Path("results_pi")
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / (
    f"compare_sigma_"
    f"Gamma_{Gamma_tilde:.6f}_"
    f"p_{p_air_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"sigma_{sigma_min:.3f}-{sigma_max:.3f}.png"
)

pdf_path= output_dir / (
    f"compare_sigma_"
    f"Gamma_{Gamma_tilde:.6f}_"
    f"p_{p_air_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"sigma_{sigma_min:.3f}-{sigma_max:.3f}.pdf"
)

csv_path = output_dir / (
    f"compare_sigma_"
    f"Gamma_{Gamma_tilde:.6f}_"
    f"p_{p_air_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}_"
    f"sigma_{sigma_min:.3f}-{sigma_max:.3f}.csv"
)

# ============================================================
# Numerical solver for one sigma
# ============================================================
def solve_for_sigma(sigma, DeltaT_min=-1.0, DeltaT_max=1.0, n_grid=3001):
    """
    For a fixed sigma, solve
        C |DeltaT| - Gamma_tilde * int omega_h dtheta = 0
    on positive and negative DeltaT sides.
    """

    C = (
        math.pi
        * (math.sqrt(1.0 + sigma) - 1.0) ** 2
        / sigma
        * alpha
        * beta
    )

    def A(DeltaT):
        return beta * (1.0 + DeltaT / 2.0)

    def energy_inside(theta, DeltaT):
        return (
                A(DeltaT)*math.log((1+sigma*math.sin(theta/2)**2)/(1+sigma)) + sigma * p_air_tilde * (1.0 + math.cos(theta))
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
        if not np.isfinite(integral):
            return np.nan
        return Gamma_tilde * integral

    def equation_abs(DeltaT):
        right = rhs(DeltaT)
        if not np.isfinite(right):
            return np.nan
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
        "sigma": sigma,
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
# Numerical solutions for all sigma
# ============================================================
roots_plus_num = []
roots_minus_num = []

for sigma in sigma_vals:
    result = solve_for_sigma(sigma)

    print("sigma =", sigma)
    print("roots_plus =", result["roots_plus"])
    print("roots_minus =", result["roots_minus"])

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
    16 * Gamma_tilde**2
    + math.pi**2 * sigma_vals * alpha**2 * (p_air_tilde - beta / 2)
)

valid= radicand >= 0
sqrt_valid = np.sqrt(radicand[valid])
sigma_valid = sigma_vals[valid]

denom_valid = (
    math.pi**2
    * sigma_valid
    * alpha**2
    * beta
)

deltaT_plus_approx = (
    32
    * Gamma_tilde
    / denom_valid
    * (-4 * Gamma_tilde + sqrt_valid)
)

deltaT_minus_approx = (
    32
    * Gamma_tilde
    / denom_valid
    * (-4 * Gamma_tilde - sqrt_valid)
)

# ============================================================
# Save csv
# ============================================================

data = np.column_stack(
    [
        sigma_vals,
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
        "sigma,"
        "DeltaT_c_plus_numerical,DeltaT_c_minus_numerical,"
        "DeltaT_c_plus_approx,DeltaT_c_minus_approx"
    ),
    comments="",
)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 6))

# numerical
plt.plot(
    sigma_vals,
    roots_plus_num,
    "-o",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,+}$",
    color="red"
)

plt.plot(
    sigma_vals,
    roots_minus_num,
    "-s",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,-}$",
    color="blue"
)

# analytical approximation
plt.plot(
    sigma_valid,
    deltaT_plus_approx,
    "--o",
    markersize=3,
    label=r"Approx. $\Delta \tilde{T}_{c,+}$",
    color="orange"
)

plt.plot(
    sigma_valid,
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

    required_cols = {"sigma", "T_minus", "T_plus"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing from the CSV: {missing}")

    sigma_csv = df["sigma"].to_numpy(dtype=float)
    T_minus_csv = df["T_minus"].to_numpy(dtype=float)
    T_plus_csv = df["T_plus"].to_numpy(dtype=float)


    plt.plot(
        sigma_csv,
        T_plus_csv,
        marker="x",
        linestyle="None",
        label=r"Determined $\Delta\tilde{T}_{c,+}$",
        markersize=10,
        color="purple"
    )

    plt.plot(
        sigma_csv,
        T_minus_csv,
        marker="^",
        linestyle="None",
        label=r"Determined $\Delta\tilde{T}_{c,-}$",
        markersize=10,
        color="black"
    )

else:
    print("overlay csv not found:", overlay_csv_path)


plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel(
    rf"$\tilde{{\sigma}}$ ",fontsize=40
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
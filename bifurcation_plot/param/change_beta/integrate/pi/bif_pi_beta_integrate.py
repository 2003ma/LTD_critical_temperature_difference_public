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
Gamma_tilde = Gamma / math.sqrt(n * R * T_top * I)

alpha = 0.157
# beta = 0.948

beta_min=0.9
beta_max=0.98
beta_step=0.01

beta_vals = np.arange(beta_min, beta_max + beta_step / 2, beta_step)

overlay_csv_path = Path("compare_beta_sigma0.011069_p0.497586_gamma_0.000177_PO101300.0_alpha_0.157_tau_0.0137.csv") 

# -----------------------------
# Output
# -----------------------------
output_dir = Path("results_pi")
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / (
    f"compare_beta_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"gamma_{Gamma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta_min:.3f}-{beta_max:.3f}.png"
)

pdf_path = output_dir / (
    f"compare_beta_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"gamma_{Gamma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta_min:.3f}-{beta_max:.3f}.pdf"
)

csv_path = output_dir / (
    f"compare_beta_"
    f"p_{p_air_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"gamma_{Gamma_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta_min:.3f}-{beta_max:.3f}.csv"
)

# ============================================================
# Numerical solver for one beta
# ============================================================
def solve_for_beta(beta, DeltaT_min=-1.0, DeltaT_max=1.0, n_grid=3001):
    """
    For a fixed beta, solve
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

        return np.min(vals) >=0

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
        return Gamma_tilde * integral

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
        "beta": beta,
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

for beta in beta_vals:
    result = solve_for_beta(beta)
    print(f"beta={beta:.3f}: roots_plus={result['roots_plus']}, roots_minus={result['roots_minus']}")


    root_plus = select_root(result["roots_plus"], "+")
    root_minus = select_root(result["roots_minus"], "-")

    roots_plus_num.append(root_plus)
    roots_minus_num.append(root_minus)

roots_plus_num = np.array(roots_plus_num)
roots_minus_num = np.array(roots_minus_num)

# ============================================================
# Analytical approximation
# ============================================================

radicand_all = (
    16 * Gamma_tilde**2
    + math.pi**2 * sigma_tilde * alpha**2 * (p_air_tilde - beta_vals / 2)
)

valid = radicand_all >= 0

deltaT_plus_approx = np.full_like(beta_vals, np.nan, dtype=float)
deltaT_minus_approx = np.full_like(beta_vals, np.nan, dtype=float)

sqrt_valid = np.sqrt(radicand_all[valid])
beta_valid = beta_vals[valid]


denom_valid = (
    math.pi**2
    * sigma_tilde
    * alpha**2
    * beta_valid
)

deltaT_plus_approx[valid] = (
    32
    * Gamma_tilde
    / denom_valid
    * (-4 * Gamma_tilde + sqrt_valid)
)

deltaT_minus_approx[valid] = (
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
        beta_vals,
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
        "beta,"
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
    beta_vals,
    roots_plus_num,
    "-o",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,+}$",
    color="red"
)

plt.plot(
    beta_vals,
    roots_minus_num,
    "-s",
    markersize=3,
    label=r"Numerical $\Delta \tilde{T}_{c,-}$",
    color="blue"
)

# analytical approximation
plt.plot(
    beta_valid,
    deltaT_plus_approx,
    "--o",
    markersize=3,
    label=r"Approx. $\Delta \tilde{T}_{c,+}$",
    color="orange"
)

plt.plot(
    beta_valid,
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

    required_cols = {"beta", "T_minus", "T_plus"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing from the CSV: {missing}")

    beta_csv = df["beta"].to_numpy(dtype=float)
    T_minus_csv = df["T_minus"].to_numpy(dtype=float)
    T_plus_csv = df["T_plus"].to_numpy(dtype=float)


    plt.plot(
        beta_csv,
        T_plus_csv,
        marker="x",
        linestyle="None",
        label=r"Determined $\Delta\tilde{T}_{c,+}$",
        markersize=10,
        color="purple"
    )

    plt.plot(
        beta_csv,
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
    rf"${{\beta}}$ ",fontsize=40
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
print("saved:", pdf_path)
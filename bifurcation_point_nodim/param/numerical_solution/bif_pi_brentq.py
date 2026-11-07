import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from pathlib import Path
from matplotlib import pyplot as plt

# -----------------------------
# Parameters (notebook exact)
# -----------------------------
V0 = 44900e-9
s = 71e-6  # Power piston area
r = 3.5e-3  # Crank length
I = 5.7e-5
P0 = 101300
n = 0.00185
R = 8.314
Gamma = I / 20
T_top = 24 + 273.15
s_displacer = V0 / (2 * r)  # Displacer piston area
sigma_tilde = s/s_displacer
p_air_tilde = s_displacer *P0*r / (n * R * T_top)
Gamma_tilde = Gamma/math.sqrt(n*R*T_top*I)

alpha = 0.157
beta = 0.954
# -----------------------------
# Constant on the driving side
# -----------------------------
C = math.pi * (math.sqrt(1.0 + sigma_tilde) - 1.0) ** 2 / sigma_tilde*alpha*beta


def A(DeltaT):
    return (1.0 + DeltaT / 2.0)*beta


def energy_inside(theta, DeltaT):
    return (
        A(DeltaT)*math.log((1+sigma_tilde*math.sin(theta/2)**2)/(1+sigma_tilde)) + sigma_tilde * p_air_tilde * (1.0 + math.cos(theta))
    )


def orbit_is_real(DeltaT, n_grid=3001):
    if A(DeltaT) <= 0:
        return False

    xs = np.linspace(-math.pi, math.pi, n_grid)
    # Evaluate multiple points to check whether all values are nonnegative.
    vals = np.array([2.0 * energy_inside(x, DeltaT) for x in xs])

    # Treat all values as nonnegative if the minimum is at least zero (subject to roundoff error).
    return np.min(vals) >= 0


def omega_h(theta, DeltaT):
    val = 2.0 * energy_inside(theta, DeltaT)

    # A DeltaT that produces NaN cannot be integrated and is not a solution (a precaution already covered by orbit_is_real).
    if val < 0:
        return np.nan

    # Small negative values can appear from floating-point roundoff.
    return math.sqrt(val)


def damping_integral(DeltaT):
    if not orbit_is_real(DeltaT): 
        return np.nan

    # Integrate with respect to theta.
    val, err = quad(
        lambda th: omega_h(th, DeltaT),
        -math.pi,
        math.pi,
        epsabs=1e-12,
        epsrel=1e-12,
    )

    return val


def rhs(DeltaT):
    I = damping_integral(DeltaT)


    return Gamma_tilde * I


def equation_abs(DeltaT):
    r = rhs(DeltaT)


    return C * abs(DeltaT) - r


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


def solve_roots(xmin, xmax, n_grid=5000):
    xs = np.linspace(xmin, xmax, n_grid)
    brackets = find_sign_changes(equation_abs, xs) # Find intervals containing roots.

    roots = [] # Use brentq to find and store roots in the identified intervals.

    for a, b in brackets:
        root = brentq(
            equation_abs,
            a,
            b,
            xtol=1e-14,
            rtol=1e-14,
        )

        roots.append(root)

    return brackets, roots

# ============================================================
# Solve
# ============================================================



output_dir = Path("results_pi")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
    f"Gamma_{Gamma_tilde:.6f}_"
    f"sigma_{sigma_tilde:.6f}_"
    f"p_{p_air_tilde:.6f}_"
    f"alpha_{alpha:.3f}_"
    f"beta_{beta:.3f}.txt"
)

report_lines = []


def log(*args):
    """
    Print the text and also store it in report_lines.
    """
    text = " ".join(str(a) for a in args)
    print(text)
    report_lines.append(text)


log("C =", C)

# First search for sign changes on a grid to locate approximate root intervals.
brackets_plus, roots_plus = solve_roots(1e-12, 1.0, n_grid=5000)
brackets_minus, roots_minus = solve_roots(-1.0, -1e-12, n_grid=5000)

log()
log("parameters")
log("Gamma_tilde =", Gamma_tilde)
log("sigma_tilde =", sigma_tilde)
log("p_air_tilde =", p_air_tilde)
log("alpha =", alpha)
log("beta =", beta)

log()
log("brackets_plus")
for br in brackets_plus:
    log(br)

log()
log("brackets_minus")
for br in brackets_minus:
    log(br)

log()
log("Dimensionless results")

for i, root in enumerate(roots_plus):
    log(f"DeltaT_plus_{i+1} =", root)

for i, root in enumerate(roots_minus):
    log(f"DeltaT_minus_{i+1} =", root)

log()
log("Check")

for root in roots_plus:
    log("equation_abs(DeltaT_plus)  =", equation_abs(root))

for root in roots_minus:
    log("equation_abs(DeltaT_minus) =", equation_abs(root))

# Save to a text file.
output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print()
print("Saved to:", output_path)

# Plot the function to inspect the intervals containing roots.
xs = np.linspace(-1,2,400)

ys = [equation_abs(x) for x in xs]

plt.plot(xs, ys)
plt.axhline(0,color='k')
plt.savefig(output_dir / "equation_abs.png", dpi=300)

plt.show()
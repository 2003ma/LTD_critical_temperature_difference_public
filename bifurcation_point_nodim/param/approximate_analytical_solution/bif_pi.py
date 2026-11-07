import math
from pathlib import Path
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

output_dir = Path("results_pi")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
    f"Gamma{Gamma_tilde:.8f}_"
    f"sigma{sigma_tilde:.6f}_"
    f"p{p_air_tilde:.6f}_"
    f"alpha{alpha:.3f}_"
    f"beta{beta:.3f}.txt"
)

report_lines = []


def log(*args):
    """
    Print the text and also store it in report_lines.
    """
    text = " ".join(str(a) for a in args)
    print(text)
    report_lines.append(text)



deltaT_c_dash_plus = (
    32
    * Gamma_tilde
    / (math.pi**2 * sigma_tilde*alpha**2*beta)
    * (
        -4 * Gamma_tilde
        + math.sqrt(
            16 * Gamma_tilde**2 + math.pi**2 * sigma_tilde *alpha**2 *(p_air_tilde - 1 / 2*beta)
        )
    )
)
deltaT_c_dash_minus = (
    32
    * Gamma_tilde
    / (math.pi**2 * sigma_tilde*alpha**2*beta)
    * (
        -4 * Gamma_tilde
        - math.sqrt(
            16 * Gamma_tilde**2 + math.pi**2 * sigma_tilde *alpha**2 *(p_air_tilde - 1 / 2*beta)
        )
    )
)


# -----------------------------
# Print and save
# -----------------------------
log(f"alpha: {alpha:.3f}")
log(f"beta: {beta:.3f}")
log(f"Gamma_tilde: {Gamma_tilde:.8f}")
log(f"sigma_tilde: {sigma_tilde:.6f}")
log(f"p_air_tilde: {p_air_tilde:.6f}")
log(f"deltaT_c_tilde_plus: {deltaT_c_dash_plus:.6f}")
log(f"deltaT_c_tilde_minus: {deltaT_c_dash_minus:.6f}")

output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print()
print("Saved to:", output_path)
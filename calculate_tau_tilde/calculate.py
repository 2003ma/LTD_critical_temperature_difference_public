import math
from pathlib import Path


# ============================================================
# Reference scales and dimensionless model parameters
# ============================================================

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

print("tau_tilde=", tau_tilde)
Path(__file__).with_name("tau_tilde.txt").write_text(f"{tau_tilde}\n", encoding="utf-8")

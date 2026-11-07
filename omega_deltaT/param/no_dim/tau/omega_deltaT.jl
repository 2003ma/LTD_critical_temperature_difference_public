using DifferentialEquations
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics

# ============================================================
# Reproduction of Izumida 2018 EPL Fig.5
# <omega_tilde> - DeltaT_tilde diagram
# ============================================================
# Round at the fourth significant digit and use three significant digits.
alpha = 0.157
beta = 0.954
tau=0.0137

# -----------------------------
# Dimensionless parameters used in the paper
# -----------------------------
V0 = 44900e-9
s = 71e-6          # Power piston area
r = 3.5e-3         # Crank length
I = 5.7e-5
P0 = 101300.0
n = 0.00185
R = 8.314
Gamma = I / 20
T_top = 24 + 273.15

s_displacer = V0 / (2 * r)  # Displacer piston area

sigma_tilde = s / s_displacer
p_air_tilde = s_displacer * P0 * r / (n * R * T_top)
Gamma_tilde = Gamma / sqrt(n * R * T_top * I)
# Gamma_tilde=0.0005
# sigma_tilde=0.05
# p_air_tilde=0.52

time_scale = sqrt(
    n * R * T_top / I
)

tau_tilde = tau * time_scale
print("tau_tilde = $tau_tilde\n") #tau_tilde = 3.8793760556038013

DeltaT_minus = -0.01954
DeltaT_plus = -0.01953
width = 0.000001

n_points = Int(round((DeltaT_plus - DeltaT_minus) / width)) + 1
DeltaT_tilde_values = collect(range(DeltaT_minus, DeltaT_plus; length=n_points))

# -----------------------------
# Output
# -----------------------------
output_dir = @sprintf(
    "results/sigma%.6f_p%.6f_Gamma%.6f_PO%.1f/alpha%.3f_beta%.3f_tau%.4f",
    sigma_tilde,
    p_air_tilde,
    Gamma_tilde,
    P0,
    alpha,
    beta,
    tau
)
mkpath(output_dir)

csv_path = joinpath(
    output_dir,
    @sprintf("plus%.4f_minus%.4f_width%.5f_julia.csv", DeltaT_plus, DeltaT_minus, width)
)

# -----------------------------
# Model equations: Eqs. (14), (15), (16)
# -----------------------------
function V_tilde(theta)
    return 2.0 + sigma_tilde * (1.0 - cos(theta))
end

function Teff_tilde(theta, DeltaT_tilde, omega_tilde)
    return 1.0 + 0.5 * (1.0 + sin(theta-omega_tilde*tau_tilde) * alpha) * DeltaT_tilde
end

function p_tilde(theta, DeltaT_tilde, omega_tilde)
    numerator = Teff_tilde(theta, DeltaT_tilde, omega_tilde)
    denominator = V_tilde(theta)
    return beta * numerator / denominator
end

function rhs!(du, y, DeltaT_tilde, t_tilde)
    theta = y[1]
    omega_tilde = y[2]

    dtheta = omega_tilde

    domega =
        sigma_tilde *
            (p_tilde(theta, DeltaT_tilde, omega_tilde) - p_air_tilde) *
            sin(theta) -
            Gamma_tilde * omega_tilde

    du[1] = dtheta
    du[2] = domega
end

# -----------------------------
# One simulation
# -----------------------------

function simulate_mean_omega(DeltaT_tilde)
    if DeltaT_tilde > 0
        omega0 = 0.5
    elseif DeltaT_tilde < 0
        omega0 = -0.5
    else
        omega0 = 0.0
    end

    theta0 = 0.0

    # Long integration is important near homoclinic bifurcation.
    t_end = 300000.0
    t_start_avg = 0.95 * t_end

    # Only save the averaging window to reduce memory.
    t_eval = range(t_start_avg, t_end; length=50000)

    u0 = [theta0, omega0]
    tspan = (0.0, t_end)

    prob = ODEProblem(rhs!, u0, tspan, DeltaT_tilde)

    sol = solve(
        prob,
        Tsit5();
        saveat=t_eval,
        reltol=1e-11,
        abstol=1e-11
    )

    if string(sol.retcode) != "Success"
        @printf("[WARN] failed at DeltaT_tilde=%g: %s\n", DeltaT_tilde, string(sol.retcode))
        return NaN, NaN, NaN
    end

    t = sol.t

    theta = [u[1] for u in sol.u]
    omega_tilde = [u[2] for u in sol.u]

    omega_first = omega_tilde[1]
    omega_last = omega_tilde[end]

    # Use total phase advance rather than mean(omega).
    # This is stable for rotation frequency.
    mean_omega = mean(omega_tilde)

    return mean_omega, omega_first, omega_last
end

# -----------------------------
# Sweep DeltaT_tilde
# -----------------------------
mean_omega_values = Float64[]
omega_first_values = Float64[]
omega_last_values = Float64[]

for DeltaT_tilde in DeltaT_tilde_values
    mean_omega, omega_first, omega_last = simulate_mean_omega(DeltaT_tilde)

    push!(mean_omega_values, mean_omega)
    push!(omega_first_values, omega_first)
    push!(omega_last_values, omega_last)

    @printf(
        "DeltaT_tilde=%+.5f, mean_omega_tilde=%+.6f, omega_first=%+.6f, omega_last=%+.6f\n",
        DeltaT_tilde,
        mean_omega,
        omega_first,
        omega_last
    )
end

mean_omega_values = collect(mean_omega_values)

# -----------------------------
# Prepare limit-cycle branch
# -----------------------------
# For the stable limit cycle plot, remove fixed-point values.
# This avoids drawing a line through zero.
limit_cycle_omega = copy(mean_omega_values)

for i in eachindex(limit_cycle_omega)
    if abs(limit_cycle_omega[i]) < 1e-9
        limit_cycle_omega[i] = NaN
    end
end

# -----------------------------
# Save CSV
# -----------------------------
data = hcat(
    DeltaT_tilde_values,
    mean_omega_values,
    limit_cycle_omega,
    omega_first_values,
    omega_last_values
)

header = "DeltaT_tilde,mean_omega_tilde,limit_cycle_omega_tilde,omega_first,omega_last"

open(csv_path, "w") do io
    println(io, header)
    writedlm(io, data, ',')
end

println("CSV saved to: $csv_path")
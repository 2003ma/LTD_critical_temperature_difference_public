using DifferentialEquations
using Statistics
using DelimitedFiles
using Printf

# -----------------------------
# Parameters (SI units)
# -----------------------------
V0 = 44900e-9
s = 71e-6
r = 3.5e-3
I = 5.7e-5
P0 = 101300.0
n = 0.00185
R = 8.314
Gamma = I / 20
T_top = 24 + 273.15

alpha = 0.157
beta = 0.954
tau = 0.0137

DeltaT_minus = -6.0
DeltaT_plus = -5.8
width = 0.001

n_points = Int(round((DeltaT_plus - DeltaT_minus) / width)) + 1
DeltaT_values = collect(range(DeltaT_minus, DeltaT_plus; length=n_points))

# -----------------------------
# Output
# -----------------------------
output_dir = @sprintf(
    "results/PO_%.3f/alpha%.3f_beta%.3f_tau%.4f",
    P0,
    alpha,
    beta,
    tau
)

mkpath(output_dir)

csv_path = joinpath(
    output_dir,
    @sprintf(
        "plus%.3f_minus%.3f_width%.3f.csv",
        DeltaT_plus,
        DeltaT_minus,
        width
    )
)

# -----------------------------
# Model
# -----------------------------
function V(theta)
    return V0 + s * r * (1 - cos(theta))
end

function T_field(theta, omega, DeltaT)
    T0 = T_top + DeltaT / 2
    return T0 + alpha * sin(theta - omega * tau) * (DeltaT / 2)
end

function pressure(theta, omega, DeltaT)
    return beta * n * R * T_field(theta, omega, DeltaT) / V(theta)
end

function rhs!(du, u, DeltaT, t)

    theta = u[1]
    omega = u[2]

    du[1] = omega

    torque =
        s * r *
            (pressure(theta, omega, DeltaT) - P0) *
            sin(theta)

    du[2] = (torque - Gamma * omega) / I
end

# -----------------------------
# One simulation
# -----------------------------
function simulate_mean_omega(DeltaT)

    if DeltaT > 0
        omega0 = 20.0
    elseif DeltaT < 0
        omega0 = -20.0
    else
        omega0 = 0.0
    end

    theta0 = 0.1

    t_end = 1000.0
    t_start_avg = 0.95 * t_end

    t_eval = range(t_start_avg, t_end; length=20000)

    u0 = [theta0, omega0]
    tspan = (0.0, t_end)

    prob = ODEProblem(rhs!, u0, tspan, DeltaT)

    sol = solve(
        prob,
        Tsit5();
        saveat=t_eval,
        reltol=1e-9,
        abstol=1e-11,
    )

    if string(sol.retcode) != "Success"
        @printf("[WARN] failed at ΔT=%g\n", DeltaT)
        return NaN, NaN, NaN
    end

    omega = [u[2] for u in sol.u]

    mean_omega = mean(omega)
    omega_first = first(omega)
    omega_last = last(omega)

    return mean_omega, omega_first, omega_last
end

# -----------------------------
# Sweep
# -----------------------------
mean_omega_values = Float64[]
omega_first_values = Float64[]
omega_last_values = Float64[]

for dT in DeltaT_values

    mean_omega, omega_first, omega_last =
        simulate_mean_omega(dT)

    push!(mean_omega_values, mean_omega)
    push!(omega_first_values, omega_first)
    push!(omega_last_values, omega_last)

    @printf(
        "DeltaT=%8.4f, mean=%12.6f, first=%12.6f, last=%12.6f\n",
        dT,
        mean_omega,
        omega_first,
        omega_last,
    )
end

# -----------------------------
# Limit cycle branch
# -----------------------------
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
    DeltaT_values,
    mean_omega_values,
    omega_first_values,
    omega_last_values,
    limit_cycle_omega,
)

header = "DeltaT_C,mean_omega_rad_s,omega_first_rad_s,omega_last_rad_s,limit_cycle_omega_rad_s"

open(csv_path, "w") do io
    println(io, header)
    writedlm(io, data, ',')
end

println("CSV saved to: $csv_path")
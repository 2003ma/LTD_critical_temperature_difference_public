# plot_given_params_2d.jl
#
# Overlay experimental data and model predictions for
#   (1) ΔT versus mean rotation rate
# using the specified alpha, beta, and tau.
#
# No optimization is performed.

using CSV
using DataFrames
using DifferentialEquations
using Statistics
using Printf
using Plots

# =========================================
# 0. Input/output settings
# =========================================
const BASE_DIR = @__DIR__

const FREQ_CSV_PATH = joinpath(BASE_DIR, "experiment_plot.csv")

# Specify the desired parameters here.
const PARAMS = Float64[
    0.157,   # alpha
    0.954,    # beta
    0.0137 # tau
]
const theta0 = 0.0


const OUT_DIR = joinpath(
    BASE_DIR,
    "plots_given_params_2d",
    @sprintf(
        "alpha_%.5g_beta_%.5g_tau_%.5g_allrange",
        PARAMS[1],
        PARAMS[2],
        PARAMS[3]
    ),
)

mkpath(OUT_DIR)

# =========================================
# 1. Fixed parameters
# =========================================
const V0 = 44900e-9
const s = 71e-6
const r = 3.5e-3
const I = 5.7e-5
const P0 = 101300.0
const n_mol = 0.00185
const R = 8.314
const Gamma = I / 20
const T_top = 24 + 273.15

# =========================================
# 2. Computation settings
# =========================================
const T_SPAN_FREQ = (0.0, 400.0)
const N_EVAL_FREQ = 10000

const RTOL = 1e-8
const ATOL = 1e-8

# =========================================
# 3. Load data
# =========================================

function extract_experiment(df, id)
    dT_col = Symbol("$(id)-dT")
    w_col = Symbol("$(id)-w")

    subdf = df[:, [dT_col, w_col]]
    dropmissing!(subdf, [dT_col, w_col])

    out = DataFrame(
        DeltaT=Float64.(subdf[!, dT_col]),
        w_exp=Float64.(subdf[!, w_col]),
        experiment_id=fill(id, nrow(subdf)),
    )
    return out
end

function load_freq_data(path)
    if !isfile(path)
        error("Rotation rate data file not found: $(path)")
    end

    df = CSV.read(path, DataFrame)

    required_cols = [:dT, :w, :w_SD]
    for col in required_cols
        if !(col in propertynames(df))
            error("Column $(col) is missing from $(path). Required columns are dT, w, and w_SD.")
        end
    end

    mean_df = dropmissing(select(df, [:dT, :w, :w_SD]))

    deltaT = Float64.(mean_df.dT)
    w = Float64.(mean_df.w)
    w_sd=Float64.(mean_df.w_SD)

    valid = isfinite.(deltaT) .& isfinite.(w)

    ids = [
        "20200302-1",
        "20200302-2",
        "20200302-3",
        "20200327-1",
        "20200327-2",
        "20200324-2",
        "20200310-2",
        "20200310-3",
        "20200310-5",
        "20200311-2",
        "20200325-1",
        "20200325-2",
    ]

    # Check for required columns.
    for id in ids
        for suffix in ["dT", "w"]
            col = Symbol("$(id)-$(suffix)")
            if !(col in propertynames(df))
                error("Column $(col) is missing from $(path).")
            end
        end
    end

    # Create a DataFrame for each experiment.
    detail_dfs = [extract_experiment(df, id) for id in ids]

    return deltaT[valid], w[valid], w_sd[valid], detail_dfs
end

# =========================================
# 4. 2D model
# =========================================
function V_2d(theta)
    return V0 + s * r * (1 - cos(theta))
end

function T_field_2d(theta, omega, DeltaT, alpha, tau)
    T0 = T_top + DeltaT / 2
    return T0 + alpha * sin(theta - omega * tau) * (DeltaT / 2)
end

function pressure_2d(theta, omega, DeltaT, alpha, beta, tau)
    return beta * n_mol * R * T_field_2d(theta, omega, DeltaT, alpha, tau) / V_2d(theta)
end

function rhs_2d!(du, u, p, t)
    theta, omega = u
    DeltaT, alpha, beta, tau = p

    du[1] = omega

    torque =
        s * r *
            (pressure_2d(theta, omega, DeltaT, alpha, beta, tau) - P0) *
            sin(theta)

    du[2] = (torque - Gamma * omega) / I

    return nothing
end

function default_omega0(DeltaT)
    if DeltaT > 0
        return 20.0
    elseif DeltaT < 0
        return -20.0
    else
        return 0.0
    end
end

function solve_trajectory_2d(
    DeltaT,
    alpha,
    beta,
    tau;
    t_span,
    n_eval,
    theta0=theta0,
    omega0=nothing,
    rtol=RTOL,
    atol=ATOL,
)
    if omega0 === nothing
        omega0 = default_omega0(DeltaT)
    end

    u0 = Float64[theta0, omega0]
    p = (Float64(DeltaT), alpha, beta, tau)

    prob = ODEProblem(rhs_2d!, u0, t_span, p)
    t_eval = range(t_span[1], t_span[2], length=n_eval)

    sol = try
        solve(
            prob,
            Tsit5();
            saveat=t_eval,
            reltol=rtol,
            abstol=atol,
        )
    catch err
        println("ODE error at DeltaT=$(DeltaT): ", err)
        return nothing
    end

    if sol.retcode != ReturnCode.Success || length(sol.u) < 10
        println("ODE failed at DeltaT=$(DeltaT): ", sol.retcode)
        return nothing
    end

    theta = [u[1] for u in sol.u]
    omega = [u[2] for u in sol.u]

    if any(!isfinite, theta) || any(!isfinite, omega)
        return nothing
    end

    return sol.t, theta, omega
end

# =========================================
# 5. Predict rotation rates
# =========================================
function simulate_mean_freq_2d(DeltaT, alpha, beta, tau)
    traj = solve_trajectory_2d(
        DeltaT,
        alpha,
        beta,
        tau;
        t_span=T_SPAN_FREQ,
        n_eval=N_EVAL_FREQ,
    )

    if traj === nothing
        return NaN
    end

    _, _, omega = traj

    half = length(omega) ÷ 2
    omega_ss = omega[(half+1):end]

    mean_omega = mean(omega_ss)
    mean_freq = mean_omega / (2π)

    return mean_freq
end

function model_freq_2d(deltaT_model, params)
    alpha, beta, tau = params

    w_model = Float64[]

    for dT in deltaT_model
        w = simulate_mean_freq_2d(dT, alpha, beta, tau)
        print(@sprintf("DeltaT=%g: w_model=%.4g Hz\n", dT, w))

        if w === nothing || !isfinite(w)
            push!(w_model, NaN)
        else
            push!(w_model, w)
        end
    end

    return w_model
end

# =========================================
# 7. Plot rotation rate data and model predictions
# =========================================
function plot_freq_fit(deltaT_data, w_data, w_sd, params, out_dir, detail_dfs; model_width=0.1)
    alpha, beta, tau = params

    # -----------------------------
    # experimental data
    # -----------------------------
    detail_df = DataFrame(
        DeltaT=Float64.(deltaT_data),
        w_exp=Float64.(w_data),
        w_sd=Float64.(w_sd)
    )

    # -----------------------------
    # model data
    # -----------------------------
    dT_min = minimum(detail_df.DeltaT)
    dT_max = maximum(detail_df.DeltaT)

    deltaT_model = collect(dT_min:model_width:dT_max)

    # Append the final point dT_max if it was not included.
    if last(deltaT_model) < dT_max
        push!(deltaT_model, dT_max)
    end

    w_model = model_freq_2d(deltaT_model, params)

    model_df = DataFrame(
        DeltaT=Float64.(deltaT_model),
        w_model=Float64.(w_model),
    )

    # Remove NaN values.
    model_df = filter(row -> isfinite(row.w_model), model_df)

    # Sort as a precaution.
    sort!(model_df, :DeltaT)

    plt=hline!([0.0]; linewidth=1, label=false, color=:gray)

    plot!(
        plt,
        model_df.DeltaT,
        model_df.w_model;
        label="model",
        linewidth=3,
        color=:lightblue,
        grid=false,
        fontsize=24,
    )
    scatter!(
        plt,
        detail_df.DeltaT,
        detail_df.w_exp;
        yerror=detail_df.w_sd,
        label="experiment",
        xlabel="ΔT [K]",
        ylabel="⟨ω⟩ / 2π [Hz]",
        markersize=3,
        marker=:circle,
        markerstrokecolor=:black,
        markerstrokewidth=1.5,
        markercolor=:black,
        grid=false,
        framestyle=:box,
        # Font sizes
        guidefontsize=12,       # x- and y-axis labels
        tickfontsize=10,        # Tick labels
        legendfontsize=10,      # Legend
        titlefontsize=22,       # If a title is included
        size=(350, 350)
    )
    # Plot experimental data.
    for (i, df) in enumerate(detail_dfs)
        if nrow(df) == 0
            continue
        end

        scatter!(
            plt,
            df.DeltaT,
            df.w_exp;
            label=false,
            markersize=1,
            marker=:circle,
            markerstrokecolor=:black,
            markerstrokewidth=1,
            markercolor=:black,
        )
    end


    # hline!(plt, [0.0]; linewidth=1, label=false, color=:gray)

    fig_path = joinpath(out_dir, "freq_fit_given_params_alpha$(PARAMS[1])_beta$(PARAMS[2])_tau$(PARAMS[3]).pdf")
    savefig(plt, fig_path)

    println("Saved frequency fit to: ", fig_path)
end

# =========================================
# 9. Run
# =========================================
function main()
    println("PARAMS = ", PARAMS)
    println("OUT_DIR = ", OUT_DIR)

    deltaT_data, w_data, w_sd, detail_dfs = load_freq_data(FREQ_CSV_PATH)

    println("Loaded frequency data: n = ", length(w_data))

    plot_freq_fit(deltaT_data, w_data, w_sd, PARAMS, OUT_DIR, detail_dfs)

    println("\nDone.")
    println("Saved plots to: ", OUT_DIR)
end

main()
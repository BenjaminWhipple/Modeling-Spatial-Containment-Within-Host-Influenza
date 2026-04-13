"""
Post-hoc sensitivity analysis for spatial containment of infection.

Runs one-at-a-time (OAT) and pairwise parameter sweeps around calibrated values
and evaluates whether the infection is spatially contained.

Containment criterion:
  - Fraction of target cells remaining at final time >= min_T_fraction (default 0.80)
  - Aggregate infection (E1+E2+I) at final time <= max_final_infection (default 10.0)

Parameters swept (OAT):
  p_V, p_F, a_F, K_F, k_PV, DF_DV_ratio

Pairwise sweeps:
  (p_V, a_F), (p_V, p_F), (p_V, K_F), (a_F, p_F), (a_F, K_F), (p_F, K_F)

Usage:
    julia --project=. scripts/containment_posthoc.jl

Or from REPL:
    include("scripts/containment_posthoc.jl")
"""

using Printf
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using CSV
using DataFrames
using Plots
using Measures
using Dates
gr()

include(joinpath(@__DIR__, "..", "src", "models", "TIVF5_2D.jl"))
using .TIVF5_2D

include(joinpath(@__DIR__, "..", "src", "pde_utils.jl"))

# ============================================================================
# Parameter names (order matches θ vector used in set_params!)
# ============================================================================
const PARAM_NAMES = [
    "beta",      # 1  - infection rate
    "k_E",       # 2  - eclipse timing
    "delta_I",   # 3  - infected cell death
    "p_V",       # 4  - viral production
    "c_V",       # 5  - viral clearance
    "D_V",       # 6  - viral diffusion
    "k_PV",      # 7  - IFN production blocking
    "p_F",       # 8  - IFN production rate
    "c_F",       # 9  - IFN clearance
    "D_F",       # 10 - IFN diffusion
    "delta_FV",  # 11 - IFN-mediated viral clearance
    "k_FV",      # 12 - clearance saturation
    "k_IF",      # 13 - IFN infection blocking
    "a_F",       # 14 - positive feedback strength
    "K_F"        # 15 - feedback threshold
]

const PARAM_INDEX = Dict(name => i for (i, name) in enumerate(PARAM_NAMES))

# ============================================================================
# Calibrated baseline parameters (from test_tivf5_loss.jl)
# ============================================================================
function calibrated_baseline()
    return [
        8.000000e-02,  # beta
        3.177313e+00,  # k_E
        4.147009e+00,  # delta_I
        2.928123e+01,  # p_V
        1.000000e-01,  # c_V
        1.000000e-06,  # D_V
        6.659183e+00,  # k_PV
        6.066115e-02,  # p_F
        1.000000e-03,  # c_F
        9.463308e-01,  # D_F
        3.506720e-03,  # delta_FV
        9.389691e+02,  # k_FV
        1.000000e+12,  # k_IF
        9.872138e+03,  # a_F
        1.963406e+01   # K_F
    ]
end

# ============================================================================
# Containment detection
# ============================================================================
"""
    is_contained(sol, N; min_T_fraction=0.80, max_final_infection=10.0, rtol=0.02) -> Bool

Returns `true` if the infection is spatially contained:
  1. Fraction of target cells remaining >= min_T_fraction * (1 - rtol)
  2. Aggregate infection (E1+E2+I) at final time <= max_final_infection * (1 + rtol)

The relative tolerance `rtol` accounts for the fact that the calibration enforces
containment via a soft penalty rather than a hard constraint, so the calibrated
baseline may sit marginally outside the nominal threshold.
"""
function is_contained(sol, N::Int;
                      min_T_fraction::Float64=0.80,
                      max_final_infection::Float64=10.0,
                      rtol::Float64=0.02)
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)

    T_initial = sum(@view sol.u[1][idx_T])
    T_final   = sum(@view sol.u[end][idx_T])
    frac_remaining = T_final / max(T_initial, 1e-30)

    aggregate_infected = sum(@view sol.u[end][idx_E1]) +
                         sum(@view sol.u[end][idx_E2]) +
                         sum(@view sol.u[end][idx_I])

    return (frac_remaining >= min_T_fraction * (1.0 - rtol)) &&
           (aggregate_infected <= max_final_infection * (1.0 + rtol))
end

# ============================================================================
# Simulation runner
# ============================================================================
"""
Run a single forward simulation and return (sol, contained::Bool).
Returns (nothing, false) on solver failure.
"""
function run_sim(θ::Vector{Float64}, prob, N::Int, cb;
                 reltol::Float64=1e-4, dtmax::Float64=0.05,
                 min_T_fraction::Float64=0.80,
                 max_final_infection::Float64=10.0)
    TIVF5_2D.set_params!(prob, θ)

    atol = TIVF5_2D.abstol(N)

    sol = try
        solve(prob, QNDF(linsolve=KLUFactorization(), autodiff=false);
              reltol=reltol, abstol=atol, dtmax=dtmax,
              saveat=0.1, dense=false, save_everystep=false,
              callback=cb)
    catch
        return (nothing, false)
    end

    if sol.retcode != ReturnCode.Success
        return (nothing, false)
    end

    contained = is_contained(sol, N;
                             min_T_fraction=min_T_fraction,
                             max_final_infection=max_final_infection)
    return (sol, contained)
end

# ============================================================================
# OAT sweep
# ============================================================================
"""
One-at-a-time sweep for a single named parameter.

For the special name "DF_DV_ratio", we vary the ratio D_F / D_V by scaling D_F
while keeping D_V fixed, sweeping the log10 of the ratio.

Returns a DataFrame with columns:
  parameter, log10_value, param_value, contained (0/1), solver_success (0/1)
"""
function oat_sweep(param_name::String, θ_base::Vector{Float64},
                   prob, N::Int, cb;
                   W::Float64=2.0, n_points::Int=11,
                   min_T_fraction::Float64=0.80,
                   max_final_infection::Float64=10.0)

    rows = NamedTuple[]

    if param_name == "DF_DV_ratio"
        # Sweep the ratio D_F / D_V by varying D_F, keeping D_V fixed
        DV_base = θ_base[PARAM_INDEX["D_V"]]
        DF_base = θ_base[PARAM_INDEX["D_F"]]
        ratio_base = DF_base / DV_base
        log10_center = log10(ratio_base)

        log10_vals = range(log10_center - W, log10_center + W; length=n_points)

        for (i, lv) in enumerate(log10_vals)
            ratio_val = 10.0^lv
            θ = copy(θ_base)
            θ[PARAM_INDEX["D_F"]] = ratio_val * DV_base  # D_F = ratio * D_V

            @printf("  [%d/%d] DF/DV ratio = %.4e (log10 = %.3f)... ", i, n_points, ratio_val, lv)

            sol, contained = run_sim(θ, prob, N, cb;
                                     min_T_fraction=min_T_fraction,
                                     max_final_infection=max_final_infection)
            success = sol !== nothing

            status_str = success ? (contained ? "CONTAINED" : "NOT CONTAINED") : "FAILED"
            println(status_str)

            push!(rows, (
                parameter      = "DF_DV_ratio",
                log10_value    = lv,
                param_value    = ratio_val,
                contained      = contained ? 1 : 0,
                solver_success = success ? 1 : 0
            ))
        end
    else
        # Standard single-parameter sweep
        idx = PARAM_INDEX[param_name]
        base_val = θ_base[idx]
        log10_center = log10(base_val)

        log10_vals = range(log10_center - W, log10_center + W; length=n_points)

        for (i, lv) in enumerate(log10_vals)
            θ = copy(θ_base)
            θ[idx] = 10.0^lv

            @printf("  [%d/%d] %s = %.4e (log10 = %.3f)... ", i, n_points, param_name, θ[idx], lv)

            sol, contained = run_sim(θ, prob, N, cb;
                                     min_T_fraction=min_T_fraction,
                                     max_final_infection=max_final_infection)
            success = sol !== nothing

            status_str = success ? (contained ? "CONTAINED" : "NOT CONTAINED") : "FAILED"
            println(status_str)

            push!(rows, (
                parameter      = param_name,
                log10_value    = lv,
                param_value    = θ[idx],
                contained      = contained ? 1 : 0,
                solver_success = success ? 1 : 0
            ))
        end
    end

    return DataFrame(rows)
end

# ============================================================================
# Pairwise sweep
# ============================================================================
"""
Helper: resolve a parameter name (possibly "DF_DV_ratio") to its log10 center
and a function that applies a log10 value to a θ vector.

Returns `(log10_center, apply!)` where `apply!(θ, lv)` sets the parameter in θ.
"""
function _resolve_param(name::String, θ_base::Vector{Float64})
    if name == "DF_DV_ratio"
        DV_base = θ_base[PARAM_INDEX["D_V"]]
        DF_base = θ_base[PARAM_INDEX["D_F"]]
        log10_center = log10(DF_base / DV_base)
        apply! = (θ, lv) -> begin
            θ[PARAM_INDEX["D_F"]] = 10.0^lv * θ[PARAM_INDEX["D_V"]]
        end
        return (log10_center, apply!)
    else
        idx = PARAM_INDEX[name]
        log10_center = log10(θ_base[idx])
        apply! = (θ, lv) -> begin
            θ[idx] = 10.0^lv
        end
        return (log10_center, apply!)
    end
end

"""
Pairwise sweep for two named parameters (supports "DF_DV_ratio" for either axis).
Returns a DataFrame with columns:
  param1, param2, log10_val1, log10_val2, val1, val2, contained (0/1), solver_success (0/1)
"""
function pairwise_sweep(name1::String, name2::String,
                        θ_base::Vector{Float64}, prob, N::Int, cb;
                        W::Float64=2.0, n_points::Int=11,
                        min_T_fraction::Float64=0.80,
                        max_final_infection::Float64=10.0)

    log10_center1, apply1! = _resolve_param(name1, θ_base)
    log10_center2, apply2! = _resolve_param(name2, θ_base)

    lv1s = collect(range(log10_center1 - W, log10_center1 + W; length=n_points))
    lv2s = collect(range(log10_center2 - W, log10_center2 + W; length=n_points))

    rows = NamedTuple[]
    total = n_points * n_points
    count = 0

    for lv1 in lv1s
        for lv2 in lv2s
            count += 1
            θ = copy(θ_base)
            apply1!(θ, lv1)
            apply2!(θ, lv2)

            @printf("  [%d/%d] %s=%.3e, %s=%.3e ... ",
                    count, total, name1, 10.0^lv1, name2, 10.0^lv2)

            sol, contained = run_sim(θ, prob, N, cb;
                                     min_T_fraction=min_T_fraction,
                                     max_final_infection=max_final_infection)
            success = sol !== nothing

            status_str = success ? (contained ? "CONTAINED" : "NOT CONTAINED") : "FAILED"
            println(status_str)

            push!(rows, (
                param1         = name1,
                param2         = name2,
                log10_val1     = lv1,
                log10_val2     = lv2,
                val1           = 10.0^lv1,
                val2           = 10.0^lv2,
                contained      = contained ? 1 : 0,
                solver_success = success ? 1 : 0
            ))
        end
    end

    return DataFrame(rows)
end

# ============================================================================
# Plotting helpers
# ============================================================================
"""
Line plot for OAT sweep showing containment (binary) vs log10(parameter).
"""
function plot_oat(df::DataFrame, param_name::String, output_dir::String;
                  θ_base::Vector{Float64}=Float64[])
    x = df.log10_value
    y = df.contained

    # Colour points by outcome: green = contained, red = not contained
    colors = [yi == 1 ? :green : :red for yi in y]

    p = plot(x, y;
        seriestype=:scatter,
        markersize=6,
        markercolor=colors,
        markerstrokewidth=0,
        ylims=(-0.1, 1.1),
        yticks=([0, 1], ["Not Contained", "Contained"]),
        xlabel="log₁₀($param_name)",
        ylabel="Containment",
        title="OAT: $param_name",
        legend=false,
        size=(700, 350),
        left_margin=8mm,
        bottom_margin=6mm
    )

    # Add a connecting line to show transitions
    plot!(p, x, y; linewidth=1.5, linecolor=:gray, alpha=0.5, label="")

    # Mark baseline with a vertical dashed line
    if param_name == "DF_DV_ratio" && length(θ_base) >= 10
        DV = θ_base[PARAM_INDEX["D_V"]]
        DF = θ_base[PARAM_INDEX["D_F"]]
        baseline_log10 = log10(DF / DV)
    elseif haskey(PARAM_INDEX, param_name) && length(θ_base) >= PARAM_INDEX[param_name]
        baseline_log10 = log10(θ_base[PARAM_INDEX[param_name]])
    else
        baseline_log10 = nothing
    end

    if baseline_log10 !== nothing
        vline!(p, [baseline_log10]; linestyle=:dash, linecolor=:blue, linewidth=1.5, label="")
    end

    savefig(p, joinpath(output_dir, "oat_$(param_name).png"))
    println("  Saved: oat_$(param_name).png")
    return p
end

"""
Heatmap for pairwise sweep showing containment.
"""
function plot_pairwise(df::DataFrame, name1::String, name2::String, output_dir::String)
    lv1s = sort(unique(df.log10_val1))
    lv2s = sort(unique(df.log10_val2))

    n1 = length(lv1s)
    n2 = length(lv2s)

    # Build matrix: rows = param2, cols = param1 (for heatmap convention)
    Z = fill(NaN, n2, n1)
    for row in eachrow(df)
        i1 = findfirst(==(row.log10_val1), lv1s)
        i2 = findfirst(==(row.log10_val2), lv2s)
        if i1 !== nothing && i2 !== nothing
            Z[i2, i1] = Float64(row.contained)
        end
    end

    p = heatmap(lv1s, lv2s, Z;
        clims=(0.0, 1.0),
        color=cgrad([:red, :green]),
        colorbar=false,
        xlabel="log₁₀($name1)",
        ylabel="log₁₀($name2)",
        title="Pairwise: $name1 vs $name2",
        aspect_ratio=:auto,
        size=(600, 500),
        left_margin=8mm,
        bottom_margin=6mm
    )

    savefig(p, joinpath(output_dir, "pairwise_$(name1)_$(name2).png"))
    println("  Saved: pairwise_$(name1)_$(name2).png")
    return p
end

# ============================================================================
# Main
# ============================================================================
function main(;
              # ---- sweep configuration (easily modifiable) ----
              W::Float64  = 2.0,    # log10 half-width (±W decades)
              N_pts::Int  = 7,     # number of sweep points per axis
              nx::Int     = 32,
              ny::Int     = 32,
              # ---- containment thresholds ----
              min_T_fraction::Float64     = 0.50,
              #max_final_infection::Float64 = 10.0, # For the full containment and clearance
              max_final_infection::Float64 = 1e15, # For just spatial containment.
              # ---- output ----
              #output_dir::String = "PostHocResults")
              output_dir::String = "PostHocResults-noclearance")

    # ------------------------------------------------------------------
    # OAT parameters to sweep (add / remove entries here)
    # ------------------------------------------------------------------
    oat_params = [
        "p_V",
        "p_F",
        "a_F",
        "K_F",
        "k_PV",
        "DF_DV_ratio"   # relative magnitude of D_F / D_V
    ]

    # ------------------------------------------------------------------
    # Pairwise parameter pairs to sweep (add / remove entries here)
    # All C(6,2) = 15 pairs of {p_V, p_F, a_F, K_F, k_PV, DF_DV_ratio}
    # ------------------------------------------------------------------
    _pw_params = ["p_V", "p_F", "a_F", "K_F", "k_PV", "DF_DV_ratio"]
    pairwise_pairs = [(a, b) for i in eachindex(_pw_params)
                              for j in (i+1):length(_pw_params)
                              for (a, b) in ((_pw_params[i], _pw_params[j]),)]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    isdir(output_dir) || mkpath(output_dir)

    θ_base = calibrated_baseline()

    println("=" ^ 70)
    println("Post-Hoc Containment Sensitivity Analysis")
    println("Generated: $(Dates.now())")
    println("=" ^ 70)
    println()
    println("Baseline parameters (calibrated):")
    for (i, name) in enumerate(PARAM_NAMES)
        @printf("  %10s = %.6e  (log10 = %+.3f)\n", name, θ_base[i], log10(θ_base[i]))
    end
    println()
    @printf("Sweep half-width W = %.1f decades  |  Points per axis N = %d\n", W, N_pts)
    @printf("Containment thresholds: min_T_fraction = %.2f, max_final_infection = %.1f\n",
            min_T_fraction, max_final_infection)
    println()

    # Build model / problem (shared across all sweeps)
    pars = TIVF5_2D.Params(;
        beta     = θ_base[1],
        k_E      = θ_base[2],
        delta_I  = θ_base[3],
        p_V      = θ_base[4],
        c_V      = θ_base[5],
        D_V      = θ_base[6],
        k_PV     = θ_base[7],
        p_F      = θ_base[8],
        c_F      = θ_base[9],
        D_F      = θ_base[10],
        delta_FV = θ_base[11],
        k_FV     = θ_base[12],
        k_IF     = θ_base[13],
        a_F      = θ_base[14],
        K_F      = θ_base[15],
        eps_diff = 1e-6,
        delta_smooth = 1e-6
    )

    x, y, prob, N = TIVF5_2D.make_problem(;
        nx=nx, ny=ny, tspan=(0.0, 10.0),
        pars=pars,
        T0=100.0,
        T_init=nothing,
        E10=0.0, E20=0.0, I0=0.0,
        F0=1e-2,
        V_amp=1.0, sigma=0.04, center=(0.5, 0.5)
    )

    cb = positivity_callback_all(N; nfields=TIVF5_2D.nfields())

    # ------------------------------------------------------------------
    # 1. OAT sweeps
    # ------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("ONE-AT-A-TIME SWEEPS")
    println("=" ^ 70)

    for param_name in oat_params
        println("\n--- OAT: $param_name ---")
        df = oat_sweep(param_name, θ_base, prob, N, cb;
                       W=W, n_points=N_pts,
                       min_T_fraction=min_T_fraction,
                       max_final_infection=max_final_infection)

        csv_path = joinpath(output_dir, "oat_$(param_name).csv")
        CSV.write(csv_path, df)
        println("  Saved CSV: $csv_path")

        plot_oat(df, param_name, output_dir; θ_base=θ_base)
    end

    # ------------------------------------------------------------------
    # 2. Pairwise sweeps
    # ------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("PAIRWISE SWEEPS")
    println("=" ^ 70)

    for (name1, name2) in pairwise_pairs
        println("\n--- Pairwise: $name1 x $name2 ---")
        df = pairwise_sweep(name1, name2, θ_base, prob, N, cb;
                            W=W, n_points=N_pts,
                            min_T_fraction=min_T_fraction,
                            max_final_infection=max_final_infection)

        csv_path = joinpath(output_dir, "pairwise_$(name1)_$(name2).csv")
        CSV.write(csv_path, df)
        println("  Saved CSV: $csv_path")

        plot_pairwise(df, name1, name2, output_dir)
    end

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("OAT sweeps:      $(length(oat_params)) parameters x $N_pts points = $(length(oat_params)*N_pts) simulations")
    println("Pairwise sweeps: $(length(pairwise_pairs)) pairs x $(N_pts)x$(N_pts) = $(length(pairwise_pairs)*N_pts^2) simulations")
    println("Total simulations: $(length(oat_params)*N_pts + length(pairwise_pairs)*N_pts^2)")
    println("All results saved to: $output_dir/")
    println("=" ^ 70)
    println("Done.")
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    main(max_final_infection = 1e15, output_dir = "PostHocResults-noclearance",N_pts=31)
    main(max_final_infection = 10.0, output_dir = "PostHocResults-clearance",N_pts=31)
end

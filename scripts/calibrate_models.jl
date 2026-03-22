"""
Batch calibration script for TIVF5_2D model.
Runs multiple calibration configurations with different fixed/free parameter combinations.
"""
using Printf
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using Plots
using Measures
using Dates
gr()

include("../src/models/TIVF5_2D.jl")
using .TIVF5_2D

include("../src/calibration/Calibration.jl")
using .Calibration

include("../src/pde_utils.jl")

# Configuration struct
"""
Configuration for a single calibration run.
- `name`: identifier for this configuration (used in filenames)
- `description`: human-readable description
- `widths`: search widths in log10 space for each parameter (0.0 = fixed)
- `fixed_values`: Dict of parameter name => fixed value (in linear space)
                  Parameters not in this dict use the default baseline value
"""
struct CalibrationConfig
    name::String
    description::String
    widths::Vector{Float64}           # length 15, 0.0 means fixed
    fixed_values::Dict{String, Float64}  # name => value (linear space)
end

# Parameter names for reference (order matters - matches θ vector)
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

# Create name -> index mapping
const PARAM_INDEX = Dict(name => i for (i, name) in enumerate(PARAM_NAMES))

"""
Get parameter index from name.
"""
function param_idx(name::String)
    haskey(PARAM_INDEX, name) || error("Unknown parameter name: $name. Valid names: $(PARAM_NAMES)")
    return PARAM_INDEX[name]
end

# Default widths (all free except k_E which is narrow)
const DEFAULT_WIDTHS = [
    1.0,  # beta
    0.1,  # k_E (narrow - timing is sensitive and relatively well known)
    1.0,  # delta_I
    1.0,  # p_V
    1.0,  # c_V
    1.0,  # D_V
    1.0,  # k_PV
    1.0,  # p_F
    1.0,  # c_F
    1.0,  # D_F
    1.0,  # delta_FV
    1.0,  # k_FV
    1.0,  # k_IF
    1.0,  # a_F
    1.0   # K_F
]

# Special values for fixing parameters
const VAL_ZERO = 1e-12      # Effectively zero (can't use actual 0 in log space)
const VAL_LARGE = 1e12      # Effectively infinite
const VAL_ONE = 1.0         # Unity

"""
Create a config with specified parameters fixed at given values.

Arguments:
- `name`: configuration identifier
- `description`: human-readable description  
- `fixed_params`: Dict of parameter name => fixed value (linear space)
                  Use VAL_ZERO, VAL_LARGE, or specific values

Example:
    make_config("no_feedback", "No IFN positive feedback", 
                Dict("a_F" => VAL_ZERO))
"""
function make_config(name::String, description::String, 
                     fixed_params::Dict{String, Float64}=Dict{String, Float64}())
    widths = copy(DEFAULT_WIDTHS)
    for param_name in keys(fixed_params)
        idx = param_idx(param_name)
        widths[idx] = 0.0
    end
    return CalibrationConfig(name, description, widths, fixed_params)
end

"""
Apply fixed values to baseline parameter vector.
Returns modified θ_base with fixed values substituted.
"""
function apply_fixed_values(θ_base::Vector{Float64}, config::CalibrationConfig)
    θ = copy(θ_base)
    for (param_name, val) in config.fixed_values
        idx = param_idx(param_name)
        θ[idx] = val
    end
    return θ
end

"""
Count the number of parameters that are actually calibrated (non-zero width).
"""
function count_calibrated_params(config::CalibrationConfig)
    return count(w -> w > 0.0, config.widths)
end

# Define all 16 model configurations:
# Four IFN-mediated mechanisms:
# 1. IFN blocks infection:      β * k_IF/(k_IF + F)     | Null: k_IF = ∞ (VAL_LARGE)
# 2. IFN blocks production:     p_V * k_PV/(k_PV + F)   | Null: k_PV = ∞ (VAL_LARGE)
# 3. IFN increases clearance:   -δ_FV * F * (F/(F+k_FV)) * V | Null: delta_FV = 0 (VAL_ZERO)
# 4. IFN positive feedback:     p_F * I * (1 + a_F*F²/(K_F² + F²)) | Null: a_F = 0 (VAL_ZERO)

# Helper function to build model configs systematically
function make_model_config(model_num::Int, 
                           blocks_infection::Bool,
                           blocks_production::Bool, 
                           increases_clearance::Bool,
                           positive_feedback::Bool)
    
    fixed_params = Dict{String, Float64}()
    
    # Mechanism 1: IFN blocks infection (k_IF)
    if !blocks_infection
        fixed_params["k_IF"] = VAL_LARGE  # k_IF = ∞ disables this mechanism
    end
    
    # Mechanism 2: IFN blocks viral production (k_PV)
    if !blocks_production
        fixed_params["k_PV"] = VAL_LARGE  # k_PV = ∞ disables this mechanism
    end
    
    # Mechanism 3: IFN increases viral clearance (delta_FV)
    if !increases_clearance
        fixed_params["delta_FV"] = VAL_ZERO  # delta_FV = 0 disables this mechanism
    end
    
    # Mechanism 4: IFN positive feedback (a_F)
    if !positive_feedback
        fixed_params["a_F"] = VAL_ZERO  # a_F = 0 disables this mechanism
    end
    
    # Build description
    mechanisms = String[]
    blocks_infection && push!(mechanisms, "blocks infection")
    blocks_production && push!(mechanisms, "blocks production")
    increases_clearance && push!(mechanisms, "increases clearance")
    positive_feedback && push!(mechanisms, "positive feedback")
    
    if isempty(mechanisms)
        desc = "Null model (no IFN mechanisms)"
    elseif length(mechanisms) == 4
        desc = "Full model (all IFN mechanisms)"
    else
        desc = "IFN: " * join(mechanisms, ", ")
    end
    
    name = @sprintf("M%02d", model_num)
    
    return make_config(name, desc, fixed_params)
end

# Build all 16 models systematically
# Each row represents: (blocks_infection, blocks_production, increases_clearance, positive_feedback)
const MODEL_SPECS = [
    #    Blocks Inf  Blocks Prod  Inc Clear  Pos Feedback
    (1,  false,      false,       false,     false),  # M1:  Null model
    (2,  true,       false,       false,     false),  # M2:  Blocks infection only
    (3,  false,      true,        false,     false),  # M3:  Blocks production only
    (4,  true,       true,        false,     false),  # M4:  Blocks infection + production
    (5,  false,      false,       true,      false),  # M5:  Increases clearance only
    (6,  true,       false,       true,      false),  # M6:  Blocks infection + increases clearance
    (7,  false,      true,        true,      false),  # M7:  Blocks production + increases clearance
    (8,  true,       true,        true,      false),  # M8:  Blocks infection + production + clearance
    (9,  false,      false,       false,     true),   # M9:  Positive feedback only
    (10, true,       false,       false,     true),   # M10: Blocks infection + feedback
    (11, false,      true,        false,     true),   # M11: Blocks production + feedback
    (12, true,       true,        false,     true),   # M12: Blocks infection + production + feedback
    (13, false,      false,       true,      true),   # M13: Increases clearance + feedback
    (14, true,       false,       true,      true),   # M14: Blocks infection + clearance + feedback
    (15, false,      true,        true,      true),   # M15: Blocks production + clearance + feedback
    (16, true,       true,        true,      true),   # M16: Full model (all mechanisms)
]

const CONFIGS = [
    make_model_config(spec...) for spec in MODEL_SPECS
]

# Result storage
struct CalibrationResult
    config::CalibrationConfig
    θbest::Vector{Float64}
    nll_best::Float64
    nll_baseline::Float64
    elapsed_time::Float64
    converged::Bool
    n_calibrated::Int  # Number of parameters actually calibrated
end

# Area extent utilities
"""
Count area above threshold over tissue.
"""
@inline function area_above_threshold(u_slice, tissue::BitVector, dA::Float64, thr::Float64)
    cnt = 0
    @inbounds for i in eachindex(u_slice)
        if tissue[i] && (u_slice[i] > thr)
            cnt += 1
        end
    end
    return cnt * dA
end

"""
Compute time series of area (number of cells × dA) above a fixed threshold.
"""
function area_timeseries(sol, idxrng, tissue::BitVector, dA::Float64; thr::Float64=1e-6)
    A = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        u = sol.u[k]
        us = @view u[idxrng]
        A[k] = area_above_threshold(us, tissue, dA, thr)
    end
    return A
end

# Core calibration function
function run_calibration(config::CalibrationConfig, prob, N, loss, θ_default::Vector{Float64};
                         opts::DEOptions, verbose::Bool=true)
    
    # Apply fixed values to get config-specific baseline
    θ_base = apply_fixed_values(θ_default, config)
    ϕ_base = log10.(θ_base)
    
    # Count calibrated parameters
    n_calibrated = count_calibrated_params(config)
    
    if verbose
        println("\n" * "="^70)
        println("Running: $(config.name)")
        println("Description: $(config.description)")
        println("Calibrating $n_calibrated / $(length(PARAM_NAMES)) parameters")
        println("="^70)
        
        # Show which parameters are fixed and their values
        if !isempty(config.fixed_values)
            println("Fixed parameters:")
            for (name, val) in sort(collect(config.fixed_values))
                @printf("  %10s = %.2e (log10: %.2f)\n", name, val, log10(val))
            end
        else
            println("All parameters free")
        end
        println()
    end
    
    # Create bounds
    bounds = Bounds(ϕ_base .- config.widths, ϕ_base .+ config.widths)
    
    # Print bounds
    if verbose
        println("Parameter bounds (log10 space):")
        for (i, name) in enumerate(PARAM_NAMES)
            fixed_str = config.widths[i] == 0.0 ? " (FIXED)" : ""
            @printf("  %10s: [%7.2f, %7.2f] (center: %7.2f)%s\n", 
                    name, bounds.lower[i], bounds.upper[i], ϕ_base[i], fixed_str)
        end
        println()
    end
    
    # Test baseline
    nll_baseline = loss(ϕ_base)
    if verbose
        @printf("Baseline NLL = %.6e\n", nll_baseline)
    end
    
    # Run DE
    start_time = time()
    xbest, fbest, st = de_optimize(loss, bounds;
        opts=opts,
        sobol_skip=16,
        strategy=Calibration.DE.Best1,
        synchronous=true
    )
    elapsed = time() - start_time
    
    θbest = 10.0 .^ xbest
    converged = (st.gen < opts.max_gens)
    
    if verbose
        @printf("Best NLL = %.6e (improvement: %.2f%%)\n", fbest, 100*(nll_baseline - fbest)/nll_baseline)
        @printf("Elapsed time: %.2f seconds\n", elapsed)
        
        # Print best parameters
        println("\nBest parameters (linear space):")
        for (i, name) in enumerate(PARAM_NAMES)
            fixed_str = config.widths[i] == 0.0 ? " (FIXED)" : ""
            @printf("  %10s: %.6e%s\n", name, θbest[i], fixed_str)
        end
    end
    
    return CalibrationResult(config, θbest, fbest, nll_baseline, elapsed, converged, n_calibrated)
end

# Plotting functions
function generate_plots(result::CalibrationResult, prob, N, loss, config_dir::String, nx::Int, ny::Int)
    config = result.config
    θbest = result.θbest
    
    # Set parameters
    ϕbest = log10.(θbest)
    TIVF5_2D.set_params!(prob, ϕbest; params_in_log10=true)
    
    # Get GP time grids
    tV = vec(loss.gpV.t_eval)
    tF = vec(loss.gpF.t_eval)
    
    # Union time grid
    t_eval = sort!(unique!(vcat(tV, tF)))
    
    # Solve
    abstol = TIVF5_2D.abstol(N)
    cb = positivity_callback_all(N; nfields=TIVF5_2D.nfields())
    
    prob2 = remake(prob; tspan=(minimum(t_eval), maximum(t_eval)))
    sol = solve(prob2, QNDF(linsolve=KLUFactorization(), autodiff=false);
        reltol=1e-4, abstol=abstol, dtmax=0.05,
        saveat=t_eval, dense=false, save_everystep=false,
        callback=cb
    )
    
    if sol.retcode != :Success && sol.retcode != ReturnCode.Success
        @warn "Solve failed for $(config.name): $(sol.retcode)"
        return nothing
    end
    
    # Find indices in solution for GP times
    function find_indices(t_all, t_sub)
        [findfirst(t -> abs(t - ts) < 1e-10, t_all) for ts in t_sub]
    end
    
    idxV = find_indices(sol.t, tV)
    idxF = find_indices(sol.t, tF)
    
    # Field indices
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)
    idx_V  = (4N+1):(5N)
    idx_F  = (5N+1):(6N)
    
    # Extract model predictions
    yV_all = TIVF5_2D.default_obs_totalV_log10(sol, N)
    yF_all = TIVF5_2D.default_obs_totalF_log10(sol, N)
    
    yV_model = yV_all[idxV]
    yF_model = yF_all[idxF]
    
    # GP data
    muV_gp = vec(loss.gpV.mu)
    σV_gp = sqrt.(max.(diag(loss.gpV.cov), 0.0))
    muF_gp = vec(loss.gpF.mu)
    σF_gp = sqrt.(max.(diag(loss.gpF.cov), 0.0))
    
    kband = 1.0
    
    # Virus GP comparison plot
    pV = plot(tV, muV_gp; ribbon=(kband .* σV_gp, kband .* σV_gp),
              fillalpha=0.2, label="Virus GP μ±σ", xlabel="t", ylabel="log10(total V)",
              title="$(config.name): Virus")
    plot!(pV, tV, yV_model; lw=2, label="Model")
    savefig(pV, joinpath(config_dir, "virus_comparison.png"))
    println("  Saved: virus_comparison.png")
    
    # IFN GP comparison plot
    pF = plot(tF, muF_gp; ribbon=(kband .* σF_gp, kband .* σF_gp),
              fillalpha=0.2, label="IFN GP μ±σ", xlabel="t", ylabel="log10(total F)",
              title="$(config.name): IFN")
    plot!(pF, tF, yF_model; lw=2, label="Model")
    savefig(pF, joinpath(config_dir, "ifn_comparison.png"))
    println("  Saved: ifn_comparison.png")
    
    # Combined V + F plot
    p_combined = plot(pV, pF; layout=(1, 2), size=(1000, 400), margin=5mm)
    savefig(p_combined, joinpath(config_dir, "combined_comparison.png"))
    println("  Saved: combined_comparison.png")
    
    # Aggregate mass trajectories
    eps = 1e-12
    Tmass  = [sum(@view u[idx_T])  + eps for u in sol.u]
    E1mass = [sum(@view u[idx_E1]) + eps for u in sol.u]
    E2mass = [sum(@view u[idx_E2]) + eps for u in sol.u]
    Imass  = [sum(@view u[idx_I])  + eps for u in sol.u]
    Vmass  = [sum(@view u[idx_V])  + eps for u in sol.u]
    Fmass  = [sum(@view u[idx_F])  + eps for u in sol.u]
    
    yticks_vals = 10.0 .^ (0:7)
    yticks_labs = ["$k" for k in 0:7]
    
    pAgg = plot(
        t_eval, max.(Tmass, eps);
        label="∫T dA",
        xlabel="t",
        ylabel="log10 mass",
        title="Aggregate trajectories: $(config.name)",
        yscale=:log10,
        yticks=(yticks_vals, yticks_labs),
        ylims=(1.0, 1e7)
    )
    plot!(pAgg, t_eval, E1mass .+ 1.0; label="∫E1 dA")
    plot!(pAgg, t_eval, E2mass .+ 1.0; label="∫E2 dA")
    plot!(pAgg, t_eval, Imass  .+ 1.0; label="∫I dA")
    plot!(pAgg, t_eval, Vmass  .+ 1.0; label="∫V dA")
    plot!(pAgg, t_eval, Fmass  .+ 1.0; label="∫F dA")
    savefig(pAgg, joinpath(config_dir, "aggregate_masses.png"))
    println("  Saved: aggregate_masses.png")
    
    # Spatial extent (area above threshold) plots
    #tissue = prob2.p.tissue
    #dA = prob2.p.dA

    tissue = prob2.p.tissue
    dA_raw = prob2.p.dA
    
    # Correct dA so total tissue area = 1.0. dA_raw currently includes the boundary cells.
    n_tissue_cells = count(tissue)
    dA = 1.0 / n_tissue_cells
    
    # Multiple thresholds for different insights
    thresholds = [1e-6, 0.1, 1.0, 10.0]
    
    for thr in thresholds
        AT  = area_timeseries(sol, idx_T,  tissue, dA; thr=thr)
        AE1 = area_timeseries(sol, idx_E1, tissue, dA; thr=thr)
        AE2 = area_timeseries(sol, idx_E2, tissue, dA; thr=thr)
        AI  = area_timeseries(sol, idx_I,  tissue, dA; thr=thr)
        AV  = area_timeseries(sol, idx_V,  tissue, dA; thr=thr)
        AF  = area_timeseries(sol, idx_F,  tissue, dA; thr=thr)
        
        pA = plot(sol.t, AT;  lw=2, label="T",  xlabel="t", ylabel="Area",
                  title="Spatial extent (thr=$thr): $(config.name)")
        plot!(pA, sol.t, AE1; lw=2, label="E1")
        plot!(pA, sol.t, AE2; lw=2, label="E2")
        plot!(pA, sol.t, AI;  lw=2, label="I")
        plot!(pA, sol.t, AV;  lw=2, label="V")
        plot!(pA, sol.t, AF;  lw=2, label="F")
        
        thr_str = replace(@sprintf("%.0e", thr), "." => "p")
        savefig(pA, joinpath(config_dir, "area_extent_thr$(thr_str).png"))
        println("  Saved: area_extent_thr$(thr_str).png")
    end
    
    # Combined area extent plot (all fields, one threshold)
    thr_main = 1e-6
    AT  = area_timeseries(sol, idx_T,  tissue, dA; thr=thr_main)
    AE1 = area_timeseries(sol, idx_E1, tissue, dA; thr=thr_main)
    AE2 = area_timeseries(sol, idx_E2, tissue, dA; thr=thr_main)
    AI  = area_timeseries(sol, idx_I,  tissue, dA; thr=thr_main)
    AV  = area_timeseries(sol, idx_V,  tissue, dA; thr=thr_main)
    AF  = area_timeseries(sol, idx_F,  tissue, dA; thr=thr_main)
    
    pA_main = plot(sol.t, AT;  lw=2, label="T",  xlabel="t", ylabel="Area",
              title="Spatial extent: $(config.name)")
    plot!(pA_main, sol.t, AE1; lw=2, label="E1")
    plot!(pA_main, sol.t, AE2; lw=2, label="E2")
    plot!(pA_main, sol.t, AI;  lw=2, label="I")
    plot!(pA_main, sol.t, AV;  lw=2, label="V")
    plot!(pA_main, sol.t, AF;  lw=2, label="F")
    savefig(pA_main, joinpath(config_dir, "area_extent.png"))
    println("  Saved: area_extent.png")
    
    return p_combined
end

function generate_summary_report(results::Vector{CalibrationResult}, output_dir::String)
    # Sort by best NLL
    sorted_results = sort(results, by=r -> r.nll_best)
    
    report_path = joinpath(output_dir, "calibration_summary.txt")
    open(report_path, "w") do io
        println(io, "="^100)
        println(io, "TIVF5 Model Comparison Summary Report")
        println(io, "Generated: $(Dates.now())")
        println(io, "="^100)
        println(io)
        
        # Mechanism legend
        println(io, "Mechanism Key:")
        println(io, "  BI = IFN blocks infection (k_IF)")
        println(io, "  BP = IFN blocks viral production (k_PV)")
        println(io, "  IC = IFN increases viral clearance (delta_FV)")
        println(io, "  PF = IFN positive feedback (a_F)")
        println(io)
        println(io, "Null settings: k_IF=∞, k_PV=∞, delta_FV=0, a_F=0")
        println(io)
        
        # Ranking table
        println(io, "Ranking by NLL (best to worst):")
        println(io, "-"^100)
        @printf(io, "%4s  %-6s  %4s  %-4s %-4s %-4s %-4s  %12s  %12s  %8s\n", 
                "Rank", "Model", "nCal", "BI", "BP", "IC", "PF", "NLL", "Improvement", "Time(s)")
        println(io, "-"^100)
        
        for (rank, r) in enumerate(sorted_results)
            improvement = 100 * (r.nll_baseline - r.nll_best) / abs(r.nll_baseline)
            
            # Determine which mechanisms are active (not in fixed_values with null setting)
            bi = !haskey(r.config.fixed_values, "k_IF") ? "✓" : "·"
            bp = !haskey(r.config.fixed_values, "k_PV") ? "✓" : "·"
            ic = !haskey(r.config.fixed_values, "delta_FV") ? "✓" : "·"
            pf = !haskey(r.config.fixed_values, "a_F") ? "✓" : "·"
            
            @printf(io, "%4d  %-6s  %4d  %-4s %-4s %-4s %-4s  %12.4e  %11.2f%%  %8.1f\n",
                    rank, r.config.name, r.n_calibrated, bi, bp, ic, pf, 
                    r.nll_best, improvement, r.elapsed_time)
        end
        println(io, "-"^100)
        println(io)
        
        # Model comparison table (in model order, not rank order)
        println(io, "\nModel Comparison (in model order):")
        println(io, "-"^100)
        @printf(io, "%-6s  %-4s %-4s %-4s %-4s  %12s  %-40s\n",
                "Model", "BI", "BP", "IC", "PF", "NLL", "Description")
        println(io, "-"^100)
        
        # Sort by model name for this table
        model_order = sort(results, by=r -> r.config.name)
        for r in model_order
            bi = !haskey(r.config.fixed_values, "k_IF") ? "✓" : "·"
            bp = !haskey(r.config.fixed_values, "k_PV") ? "✓" : "·"
            ic = !haskey(r.config.fixed_values, "delta_FV") ? "✓" : "·"
            pf = !haskey(r.config.fixed_values, "a_F") ? "✓" : "·"
            
            desc = length(r.config.description) > 38 ? r.config.description[1:35] * "..." : r.config.description
            @printf(io, "%-6s  %-4s %-4s %-4s %-4s  %12.4e  %-40s\n",
                    r.config.name, bi, bp, ic, pf, r.nll_best, desc)
        end
        println(io, "-"^100)
        println(io)
        
        # Best parameters for all configurations
        println(io, "\nAll configurations - Best parameters (ranked by NLL):")
        println(io, "="^100)
        
        for (rank, r) in enumerate(sorted_results)
            println(io, "\n#$rank: $(r.config.name)")
            println(io, "Description: $(r.config.description)")
            println(io, "Calibrated parameters: $(r.n_calibrated) / $(length(PARAM_NAMES))")
            
            # Show fixed parameters
            if !isempty(r.config.fixed_values)
                println(io, "Fixed (null): ", join(["$k=$(v < 1 ? "0" : "∞")" for (k,v) in r.config.fixed_values], ", "))
            end
            println(io, "-"^40)
            
            # Print parameters
            println(io, "pars = TIVF5_2D.Params(;")
            @printf(io, "    beta=%.6e,\n", r.θbest[1])
            @printf(io, "    k_E=%.6e,\n", r.θbest[2])
            @printf(io, "    delta_I=%.6e,\n", r.θbest[3])
            @printf(io, "    k_IF=%.6e,\n", r.θbest[13])
            @printf(io, "    p_V=%.6e,\n", r.θbest[4])
            @printf(io, "    c_V=%.6e,\n", r.θbest[5])
            @printf(io, "    D_V=%.6e,\n", r.θbest[6])
            @printf(io, "    k_PV=%.6e,\n", r.θbest[7])
            @printf(io, "    p_F=%.6e,\n", r.θbest[8])
            @printf(io, "    c_F=%.6e,\n", r.θbest[9])
            @printf(io, "    D_F=%.6e,\n", r.θbest[10])
            @printf(io, "    a_F=%.6e,\n", r.θbest[14])
            @printf(io, "    K_F=%.6e,\n", r.θbest[15])
            @printf(io, "    delta_FV=%.6e,\n", r.θbest[11])
            @printf(io, "    k_FV=%.6e,\n", r.θbest[12])
            println(io, "    eps_diff=1e-6,")
            println(io, "    delta_smooth=1e-6")
            println(io, ")")
        end
    end
    
    println("Saved summary report: $report_path")
    return report_path
end

# Summary comparison plot
function generate_summary_plot(results::Vector{CalibrationResult}, output_dir::String)
    # Sort by NLL
    sorted_results = sort(results, by=r -> r.nll_best)
    
    names = [r.config.name for r in sorted_results]
    nlls = [r.nll_best for r in sorted_results]
    improvements = [100 * (r.nll_baseline - r.nll_best) / abs(r.nll_baseline) for r in sorted_results]
    n_calibs = [r.n_calibrated for r in sorted_results]
    
    # Truncate long names
    short_names = [length(n) > 20 ? n[1:17] * "..." : n for n in names]
    
    # Bar plot of NLL
    p1 = bar(short_names, nlls;
             xlabel="Configuration", ylabel="NLL",
             title="Best NLL by Configuration (sorted)",
             xrotation=45, legend=false,
             margin=10mm, color=:steelblue)
    
    # Bar plot of improvement
    colors = [imp >= 0 ? :green : :red for imp in improvements]
    p2 = bar(short_names, improvements;
             xlabel="Configuration", ylabel="Improvement (%)",
             title="Improvement from Config Baseline",
             xrotation=45, legend=false,
             margin=10mm, color=colors)
    
    # Bar plot of number of calibrated parameters
    p3 = bar(short_names, n_calibs;
             xlabel="Configuration", ylabel="# Calibrated Params",
             title="Number of Calibrated Parameters",
             xrotation=45, legend=false,
             margin=10mm, color=:orange)
    
    p_summary = plot(p1, p2, p3; layout=(3, 1), size=(1200, 1000))
    savefig(p_summary, joinpath(output_dir, "calibration_summary.png"))
    
    println("Saved summary plot: calibration_summary.png")
    return p_summary
end

# Main function
function main(; 
              configs::Vector{CalibrationConfig}=CONFIGS,
              max_gens::Int=20,
              np::Int=60,
              stall_gens::Int=10,
              seed::Int=42,
              output_dir::String="Results/calibration_runs")
    
    # Create output directory
    isdir(output_dir) || mkpath(output_dir)
    
    println("="^70)
    println("TIVF5 Batch Calibration")
    println("Running $(length(configs)) configurations")
    println("Output directory: $output_dir")
    println("="^70)
    
    nx, ny = 32, 32
    
    # Default baseline parameters (before any fixing)
    pars = TIVF5_2D.Params(;
        beta=0.8,
        k_E=4.0,
        delta_I=3.0,
        k_IF=2.0,
        p_V=10.0,
        c_V=1.0,
        D_V=1e-5,
        k_PV=1.0,
        p_F=0.03,
        c_F=0.01,
        D_F=1e-1,
        mF=1.0,
        a_F=1000.0,
        K_F=1e1,
        delta_FV=0.001,
        k_FV=1e2,
        eps_diff=1e-6,
        delta_smooth=1e-6
    )
    
    x, y, prob, N = TIVF5_2D.make_problem(;
        nx=nx, ny=ny, tspan=(0.0, 10.0),
        pars=pars,
        T0=100.0,
        T_init=nothing,
        E10=0.0,
        E20=0.0,
        I0=0.0,
        F0=1e-2,
        V_amp=1.0,
        sigma=0.04,
        center=(0.5, 0.5)
    )
    
    abstol = TIVF5_2D.abstol(N)
    cb = positivity_callback_all(N; nfields=TIVF5_2D.nfields())
    
    loss = TIVF5_2D.make_tivf_gp_loss(prob, N;
        gp_path_V="src/data/gp_parameters/Toapanta_Virus_gp.npz",
        gp_path_F="src/data/gp_parameters/IFN_gp.npz",
        abstol=abstol,
        reltol=1e-4,
        dtmax=0.05,
        penalty=1e20,
        params_in_log10=true,
        callback=cb,
        wF=1.0,
        min_T_fraction=0.8,                 # at least 80% of T must remain at final time
        infection_penalty_weight=20.0,      # strength of the penalty
        max_final_infection=10.0,           # aggregate E1+E2+I at final time must be < 10
        clearance_penalty_weight=20.0       # strength of the clearance penalty
    )
    
    # Default parameter vector (before fixing)
    θ_raw = [
        pars.beta, pars.k_E, pars.delta_I, pars.p_V, pars.c_V,
        pars.D_V, pars.k_PV, pars.p_F, pars.c_F, pars.D_F,
        pars.delta_FV, pars.k_FV, pars.k_IF, pars.a_F, pars.K_F
    ]

    θ_default = θ_raw #10.0 .^ floor.(log10.(θ_raw))

    # DE options (shared across all runs)
    opts = DEOptions{Float64}(
        NP=np,
        F=0.7,
        CR=0.9,
        max_gens=max_gens,
        constraint=Calibration.OptimTypes.Clamp,
        seed=seed,
        stall_gens=stall_gens,
        verbose=true
    )
    
    # Run all calibrations
    results = CalibrationResult[]
    
    total_start = time()
    for (i, config) in enumerate(configs)
        println("\n" * "="^30)
        println("[$i/$(length(configs))] Starting: $(config.name)")
        println("="^30)
        
        # Create config-specific output directory
        config_dir = joinpath(output_dir, config.name)
        isdir(config_dir) || mkpath(config_dir)
        
        result = run_calibration(config, prob, N, loss, θ_default; opts=opts, verbose=true)
        push!(results, result)
        
        # Generate plots for this run (saved in config-specific folder)
        println("\nGenerating plots for $(config.name)...")
        generate_plots(result, prob, N, loss, config_dir, nx, ny)
    end
    total_elapsed = time() - total_start
    
    println("\n" * "="^70)
    @printf("Total elapsed time: %.2f seconds (%.2f minutes)\n", total_elapsed, total_elapsed/60)
    println("="^70)
    
    # Generate summary
    generate_summary_report(results, output_dir)
    generate_summary_plot(results, output_dir)
    
    return results
end

# Convenience function to run specific configs
"""
Run a subset of configurations by name.

Example:
    run_configs(["baseline", "no_feedback", "strong_feedback"])
"""
function run_configs(config_names::Vector{String}; kwargs...)
    selected = filter(c -> c.name in config_names, CONFIGS)
    if length(selected) != length(config_names)
        found = [c.name for c in selected]
        missing = setdiff(config_names, found)
        @warn "Some configs not found: $missing"
    end
    return main(; configs=selected, kwargs...)
end

"""
Create and run a custom configuration.

Example:
    run_custom("my_test", "Test with a_F=500", Dict("a_F" => 500.0))
"""
function run_custom(name::String, description::String, 
                    fixed_params::Dict{String, Float64}; kwargs...)
    config = make_config(name, description, fixed_params)
    return main(; configs=[config], kwargs...)
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    # Run only models M09–M16 (indices 9:16 in CONFIGS)
    #results = main(; configs=CONFIGS[10:16], max_gens=100, stall_gens=20)
    
    # Alternative: run all configs
    results = main(; max_gens=100, stall_gens=20)
    
    # Alternative: run specific configs by name
    #results = run_configs(["M16"]; max_gens=100,stall_gens=20)
    
    # Alternative: custom single run
    # results = run_custom("test", "Testing a_F=500", Dict("a_F" => 500.0); max_gens=100)
end
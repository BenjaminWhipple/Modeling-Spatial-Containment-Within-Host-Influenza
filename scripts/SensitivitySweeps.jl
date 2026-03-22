"""
Computes first-order sensitivity sweeps for the TIVF5_2D model.
For each parameter, varies it over a log10 range while holding others fixed,
and records key outcome metrics.

Outputs:
- CSV file with columns: parameter, parameter_value, total_infected_area, 
  total_virus_AUC, total_infected_AUC, peak_time_virus, peak_virus,
  peak_time_infection, peak_infection, peak_time_IFN, peak_IFN

Usage:
    julia --project=. scripts/SensitivitySweeps.jl
    
Or from REPL:
    include("scripts/SensitivitySweeps.jl")
    run_sensitivity_sweeps(θ_base; n_points=15, log10_range=1.0)
"""

using Printf
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "src", "models", "TIVF5_2D.jl"))
using .TIVF5_2D

include(joinpath(@__DIR__, "..", "src", "pde_utils.jl"))

# Parameter definitions

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

# Outcome computation functions
"""
Compute area above threshold at each time point.
Returns time series of area values.
"""
function area_timeseries(sol, idxrng, tissue::BitVector, dA::Float64; thr::Float64=1e-6)
    A = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        u = sol.u[k]
        us = @view u[idxrng]
        cnt = 0
        for i in eachindex(us)
            if tissue[i] && (us[i] > thr)
                cnt += 1
            end
        end
        A[k] = cnt * dA
    end
    return A
end

"""
Compute AUC using trapezoidal rule.
"""
function compute_auc(t::AbstractVector, y::AbstractVector)
    auc = 0.0
    @inbounds for i in 1:(length(t)-1)
        dt = t[i+1] - t[i]
        auc += 0.5 * (y[i] + y[i+1]) * dt
    end
    return auc
end

"""
Find peak time and peak value for a time series.
"""
function find_peak(t::AbstractVector, y::AbstractVector)
    idx = argmax(y)
    return (peak_time=t[idx], peak_value=y[idx])
end

"""
Compute all outcome metrics from a solution.

Returns NamedTuple with:
- total_infected_area: maximum fraction of domain ever infected (I > threshold)
- total_virus_AUC: area under curve for total virus
- total_infected_AUC: area under curve for total infected cells
- peak_time_virus, peak_virus: timing and value of virus peak
- peak_time_infection, peak_infection: timing and value of infection peak
- peak_time_IFN, peak_IFN: timing and value of IFN peak
"""
function compute_outcomes(sol, N::Int, tissue::BitVector, dA::Float64; 
                          thr_infection::Float64=1e-6)
    # Field indices
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)
    idx_V  = (4N+1):(5N)
    idx_F  = (5N+1):(6N)
    
    eps = 1e-12
    
    # Compute aggregate quantities at each time point
    Imass = [sum(@view u[idx_I]) * dA + eps for u in sol.u]
    Vmass = [sum(@view u[idx_V]) * dA + eps for u in sol.u]
    Fmass = [sum(@view u[idx_F]) * dA + eps for u in sol.u]
    
    # Infected area time series
    AI = area_timeseries(sol, idx_I, tissue, dA; thr=thr_infection)
    
    # Total infected area (max extent ever reached)
    total_infected_area = maximum(AI)
    
    # AUCs (using log scale for numerical stability with large values)
    total_virus_AUC = compute_auc(sol.t, Vmass)
    total_infected_AUC = compute_auc(sol.t, Imass)
    
    # Peaks
    peak_V = find_peak(sol.t, Vmass)
    peak_I = find_peak(sol.t, Imass)
    peak_F = find_peak(sol.t, Fmass)
    
    return (
        total_infected_area = total_infected_area,
        total_virus_AUC = total_virus_AUC,
        total_infected_AUC = total_infected_AUC,
        peak_time_virus = peak_V.peak_time,
        peak_virus = peak_V.peak_value,
        peak_time_infection = peak_I.peak_time,
        peak_infection = peak_I.peak_value,
        peak_time_IFN = peak_F.peak_time,
        peak_IFN = peak_F.peak_value
    )
end

# Sensitivity sweep functions
"""
Run a single forward simulation with given parameters.
Returns outcomes or nothing if simulation fails.
"""
function run_simulation(θ::Vector{Float64}, prob, N::Int, tissue::BitVector, dA::Float64;
                        abstol::Float64=1e-6, reltol::Float64=1e-4, 
                        dtmax::Float64=0.05, callback=nothing)
    # Update parameters (modifies prob in-place, returns nothing)
    TIVF5_2D.set_params!(prob, θ)
    
    # Solve with autodiff=false to avoid ForwardDiff issues with pospart
    sol = solve(prob, QNDF(autodiff=false, linsolve=KrylovJL_GMRES());
                abstol=abstol, reltol=reltol, dtmax=dtmax,
                saveat=0.1, callback=callback)
    
    if sol.retcode != :Success && sol.retcode != ReturnCode.Success
        return nothing
    end
    
    return compute_outcomes(sol, N, tissue, dA)
end

"""
Run sensitivity sweep for a single parameter.

Arguments:
- param_idx: index of parameter to sweep (1-15)
- θ_base: baseline parameter vector (linear space)
- n_points: number of evaluation points
- log10_range: range in log10 space (±log10_range around baseline)
- prob, N, tissue, dA: model setup
- abstol, reltol, dtmax, callback: solver options

Returns: Vector of NamedTuples with parameter values and outcomes
"""
function sweep_parameter(param_idx::Int, θ_base::Vector{Float64}, 
                         n_points::Int, log10_range::Float64,
                         prob, N::Int, tissue::BitVector, dA::Float64;
                         abstol::Float64=1e-6, reltol::Float64=1e-4,
                         dtmax::Float64=0.05, callback=nothing)
    
    param_name = PARAM_NAMES[param_idx]
    base_value = θ_base[param_idx]
    log10_base = log10(base_value)
    
    # Generate log-spaced sweep values
    log10_values = range(log10_base - log10_range, log10_base + log10_range; length=n_points)
    
    results = NamedTuple[]
    
    for (i, log10_val) in enumerate(log10_values)
        θ = copy(θ_base)
        θ[param_idx] = 10.0^log10_val
        
        @printf("  [%d/%d] %s = %.4e (log10 = %.3f)... ", i, n_points, param_name, θ[param_idx], log10_val)
        
        try
            outcomes = run_simulation(θ, prob, N, tissue, dA; 
                                      abstol=abstol, reltol=reltol, 
                                      dtmax=dtmax, callback=callback)
            
            if outcomes === nothing
                println("FAILED (solver)")
                push!(results, (
                    parameter = param_name,
                    parameter_value = θ[param_idx],
                    log10_value = log10_val,
                    total_infected_area = NaN,
                    total_virus_AUC = NaN,
                    total_infected_AUC = NaN,
                    peak_time_virus = NaN,
                    peak_virus = NaN,
                    peak_time_infection = NaN,
                    peak_infection = NaN,
                    peak_time_IFN = NaN,
                    peak_IFN = NaN
                ))
            else
                @printf("OK (peak V=%.2e @ t=%.2f)\n", outcomes.peak_virus, outcomes.peak_time_virus)
                push!(results, (
                    parameter = param_name,
                    parameter_value = θ[param_idx],
                    log10_value = log10_val,
                    total_infected_area = outcomes.total_infected_area,
                    total_virus_AUC = outcomes.total_virus_AUC,
                    total_infected_AUC = outcomes.total_infected_AUC,
                    peak_time_virus = outcomes.peak_time_virus,
                    peak_virus = outcomes.peak_virus,
                    peak_time_infection = outcomes.peak_time_infection,
                    peak_infection = outcomes.peak_infection,
                    peak_time_IFN = outcomes.peak_time_IFN,
                    peak_IFN = outcomes.peak_IFN
                ))
            end
        catch e
            println("FAILED (exception: $e)")
            push!(results, (
                parameter = param_name,
                parameter_value = θ[param_idx],
                log10_value = log10_val,
                total_infected_area = NaN,
                total_virus_AUC = NaN,
                total_infected_AUC = NaN,
                peak_time_virus = NaN,
                peak_virus = NaN,
                peak_time_infection = NaN,
                peak_infection = NaN,
                peak_time_IFN = NaN,
                peak_IFN = NaN
            ))
        end
    end
    
    return results
end

"""
Run sensitivity sweeps for all (or a subset of) parameters.

Arguments:
- θ_base: baseline parameter vector (15 elements, linear space)
- param_indices: which parameters to sweep (default: all 1:15)
- n_points: number of evaluation points per parameter (default: 15)
- log10_range: sweep range in log10 space (default: 1.0, i.e., ±1 decade)
- output_file: path to CSV output file
- nx, ny: grid dimensions (default: 32×32)

Returns: DataFrame with all results
"""
function run_sensitivity_sweeps(θ_base::Vector{Float64};
                                param_indices::Vector{Int}=collect(1:15),
                                n_points::Int=15,
                                log10_range::Float64=1.0,
                                output_file::String="Results/sensitivity_sweeps.csv",
                                nx::Int=32, ny::Int=32)
    
    println("="^70)
    println("TIVF5_2D Sensitivity Sweeps")
    println("="^70)
    println("Baseline parameters:")
    for (i, name) in enumerate(PARAM_NAMES)
        @printf("  %10s = %.4e (log10 = %.3f)\n", name, θ_base[i], log10(θ_base[i]))
    end
    println()
    println("Sweep configuration:")
    println("  Parameters to sweep: $(length(param_indices))")
    println("  Points per parameter: $n_points")
    println("  Log10 range: ±$log10_range")
    println("  Total simulations: $(length(param_indices) * n_points)")
    println("  Output file: $output_file")
    println("="^70)
    
    # Set up model
    pars = TIVF5_2D.Params(;
        beta=θ_base[1],
        k_E=θ_base[2],
        delta_I=θ_base[3],
        p_V=θ_base[4],
        c_V=θ_base[5],
        D_V=θ_base[6],
        k_PV=θ_base[7],
        p_F=θ_base[8],
        c_F=θ_base[9],
        D_F=θ_base[10],
        delta_FV=θ_base[11],
        k_FV=θ_base[12],
        k_IF=θ_base[13],
        a_F=θ_base[14],
        K_F=θ_base[15],
        mF=1.0,
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
    
    tissue = prob.p.tissue
    n_tissue_cells = count(tissue)
    dA = 1.0 / n_tissue_cells  # Normalized so total area = 1.0
    
    abstol = 1e-6  # Scalar tolerance (simpler than per-component)
    cb = positivity_callback_all(N; nfields=TIVF5_2D.nfields())
    
    # Run sweeps
    all_results = NamedTuple[]
    start_time = time()
    
    for (sweep_idx, param_idx) in enumerate(param_indices)
        param_name = PARAM_NAMES[param_idx]
        println("\n[$sweep_idx/$(length(param_indices))] Sweeping parameter: $param_name")
        println("-"^50)
        
        results = sweep_parameter(param_idx, θ_base, n_points, log10_range,
                                  prob, N, tissue, dA;
                                  abstol=abstol, reltol=1e-4, dtmax=0.05, callback=cb)
        
        append!(all_results, results)
    end
    
    elapsed = time() - start_time
    println("\n" * "="^70)
    @printf("Completed %d simulations in %.2f seconds (%.2f sec/sim)\n", 
            length(all_results), elapsed, elapsed/length(all_results))
    
    # Convert to DataFrame and save
    df = DataFrame(all_results)
    
    # Ensure output directory exists
    output_dir = dirname(output_file)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    CSV.write(output_file, df)
    println("Results saved to: $output_file")
    
    return df
end

# Convenience functions
"""
Get default baseline parameters (matching calibrate_models.jl).
"""
function default_baseline()
    return [
        0.8,      # beta
        4.0,      # k_E
        3.0,      # delta_I
        10.0,     # p_V
        1.0,      # c_V
        1e-5,     # D_V
        1.0,      # k_PV
        0.03,     # p_F
        0.01,     # c_F
        1e-1,     # D_F
        0.001,    # delta_FV
        1e2,      # k_FV
        2.0,      # k_IF
        1000.0,   # a_F
        1e1       # K_F
    ]
end

"""
STUB
Load best parameters from a calibration result file.
"""
function load_best_params(filepath::String)
    # This would parse a calibration result file
    # For now, return default
    @warn "load_best_params not fully implemented, returning defaults"
    return default_baseline()
end

# Main entry point
function main(;
              θ_base::Vector{Float64}=default_baseline(),
              param_indices::Vector{Int}=collect(1:15),
              n_points::Int=15,
              log10_range::Float64=1.0,
              output_file::String="Results/sensitivity_sweeps.csv")
    
    run_sensitivity_sweeps(θ_base;
                           param_indices=param_indices,
                           n_points=n_points,
                           log10_range=log10_range,
                           output_file=output_file)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

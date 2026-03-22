"""
Computes AICc (corrected Akaike Information Criterion) from calibration results.
Reads NLL and nCal from calibration_summary.txt and generates model comparison tables.

AICc = 2*NLL + 2*k + (2*k*(k+1))/(n-k-1)

where:
- NLL = negative log-likelihood
- k = number of calibrated parameters (nCal)
- n = number of observations
"""

using Printf
using Dates

"""Number of observations for Virus titer data"""
const N_OBS_VIRUS = 70

"""Number of observations for IFN data"""
const N_OBS_IFN = 60
"""Total number of observations (used for AICc when fitting all)"""
const N_OBS_TOTAL = N_OBS_VIRUS + N_OBS_IFN

# Data structures

struct ModelResult
    model::String
    nCal::Int
    BI::Bool
    BP::Bool
    IC::Bool
    PF::Bool
    NLL::Float64
end

# Parsing functions

"""
Parse calibration_summary.txt and extract model results.
Returns Vector{ModelResult}.
"""
function parse_calibration_summary(filepath::String)
    results = ModelResult[]
    
    lines = readlines(filepath)
    in_ranking_table = false
    
    for line in lines
        # Detect the ranking table section
        if occursin("Rank  Model   nCal", line)
            in_ranking_table = true
            continue
        end
        
        # Skip header separators
        if startswith(line, "----") || isempty(strip(line))
            if in_ranking_table && startswith(line, "----") && length(results) > 0
                # Second dashed line ends the table
                break
            end
            continue
        end
        
        # Parse ranking table rows
        if in_ranking_table
            parts = split(strip(line))
            if length(parts) >= 7 && startswith(parts[2], "M")
                model = parts[2]
                nCal = parse(Int, parts[3])
                BI = parts[4] == "T"
                BP = parts[5] == "T"
                IC = parts[6] == "T"
                PF = parts[7] == "T"
                NLL = parse(Float64, parts[8])
                
                push!(results, ModelResult(model, nCal, BI, BP, IC, PF, NLL))
            end
        end
    end
    
    return results
end

# AICc computation
"""
Compute AICc given NLL, number of parameters k, and number of observations n.

AICc = 2*NLL + 2*k + (2*k*(k+1))/(n-k-1)

Note: Requires n > k + 1 to avoid division issues.
"""
function compute_aicc(NLL::Float64, k::Int, n::Int)
    if n <= k + 1
        @warn "n=$n is too small for k=$k parameters. AICc correction is undefined."
        return Inf
    end
    
    aic = 2.0 * NLL + 2.0 * k
    correction = (2.0 * k * (k + 1)) / (n - k - 1)
    
    return aic + correction
end

"""
Compute AIC (without small-sample correction).
"""
function compute_aic(NLL::Float64, k::Int)
    return 2.0 * NLL + 2.0 * k
end

# Report generation
"""
Generate AICc comparison report for a set of models.

Arguments:
- results: Vector{ModelResult} from parse_calibration_summary
- model_list: Vector of model names to include (e.g., ["M01", "M10", "M16"])
              If empty or nothing, include all models.
- n_obs: Number of observations for AICc calculation
- output_file: Path to write the report (optional)

Returns: String containing the formatted report
"""
function generate_aicc_report(
    results::Vector{ModelResult};
    model_list::Union{Vector{String}, Nothing}=nothing,
    n_obs::Int=N_OBS_VIRUS + N_OBS_IFN,  # Default: Virus + IFN = 95
    output_file::Union{String, Nothing}=nothing
)
    # Filter to requested models
    if model_list !== nothing && !isempty(model_list)
        filtered = filter(r -> r.model in model_list, results)
    else
        filtered = results
    end
    
    if isempty(filtered)
        error("No matching models found")
    end
    
    # Compute AICc for each model
    aicc_data = [(r, compute_aicc(r.NLL, r.nCal, n_obs), compute_aic(r.NLL, r.nCal)) for r in filtered]
    
    # Sort by AICc
    sort!(aicc_data, by=x->x[2])
    
    # Find best AICc
    best_aicc = aicc_data[1][2]
    
    # Build report
    io = IOBuffer()
    
    println(io, "="^100)
    println(io, "AICc Model Comparison Report")
    println(io, "Generated: $(Dates.now())")
    println(io, "="^100)
    println(io)
    println(io, "Configuration:")
    println(io, "  Number of observations (n): $n_obs")
    println(io, "    - Virus observations: $(N_OBS_VIRUS)")
    println(io, "    - IFN observations: $(N_OBS_IFN)")
    if n_obs == N_OBS_VIRUS + N_OBS_IFN
        println(io, "    (Using Virus + IFN only)")
    elseif n_obs == N_OBS_TOTAL
        println(io, "    - T cell observations: $(N_OBS_TCELLS)")
        println(io, "    (Using all observations)")
    end
    println(io)
    println(io, "Mechanism Key:")
    println(io, "  BI = IFN blocks infection (k_IF)")
    println(io, "  BP = IFN blocks viral production (k_PV)")
    println(io, "  IC = IFN increases viral clearance (delta_FV)")
    println(io, "  PF = IFN positive feedback (a_F)")
    println(io)
    println(io, "Models included: $(length(filtered))")
    if model_list !== nothing && !isempty(model_list)
        println(io, "  Subset: $(join(sort(model_list), ", "))")
    else
        println(io, "  (All models)")
    end
    println(io)
    
    # Header
    println(io, "-"^100)
    @printf(io, "%-6s  %4s  %3s %3s %3s %3s  %12s  %10s  %10s  %10s\n",
            "Rank", "Model", "BI", "BP", "IC", "PF", "NLL", "AIC", "AICc", "ΔAICc")
    println(io, "-"^100)
    
    # Data rows
    for (rank, (r, aicc, aic)) in enumerate(aicc_data)
        delta = aicc - best_aicc
        bi_str = r.BI ? "T" : "·"
        bp_str = r.BP ? "T" : "·"
        ic_str = r.IC ? "T" : "·"
        pf_str = r.PF ? "T" : "·"
        
        @printf(io, "%4d    %-4s  %3s %3s %3s %3s  %12.4f  %10.4f  %10.4f  %10.4f\n",
                rank, r.model, bi_str, bp_str, ic_str, pf_str, r.NLL, aic, aicc, delta)
    end
    
    println(io, "-"^100)
    println(io)
    
    # Interpretation guide
    println(io, "Interpretation Guide:")
    println(io, "  ΔAICc = 0-2:    Substantial support (essentially as good as best model)")
    println(io, "  ΔAICc = 2-4:    Strong support")
    println(io, "  ΔAICc = 4-7:    Considerably less support")
    println(io, "  ΔAICc = 7-10:   Essentially no support")
    println(io, "  ΔAICc > 10:     No support")
    println(io)
    
    # Model weights (Akaike weights)
    println(io, "Akaike Weights (model probabilities):")
    println(io, "-"^50)
    
    # Compute weights: w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    deltas = [aicc - best_aicc for (_, aicc, _) in aicc_data]
    exp_terms = exp.(-0.5 .* deltas)
    weights = exp_terms ./ sum(exp_terms)
    
    cumulative = 0.0
    for (i, ((r, aicc, _), w)) in enumerate(zip(aicc_data, weights))
        cumulative += w
        @printf(io, "  %-4s: w = %6.4f  (cumulative: %6.4f)\n", r.model, w, cumulative)
    end
    println(io, "-"^50)
    println(io)
    
    # Best model summary
    best_model = aicc_data[1][1]
    println(io, "Best Model: $(best_model.model)")
    println(io, "  Mechanisms: ", join([
        best_model.BI ? "blocks infection" : nothing,
        best_model.BP ? "blocks production" : nothing,
        best_model.IC ? "increases clearance" : nothing,
        best_model.PF ? "positive feedback" : nothing
    ] |> x -> filter(!isnothing, x), ", "))
    println(io, "  Parameters: $(best_model.nCal)")
    println(io, "  NLL: $(best_model.NLL)")
    println(io, "  AICc: $(aicc_data[1][2])")
    
    report = String(take!(io))
    
    # Write to file if specified
    if output_file !== nothing
        open(output_file, "w") do f
            write(f, report)
        end
        println("Report written to: $output_file")
    end
    
    return report
end

# Convenience functions
"""
Compute AICc for all models and write to file.
"""
function compute_all_models_aicc(;
    summary_file::String="Results/calibration_runs/calibration_summary.txt",
    output_file::String="Results/calibration_runs/aicc_all_models.txt",
    n_obs::Int=N_OBS_VIRUS + N_OBS_IFN
)
    results = parse_calibration_summary(summary_file)
    report = generate_aicc_report(results; n_obs=n_obs, output_file=output_file)
    println(report)
    return report
end

"""
Compute AICc for a subset of models and write to file.
"""
function compute_subset_aicc(
    model_list::Vector{String};
    summary_file::String="Results/calibration_runs/calibration_summary.txt",
    output_file::Union{String, Nothing}=nothing,
    n_obs::Int=N_OBS_VIRUS + N_OBS_IFN
)
    results = parse_calibration_summary(summary_file)
    report = generate_aicc_report(results; model_list=model_list, n_obs=n_obs, output_file=output_file)
    println(report)
    return report
end

# Main entry point

function main()
    println("="^60)
    println("AICc Computation for TIVF5 Model Variants")
    println("="^60)
    println()
    
    summary_file = joinpath(@__DIR__, "..", "Results", "calibration_runs", "calibration_summary.txt")
    
    if !isfile(summary_file)
        error("Calibration summary not found: $summary_file")
    end
    
    # Parse results
    results = parse_calibration_summary(summary_file)
    println("Loaded $(length(results)) model results from calibration_summary.txt")
    println()
    
    # Compute for all models
    println("="^60)
    println("ALL MODELS (n = $(N_OBS_VIRUS + N_OBS_IFN) observations)")
    println("="^60)
    
    output_all = joinpath(@__DIR__, "..", "Results", "aicc_all_models.txt")
    report_all = generate_aicc_report(results; 
        n_obs=N_OBS_VIRUS + N_OBS_IFN, 
        output_file=output_all
    )
    println(report_all)
    
    
    # Example: Reduced set (models which contain infection only.)
    println("\n")
    println("="^60)
    println("INFECTION CONTAINED MODELS ONLY")
    println("="^60)
    
    containment_models = ["M11", "M12", "M15", "M16"]
    output_containment = joinpath(@__DIR__, "..", "Results", "aicc_containment_models.txt")
    report_containment = generate_aicc_report(results;
        model_list=containment_models,
        n_obs=N_OBS_VIRUS + N_OBS_IFN,
        output_file=output_containment
    )
    println(report_containment)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

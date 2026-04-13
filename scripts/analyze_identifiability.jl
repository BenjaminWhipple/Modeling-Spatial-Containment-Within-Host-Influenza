"""
Analyze local parameter identifiability from a precomputed FIM (saved by compute_FIM.jl).

This script:
1. Loads the FIM from Results/FIM.npz
2. Performs eigendecomposition to identify structurally and practically
   unidentifiable parameter directions
3. Reports per-parameter identifiability and pairwise collinearity
4. Computes the marginal FIM (Schur complement) assuming D_V and D_F are known,
   and compares standard errors / identifiability with and without that assumption
"""

using Printf
using LinearAlgebra
using NPZ

# Parameter names (must match ordering used in compute_FIM.jl)
const PARAM_NAMES = [
    "beta",      # 1
    "k_E",       # 2
    "delta_I",   # 3
    "p_V",       # 4
    "c_V",       # 5
    "D_V",       # 6
    "k_PV",      # 7
    "p_F",       # 8
    "c_F",       # 9
    "D_F",       # 10
    "delta_FV",  # 11
    "k_FV",      # 12
    "k_IF",      # 13
    "a_F",       # 14
    "K_F"        # 15
]

# Load FIM from NPZ
"""
    load_fim(path)

Load the FIM and associated data from an NPZ file produced by compute_FIM.jl.
Returns a named tuple with the FIM, parameter values, free mask, etc.
"""
function load_fim(path::String)
    data = NPZ.npzread(path)

    FIM = data["FIM"]
    ϕ   = vec(data["phi"]) # Love the mathsym in Julia.
    θ   = vec(data["theta"])

    free_mask = if haskey(data, "free_mask")
        BitVector(data["free_mask"] .> 0.5)
    else
        trues(length(ϕ))
    end

    return (; FIM, ϕ, θ, free_mask)
end

# Identifiability analysis via eigendecomposition
"""
    eigendecomposition_analysis(FIM, names; tol_zero=1e-6, tol_weak=1e-2)

Classify identifiability of parameter directions from FIM eigenvalues.

Returns named tuple:
  - eigenvalues, eigenvectors (sorted ascending)
  - rank: numerical rank
  - unidentifiable / weakly_identifiable / identifiable indices
  - identifiable_mask: per-parameter identifiability (Bool vector)
"""
function eigendecomposition_analysis(FIM::Matrix{Float64},
                                     names::Vector{String};
                                     tol_zero::Float64=1e-6,
                                     tol_weak::Float64=1e-2)
    n = length(names)
    F = Symmetric(FIM)
    eig = eigen(F)
    λ = eig.values
    V = eig.vectors

    fim_rank = count(abs.(λ) .> tol_zero)

    unident_idx  = findall(abs.(λ) .<= tol_zero)
    weak_idx     = findall(x -> tol_zero < abs(x) <= tol_weak, λ)
    ident_idx    = findall(abs.(λ) .> tol_weak)

    # Per-parameter identifiability:
    # flag a parameter as unidentifiable if it has a large component
    # in any unidentifiable eigenvector
    identifiable_mask = trues(n)
    for k in unident_idx
        v = V[:, k]
        dominant = findall(abs.(v) .> 0.3)
        identifiable_mask[dominant] .= false
    end

    return (;
        eigenvalues  = λ,
        eigenvectors = V,
        rank         = fim_rank,
        unident_idx,
        weak_idx,
        ident_idx,
        identifiable_mask
    )
end

# Pairwise collinearity index
"""
    pairwise_collinearity(FIM, names)

Compute collinearity index for every pair of parameters.

For a pair (i,j), the collinearity index is
    δ = 1 / √(min eigenvalue of FIM[[i,j],[i,j]])
Large values (>100) indicate the pair is nearly unidentifiable together.
"""
function pairwise_collinearity(FIM::Matrix{Float64}, names::Vector{String})
    n = length(names)
    result = Dict{Tuple{String,String}, Float64}()
    for i in 1:n
        for j in (i+1):n
            sub = FIM[[i,j], [i,j]]
            λ_min = minimum(eigvals(Symmetric(sub)))
            ci_val = λ_min > 0 ? 1.0 / sqrt(λ_min) : Inf
            result[(names[i], names[j])] = ci_val
        end
    end
    return result
end

# Marginal FIM via Schur complement
"""
    marginal_fim(FIM, interest_indices)

Compute the marginal FIM for a subset of parameters (the "interest" set)
by Schur-complementing out the "nuisance" parameters.

Given:
    F = [ F_SS   F_SN ]
        [ F_NS   F_NN ]

the marginal FIM for S is:
    F_marg = F_SS − F_SN · inv(F_NN) · F_NS

This captures the information about S *after accounting for uncertainty in N*.
"""
function marginal_fim(FIM::Matrix{Float64}, interest_indices::Vector{Int})
    n = size(FIM, 1)
    nuisance_indices = setdiff(1:n, interest_indices)

    if isempty(nuisance_indices)
        return FIM[interest_indices, interest_indices]
    end

    F_SS = FIM[interest_indices, interest_indices]
    F_SN = FIM[interest_indices, nuisance_indices]
    F_NS = FIM[nuisance_indices, interest_indices]
    F_NN = FIM[nuisance_indices, nuisance_indices]

    F_NN_inv = try
        inv(F_NN)
    catch
        @warn "F_NN singular — using pseudoinverse for Schur complement"
        pinv(F_NN)
    end

    return F_SS - F_SN * F_NN_inv * F_NS
end

# Standard errors from a FIM (helper)
function se_from_fim(FIM::Matrix{Float64})
    cov = try
        inv(FIM)
    catch
        pinv(FIM)
    end
    return sqrt.(max.(diag(cov), 0.0)), cov
end

# Print: eigendecomposition report
function print_eigen_report(ident, names::Vector{String}; label::String="Full FIM")
    n = length(names)
    println("\n" * "="^90)
    println("Eigendecomposition — $label")
    println("="^90)

    @printf("Rank: %d / %d\n", ident.rank, n)
    if ident.rank < n
        @printf("Rank-deficient — %d direction(s) structurally unidentifiable\n",
                n - ident.rank)
    else
        println("Full rank — all parameter combinations locally identifiable")
    end
    println()

    # Eigenvalues
    println("Eigenvalues (ascending):")
    for (k, ev) in enumerate(ident.eigenvalues)
        flag = if abs(ev) <= 1e-6
            " <- ZERO"
        elseif abs(ev) <= 1e-2
            " <- weak"
        else
            ""
        end
        @printf("  λ_%2d = %12.4e%s\n", k, ev, flag)
    end
    println()

    # Unidentifiable directions
    if !isempty(ident.unident_idx)
        println("Structurally unidentifiable directions:")
        println("-"^70)
        for k in ident.unident_idx
            @printf("  λ_%d = %.4e\n", k, ident.eigenvalues[k])
            print("  Direction: ")
            v = ident.eigenvectors[:, k]
            terms = String[]
            for idx in sortperm(abs.(v), rev=true)
                c = v[idx]
                abs(c) > 0.05 || continue
                push!(terms, @sprintf("%+.3f·%s", c, names[idx]))
            end
            println(join(terms, "  "))
            println()
        end
    end

    # Weakly identifiable directions
    if !isempty(ident.weak_idx)
        println("Weakly identifiable directions:")
        println("-"^70)
        for k in ident.weak_idx
            @printf("  λ_%d = %.4e\n", k, ident.eigenvalues[k])
            print("  Direction: ")
            v = ident.eigenvectors[:, k]
            terms = String[]
            for idx in sortperm(abs.(v), rev=true)
                c = v[idx]
                abs(c) > 0.05 || continue
                push!(terms, @sprintf("%+.3f·%s", c, names[idx]))
            end
            println(join(terms, "  "))
            println()
        end
    end

    # Per-parameter summary
    println("Per-parameter identifiability:")
    println("-"^50)
    for i in 1:n
        status = ident.identifiable_mask[i] ? " identifiable" : "  UNIDENTIFIABLE"
        @printf("  %-10s  %s\n", names[i], status)
    end
    println("="^90)
end

# Print: collinearity report
function print_collinearity_report(collin::Dict, names::Vector{String};
                                   top_k::Int=10)
    println("\n" * "="^70)
    println("Top $top_k most collinear parameter pairs")
    println("="^70)
    @printf("%-10s %-10s %15s\n", "Param 1", "Param 2", "Collin. Index")
    println("-"^40)

    sorted = sort(collect(collin), by=x -> -x[2])
    for (k, ((p1, p2), ci_val)) in enumerate(sorted)
        k > top_k && break
        flag = ci_val > 100.0 ? "  " : ""
        @printf("%-10s %-10s %15.4e%s\n", p1, p2, ci_val, flag)
    end
    println("="^70)
end

# Print: marginal vs conditional comparison
"""
Print side-by-side comparison of standard errors when nuisance parameters
are assumed known (conditional) versus uncertain (marginal / Schur complement).
"""
function print_marginal_comparison(FIM::Matrix{Float64},
                                   interest_indices::Vector{Int},
                                   names::Vector{String},
                                   ϕ::Vector{Float64};
                                   known_param_label::String="",
                                   tol_zero::Float64=1e-6)
    n = length(interest_indices)
    nuisance_indices = setdiff(1:size(FIM, 1), interest_indices)

    interest_names  = names[interest_indices]
    nuisance_names  = names[nuisance_indices]
    ϕ_interest      = ϕ[interest_indices]

    println("\n" * "="^90)
    println("Marginal FIM Analysis (Schur Complement)")
    if !isempty(known_param_label)
        println("Known (fixed) parameters: $known_param_label")
    end
    println("="^90)
    println("Parameters of interest:  ", join(interest_names, ", "))
    println("Assumed known (nuisance): ", join(nuisance_names, ", "))
    println()

    # Conditional analysis (submatrix: assumes nuisance known exactly)
    F_cond = FIM[interest_indices, interest_indices]
    se_cond, _ = se_from_fim(F_cond)

    ident_cond = eigendecomposition_analysis(F_cond, interest_names;
                                             tol_zero=tol_zero)

    # Marginal analysis (Schur complement: nuisance uncertain)
    F_marg = marginal_fim(FIM, interest_indices)
    se_marg, _ = se_from_fim(F_marg)

    ident_marg = eigendecomposition_analysis(F_marg, interest_names;
                                              tol_zero=tol_zero)

    # Table
    println("-"^90)
    @printf("%-10s %12s %12s %15s %15s %10s\n",
            "Param", "θ (linear)", "ϕ (log10)",
            "SE(cond.)", "SE(marginal)", "Inflation")
    println("-"^90)

    for i in 1:n
        θ_i = 10.0^ϕ_interest[i]
        inflation = se_marg[i] / max(se_cond[i], 1e-30)
        flag = inflation > 5.0 ? "  " : ""
        @printf("%-10s %12.4e %12.4f %15.4e %15.4e %9.2fx%s\n",
                interest_names[i], θ_i, ϕ_interest[i],
                se_cond[i], se_marg[i], inflation, flag)
    end
    println("-"^90)

    # Eigenvalue comparison
    λ_cond = ident_cond.eigenvalues
    λ_marg = ident_marg.eigenvalues

    println("\nEigenvalues (ascending):")
    @printf("%-8s %15s %15s\n", "", "Conditional", "Marginal")
    for k in 1:n
        flag_c = abs(λ_cond[k]) <= tol_zero ? " ←ZERO" : ""
        flag_m = abs(λ_marg[k]) <= tol_zero ? " ←ZERO" : ""
        @printf("  λ_%2d   %14.4e%s %14.4e%s\n", k, λ_cond[k], flag_c, λ_marg[k], flag_m)
    end

    # Rank comparison
    @printf("\nConditional rank: %d / %d\n", ident_cond.rank, n)
    @printf("Marginal rank:    %d / %d\n", ident_marg.rank, n)

    if ident_marg.rank < ident_cond.rank
        println(" Identifiability is LOST when accounting for nuisance parameter uncertainty")
    elseif ident_marg.rank == ident_cond.rank
        println(" Rank is preserved — identifiability structure unchanged")
    end

    # Per-parameter comparison
    println("\nPer-parameter identifiability comparison:")
    println("-"^60)
    @printf("%-10s %18s %18s\n", "Param", "Conditional", "Marginal")
    println("-"^60)
    for i in 1:n
        s_c = ident_cond.identifiable_mask[i] ? "T identifiable" : "F UNIDENTIFIABLE"
        s_m = ident_marg.identifiable_mask[i] ? "T identifiable" : "F UNIDENTIFIABLE"
        changed = (ident_cond.identifiable_mask[i] != ident_marg.identifiable_mask[i]) ? " <- CHANGED" : ""
        @printf("%-10s %18s %18s%s\n", interest_names[i], s_c, s_m, changed)
    end

    # Weakly/un-identifiable directions in marginal FIM
    if !isempty(ident_marg.unident_idx) || !isempty(ident_marg.weak_idx)
        println()
        print_eigen_report(ident_marg, interest_names;
                           label="Marginal FIM (after Schur complement)")
    end

    println("="^90)
    return F_marg, ident_marg
end

# Main
function main(;
    fim_path::String = "Results/FIM.npz",
    tol_zero::Float64 = 1e-6,
    tol_weak::Float64 = 1e-2,
    top_collinear::Int = 10
)
    # Load FIM
    println("Loading FIM from: $fim_path")
    data = load_fim(fim_path)
    FIM  = data.FIM
    ϕ    = data.ϕ
    θ    = data.θ

    # Determine which parameter names correspond to the FIM rows
    free_indices = findall(data.free_mask)
    names = PARAM_NAMES[free_indices]
    n = length(names)

    println("FIM size: $n × $n")
    println("Parameters: ", join(names, ", "))
    println()

    # Full FIM identifiability analysis
    println("Section 1: Full FIM Identifiability Analysis")
    ident_full = eigendecomposition_analysis(FIM, names;
                                             tol_zero=tol_zero, tol_weak=tol_weak)
    print_eigen_report(ident_full, names; label="Full FIM (all free parameters)")

    # Collinearity
    collin = pairwise_collinearity(FIM, names)
    print_collinearity_report(collin, names; top_k=top_collinear)

    # Full CIs for reference
    se_full, cov_full = se_from_fim(FIM)
    println("\n" * "-"^90)
    println("Full FIM — Standard errors and 95% CIs (log10 space)")
    println("-"^90)
    @printf("%-10s %12s %12s %12s %12s %12s\n",
            "Param", "θ (linear)", "ϕ (log10)", "SE(log10)", "CI_lower", "CI_upper")
    println("-"^90)
    for i in 1:n
        ci_lo = 10.0^(ϕ[i] - 1.96*se_full[i])
        ci_hi = 10.0^(ϕ[i] + 1.96*se_full[i])
        @printf("%-10s %12.4e %12.4f %12.4f %12.4e %12.4e\n",
                names[i], θ[i], ϕ[i], se_full[i], ci_lo, ci_hi)
    end
    println("-"^90)

    # Marginal analysis: assuming D_V and D_F are known
    println("\n\n Section 2: Identifiability assuming D_V and D_F are known")

    # Find D_V (index 6) and D_F (index 10) positions in the FIM
    # The FIM rows correspond to free_indices, so we need to map
    dv_pos = findfirst(==(6), free_indices)   # position of D_V in FIM
    df_pos = findfirst(==(10), free_indices)  # position of D_F in FIM

    known_positions = Int[]
    known_labels = String[]

    if !isnothing(dv_pos)
        push!(known_positions, dv_pos)
        push!(known_labels, "D_V")
    else
        println("  Note: D_V was not in the free parameter set (already fixed)")
    end
    if !isnothing(df_pos)
        push!(known_positions, df_pos)
        push!(known_labels, "D_F")
    else
        println("  Note: D_F was not in the free parameter set (already fixed)")
    end

    if isempty(known_positions)
        println("  Both D_V and D_F are already fixed — nothing to compare.")
    else
        # Interest set = everything except D_V and D_F
        interest_positions = setdiff(1:n, known_positions)

        F_marg, ident_marg = print_marginal_comparison(
            FIM, interest_positions, names, ϕ;
            known_param_label=join(known_labels, ", "),
            tol_zero=tol_zero
        )
    end

    println("\n\nDone.")
    return ident_full
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

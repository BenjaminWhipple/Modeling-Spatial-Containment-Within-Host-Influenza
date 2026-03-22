"""
Compute the Fisher Information Matrix (FIM) for the TIVF5_2D model
using the Gauss-Newton approximation: FIM ≈ Jᵀ J
The Jacobian J of the normalized residual vector r(ϕ) is computed
via centered finite differences in log10 parameter space.
Since FIM = Jᵀ J, it is guaranteed positive semi-definite.

Outputs:
- FIM and Jacobian as .npz
- Parameter-level confidence intervals from inverse FIM
- Summary report to stdout
"""

using Printf
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using NPZ

# model
include("../src/models/TIVF5_2D.jl")
using .TIVF5_2D

# positivity clamp
include("../src/pde_utils.jl")

# Parameter names (must match ordering in set_params!)
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

# Residual computation from a TIVFGPNLLLoss, I should change that name.
"""
    compute_residuals(L::TIVFGPNLLLoss, ϕ::Vector{Float64})

Compute the normalized residual vector r such that the GP NLL ≈ 0.5‖r‖² + const.

The residuals are:
    rV = LV⁻¹ (yV - μV)    (Virus GP, length nV)
    rF = √wF · LF⁻¹ (yF - μF)   (IFN GP, length nF)

where LV, LF are the lower-Cholesky factors of the GP covariance matrices.

Returns `nothing` if the solve fails (penalty case).
"""
function compute_residuals(L, ϕ::Vector{Float64})
    prob = L.prob_template

    try
        TIVF5_2D.set_params!(prob, ϕ; params_in_log10=L.params_in_log10)
    catch
        return nothing
    end

    tV = vec(L.gpV.t_eval)
    tF = vec(L.gpF.t_eval)
    t_all = TIVF5_2D._union_times(tV, tF)

    prob2 = remake(prob; tspan=(t_all[1], t_all[end]))

    sol = try
        solve(prob2, L.alg;
            reltol=L.reltol,
            abstol=L.abstol,
            dtmax=L.dtmax,
            save_everystep=false,
            saveat=t_all,
            dense=false,
            callback=L.callback
        )
    catch
        return nothing
    end

    sol.retcode == ReturnCode.Success || return nothing

    yV_all = try; L.obsV(sol, L.N); catch; return nothing; end
    yF_all = try; L.obsF(sol, L.N); catch; return nothing; end

    idxV = try; TIVF5_2D._indices_in_grid(t_all, tV); catch; return nothing; end
    idxF = try; TIVF5_2D._indices_in_grid(t_all, tF); catch; return nothing; end

    yV = TIVF5_2D._take_at(yV_all, idxV)
    yF = TIVF5_2D._take_at(yF_all, idxF)

    for v in yV; isfinite(v) || return nothing; end
    for v in yF; isfinite(v) || return nothing; end

    # Normalized residuals: r = L⁻¹ (y - μ)  where L is Cholesky factor
    rV = L.gpV.chol.L \ (yV .- L.gpV.mu)
    rF = sqrt(L.wF) .* (L.gpF.chol.L \ (yF .- L.gpF.mu))

    return vcat(rV, rF)
end

# Jacobian-based FIM (Gauss-Newton approximation)
"""
    jacobian_fim(residual_fn, ϕ_free, param_names; h=1e-4, verbose=true)

Compute the Gauss-Newton FIM approximation: FIM = Jᵀ J

where J[i,k] = ∂r[i]/∂ϕ[k] is the Jacobian of the normalized residual
vector with respect to log10 parameters, computed via centered finite
differences.

Guaranteed positive semi-definite by construction.

Only requires 2n function evaluations (vs. O(n²) for the full Hessian).
"""
function jacobian_fim(residual_fn, ϕ_free::Vector{Float64},
                      param_names::Vector{String};
                      h::Float64=1e-4,
                      verbose::Bool=true)
    n = length(ϕ_free)

    # Evaluate residuals at baseline
    r0 = residual_fn(ϕ_free)
    if isnothing(r0)
        error("Residual evaluation failed at baseline parameters — cannot compute FIM.")
    end
    m = length(r0)

    if verbose
        @printf("  Residual vector length: %d\n", m)
        @printf("  Number of parameters:   %d\n", n)
        @printf("  FD step size (log10):   h = %.2e\n", h)
        @printf("  Total FD evaluations:   %d  (2 × %d params)\n", 2*n, n)
        @printf("  Baseline ‖r‖² / 2 = %.8e\n", 0.5 * dot(r0, r0))
        println()
    end

    # Build Jacobian column by column via centered differences
    J = zeros(Float64, m, n)
    for k in 1:n
        xp = copy(ϕ_free); xp[k] += h
        xm = copy(ϕ_free); xm[k] -= h

        rp = residual_fn(xp)
        rm = residual_fn(xm)

        # If a perturbed evaluation fails, use one-sided difference
        if isnothing(rp) && isnothing(rm)
            @warn "Both ±h evaluations failed for parameter $(param_names[k]) — column set to zero"
            J[:, k] .= 0.0
        elseif isnothing(rp)
            @warn "Forward evaluation failed for $(param_names[k]) — using backward difference"
            J[:, k] = (r0 .- rm) ./ h
        elseif isnothing(rm)
            @warn "Backward evaluation failed for $(param_names[k]) — using forward difference"
            J[:, k] = (rp .- r0) ./ h
        else
            J[:, k] = (rp .- rm) ./ (2.0 * h)
        end

        if verbose
            col_norm = norm(J[:, k])
            @printf("  Column %2d / %2d  %-10s  ‖∂r/∂ϕ‖ = %.4e\n",
                    k, n, param_names[k], col_norm)
        end
    end

    # Gauss-Newton FIM: Jᵀ J (symmetric PSD by construction)
    FIM = J' * J

    return FIM, J, r0
end

# Confidence intervals from FIM
"""
    confidence_intervals_from_fim(FIM, ϕ; α=0.05)

Compute approximate confidence intervals from the Fisher Information Matrix.

The FIM is the Hessian of the NLL in log10 space, so the covariance
in log10 space is approximately inv(FIM).

Returns named tuple:
  - `cov_log10`: covariance matrix in log10 space
  - `se_log10`: standard errors in log10 space
  - `ci_lower_log10`, `ci_upper_log10`: CI bounds in log10 space
  - `ci_lower`, `ci_upper`: CI bounds in linear space (10^bounds)
  - `invertible`: whether the FIM was invertible
"""
function confidence_intervals_from_fim(FIM::Matrix{Float64}, ϕ::Vector{Float64};
                                       α::Float64=0.05)
    n = length(ϕ)
    z = 1.96  # for 95% CI (normal approximation)
    if α != 0.05
        # Use quantile from normal distribution
        # For common values: α=0.01 -> z≈2.576, α=0.10 -> z≈1.645
        z = -log(α/2) * 0.6  # rough approximation; exact would need Distributions.jl
        # Better: z = quantile(Normal(), 1-α/2) if Distributions.jl were available
        # For now just use the lookup:
        if α ≈ 0.01;  z = 2.576
        elseif α ≈ 0.05; z = 1.96
        elseif α ≈ 0.10; z = 1.645
        end
    end

    invertible = true
    cov_log10 = try
        inv(FIM)
    catch e
        @warn "FIM is singular or near-singular, using pseudoinverse" exception=e
        invertible = false
        pinv(FIM)
    end

    # Standard errors in log10 space
    se_log10 = sqrt.(max.(diag(cov_log10), 0.0))

    ci_lower_log10 = ϕ .- z .* se_log10
    ci_upper_log10 = ϕ .+ z .* se_log10

    ci_lower = 10.0 .^ ci_lower_log10
    ci_upper = 10.0 .^ ci_upper_log10

    return (;
        cov_log10,
        se_log10,
        ci_lower_log10, ci_upper_log10,
        ci_lower, ci_upper,
        invertible
    )
end

# Save results to NPZ
function save_fim_npz(filepath::String, FIM::Matrix{Float64},
                      ϕ::Vector{Float64}, ci;
                      param_names::Vector{String}=PARAM_NAMES,
                      free_mask::Union{Nothing,BitVector}=nothing,
                      J::Union{Nothing,Matrix{Float64}}=nothing,
                      r0::Union{Nothing,Vector{Float64}}=nothing)
    data = Dict{String, Any}(
        "FIM"              => FIM,
        "phi"              => ϕ,
        "theta"            => 10.0 .^ ϕ,
        "cov_log10"        => ci.cov_log10,
        "se_log10"         => ci.se_log10,
        "ci_lower_log10"   => ci.ci_lower_log10,
        "ci_upper_log10"   => ci.ci_upper_log10,
        "ci_lower"         => ci.ci_lower,
        "ci_upper"         => ci.ci_upper,
    )
    if !isnothing(free_mask)
        data["free_mask"] = Float64.(free_mask)
    end
    if !isnothing(J)
        data["jacobian"] = J
    end
    if !isnothing(r0)
        data["residuals"] = r0
    end
    npzwrite(filepath, data)
    println("Saved FIM data to: $filepath")
end

# Print summary report
function print_report(ϕ::Vector{Float64}, FIM::Matrix{Float64}, ci;
                      param_names::Vector{String}=PARAM_NAMES,
                      free_indices::Union{Nothing,Vector{Int}}=nothing)
    n = length(ϕ)
    names = isnothing(free_indices) ? param_names : param_names[free_indices]

    println("\n" * "="^90)
    println("Fisher Information Matrix — Confidence Interval Report")
    println("="^90)

    # FIM condition number
    eigvals_fim = eigvals(Symmetric(FIM))
    pos_eigvals = filter(e -> e > 0, eigvals_fim)
    if length(pos_eigvals) > 0
        cond_num = maximum(pos_eigvals) / minimum(pos_eigvals)
        @printf("FIM condition number: %.4e\n", cond_num)
    else
        println("FIM condition number: ∞ (no positive eigenvalues)")
    end
    @printf("FIM invertible: %s\n", ci.invertible ? "yes" : "no (used pseudoinverse)")
    println()

    # Eigenvalue summary
    println("FIM eigenvalues:")
    for (k, ev) in enumerate(sort(eigvals_fim))
        @printf("  λ_%d = %.4e\n", k, ev)
    end
    println()

    # Parameter table
    println("-"^90)
    @printf("%-10s %12s %12s %12s %12s %12s\n",
            "Param", "θ (linear)", "ϕ (log10)", "SE(log10)", "CI_lower", "CI_upper")
    println("-"^90)

    for i in 1:n
        θ_i = 10.0^ϕ[i]
        @printf("%-10s %12.4e %12.4f %12.4f %12.4e %12.4e\n",
                names[i], θ_i, ϕ[i], ci.se_log10[i],
                ci.ci_lower[i], ci.ci_upper[i])
    end
    println("-"^90)

    # Correlation matrix
    D = Diagonal(1.0 ./ max.(ci.se_log10, 1e-30))
    corr = D * ci.cov_log10 * D
    println("\nCorrelation matrix (log10 space):")
    @printf("%-10s", "")
    for j in 1:min(n, 15)
        @printf(" %8s", names[j][1:min(end,8)])
    end
    println()
    for i in 1:n
        @printf("%-10s", names[i])
        for j in 1:n
            @printf(" %8.3f", corr[i,j])
        end
        println()
    end
    println("="^90)
end

# Main
function main(;
    output_dir::String = "Results",
    # Indices of free parameters (1-based). nothing = all 15.
    free_param_indices::Union{Nothing,Vector{Int}} = nothing,
    h::Float64 = 1e-4,     # finite-difference step in log10 space
    α::Float64 = 0.05      # confidence level (0.05 = 95% CI)
)
    isdir(output_dir) || mkpath(output_dir)

    nx, ny = 32, 32

    # Parameter set to evaluate at. Eventually I should shift this to be more programatic.
    pars = TIVF5_2D.Params(;
        beta=8.000000e-02,
        k_E=3.177313e+00,
        delta_I=4.147009e+00,
        k_IF=1.000000e+12,
        p_V=2.928123e+01,
        c_V=1.000000e-01,
        D_V=1.000000e-06,
        k_PV=6.659183e+00,
        p_F=6.066115e-02,
        c_F=1.000000e-03,
        D_F=9.463308e-01,
        a_F=9.872138e+03,
        K_F=1.963406e+01,
        delta_FV=3.506720e-03,
        k_FV=9.389691e+02,
        eps_diff=1e-6,
        delta_smooth=1e-6
    )

    # Build problem
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

    # Build loss function
    loss = TIVF5_2D.make_tivf_gp_loss(prob, N;
        gp_path_V="src/data/gp_parameters/Toapanta_Virus_gp.npz",
        gp_path_F="src/data/gp_parameters/IFN_gp.npz",
        abstol=abstol,
        reltol=1e-4,
        dtmax=0.05,
        penalty=1e20,
        params_in_log10=true,
        callback=cb,
        wF=1.0
    )

    # Full parameter vector in log10 space
    ϕ_full = log10.([
        pars.beta, pars.k_E, pars.delta_I,
        pars.p_V, pars.c_V, pars.D_V, pars.k_PV,
        pars.p_F, pars.c_F, pars.D_F, pars.delta_FV, pars.k_FV,
        pars.k_IF, pars.a_F, pars.K_F
    ])

    # Determine which parameters are free
    free_idx = isnothing(free_param_indices) ? collect(1:15) : free_param_indices
    n_free = length(free_idx)
    free_mask = falses(15)
    free_mask[free_idx] .= true

    println("="^70)
    println("Fisher Information Matrix Computation")
    println("="^70)
    println("Free parameters ($n_free / 15):")
    for i in free_idx
        @printf("  [%2d] %-10s = %.4e  (log10: %.4f)\n",
                i, PARAM_NAMES[i], 10.0^ϕ_full[i], ϕ_full[i])
    end
    @printf("FD step size (log10): h = %.2e\n", h)
    println()

    # Baseline NLL
    nll0 = loss(ϕ_full)
    @printf("Baseline NLL = %.8e\n", nll0)
    if nll0 >= 1e19
        error("Baseline NLL is at penalty level — parameters likely cause solver failure.")
    end

    # Build reduced residual function that only varies free parameters
    function reduced_residuals(ϕ_free::Vector{Float64})
        ϕ = copy(ϕ_full)
        for (k, idx) in enumerate(free_idx)
            ϕ[idx] = ϕ_free[k]
        end
        return compute_residuals(loss, ϕ)
    end

    ϕ_free = ϕ_full[free_idx]
    free_names = PARAM_NAMES[free_idx]

    # Compute Jacobian-based FIM
    println("\nComputing Jacobian of residuals via centered finite differences...")
    println("(Gauss-Newton FIM = Jᵀ J — guaranteed positive semi-definite)\n")
    t_start = time()
    FIM, J, r0 = jacobian_fim(reduced_residuals, ϕ_free, free_names; h=h, verbose=true)
    elapsed = time() - t_start

    @printf("\nJacobian FIM computation took %.2f seconds\n", elapsed)
    @printf("Jacobian shape: %d residuals × %d parameters\n", size(J)...)
    @printf("Baseline 0.5‖r‖² = %.8e  (cf. NLL = %.8e)\n", 0.5*dot(r0,r0), nll0)

    # Sanity check
    neg_diag = findall(diag(FIM) .< 0)
    if !isempty(neg_diag)
        @warn "Unexpected negative diagonal in Jᵀ J (should not happen): indices $neg_diag"
    else
        println("FIM is positive semi-definite (all diag(Jᵀ J) ≥ 0)")
    end

    # Confidence intervals
    ci = confidence_intervals_from_fim(FIM, ϕ_free; α=α)

    # Print report
    print_report(ϕ_free, FIM, ci;
                 param_names=PARAM_NAMES, free_indices=free_idx)

    # Save to NPZ
    npz_path = joinpath(output_dir, "FIM.npz")
    save_fim_npz(npz_path, FIM, ϕ_free, ci;
                 param_names=PARAM_NAMES, free_mask=free_mask, J=J, r0=r0)

    println("\nDone.")
    return FIM, J, ci
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    # Compute FIM for all 15 parameters
    #FIM, J, ci = main()

    # Alternative: compute FIM only for a subset of parameters
    # Skip k_IF (13) which is fixed at 1e12 for M15. c_F (9) is also excluded due to being visibly expected to be non-identifiable.
    FIM, J, ci = main(; free_param_indices=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15]) 
end

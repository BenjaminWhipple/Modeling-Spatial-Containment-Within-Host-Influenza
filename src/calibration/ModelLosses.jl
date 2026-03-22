module ModelLosses

using LinearAlgebra
using DifferentialEquations
using LinearSolve

include("GPLikelihood.jl")
using .GPLikelihood

export TIVGPNLLLoss, make_tiv_gp_loss, default_obs_totalV_log10, set_params!

struct TIVGPNLLLoss{P,G,A,F,C}
    prob_template::P
    N::Int
    gp::G
    alg::A
    reltol::Float64
    abstol::Vector{Float64}
    dtmax::Float64
    penalty::Float64
    obs::F
    params_in_log10::Bool
    callback::C
end

"""
Observation: log10(total V) at saved times.

Robustness:
- sums only positive V contributions (max(u,0))
- logs log10(s + eps), where s >= 0
"""
function default_obs_totalV_log10(sol, N::Int; eps::Float64=1e-12)
    idx_V = (2N + 1):(3N)
    y = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        u = sol.u[k]
        s = 0.0
        for j in idx_V
            v = u[j]
            s += (v > 0.0 ? v : 0.0)
        end
        y[k] = log10(s + eps)
    end
    return y
end

"""
Set parameters into the problem cache in-place.

x is either:
- linear parameters θ
- or log10 parameters φ (if params_in_log10=true), where θ = 10.^φ

Order (length 5):
  1 beta
  2 delta_I
  3 p_V
  4 c_V
  5 D_V
"""
function set_params!(prob, x::AbstractVector{<:Real}; params_in_log10::Bool=false)
    cache = prob.p  # your TIV2D.Cache
    θ = params_in_log10 ? (10.0 .^ Float64.(x)) : Float64.(x)

    length(θ) == 5 || error("set_params!: expected 5 params [beta, delta_I, p_V, c_V, D_V], got $(length(θ))")

    p = cache.p
    p.beta    = θ[1]
    p.delta_I = θ[2]
    p.p_V     = θ[3]
    p.c_V     = θ[4]
    p.D_V     = θ[5]

    # keep derived quantity consistent
    cache.inv_mV1 = 1.0 / (p.mV + 1.0)
    return nothing
end

function (L::TIVGPNLLLoss)(x::AbstractVector{<:Real})
    prob = L.prob_template

    # set parameters (in-place)
    try
        set_params!(prob, x; params_in_log10=L.params_in_log10)
    catch
        return L.penalty
    end

    # ensure t_eval is a Vector
    t_eval = vec(L.gp.t_eval)
    tmin = minimum(t_eval)
    tmax = maximum(t_eval)

    prob2 = remake(prob; tspan=(tmin, tmax))

    sol = try
        solve(prob2, L.alg;
            reltol=L.reltol,
            abstol=L.abstol,
            dtmax=L.dtmax,
            save_everystep=false,
            saveat=t_eval,
            dense=false,
            callback=L.callback
        )
    catch
        return L.penalty
    end

    if sol.retcode != ReturnCode.Success
        return L.penalty
    end

    y = try
        L.obs(sol, L.N)
    catch
        return L.penalty
    end

    @inbounds for i in eachindex(y)
        if !isfinite(y[i])
            return L.penalty
        end
    end

    return L.gp(y)
end

function make_tiv_gp_loss(prob_template, N::Int; gp_path::AbstractString,
        alg = QNDF(linsolve=KLUFactorization(), autodiff=false),
        reltol::Float64 = 1e-4,
        abstol::Vector{Float64},
        dtmax::Float64 = 0.05,
        penalty::Float64 = 1e20,
        obs = default_obs_totalV_log10,
        params_in_log10::Bool = true,
        callback = nothing
    )

    gp = load_gp_likelihood_npz(gp_path)
    return TIVGPNLLLoss(prob_template, N, gp, alg, reltol, abstol, dtmax, penalty, obs, params_in_log10, callback)
end

end # module

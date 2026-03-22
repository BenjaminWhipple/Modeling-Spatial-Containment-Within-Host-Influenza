module TIVF5_2D

using SparseArrays
using SparseDiffTools
using DifferentialEquations
using LinearSolve
using LinearAlgebra

include("../pde_utils.jl")
include("../Domain2D.jl")
using .Domain2D: idx2, make_grid, make_tissue_mask, laplacian_neumann_masked_2d!

include("../calibration/GPLikelihood.jl")
using .GPLikelihood

export Params, make_problem, abstol, nfields, observe
export TIVFGPNLLLoss, make_tivf_gp_loss,
       default_obs_totalV_log10, default_obs_totalF_log10, set_params!

# Parameters
Base.@kwdef mutable struct Params
    # infection / eclipse / infected
    beta::Float64    = 0.5

    # Eclipse progression rate (E1 -> E2 -> I). Erlang-2 mean delay = 2/k_E
    k_E::Float64     = 2.0

    delta_I::Float64 = 1.5

    # IFN reduces infection via factor k_IF/(k_IF + F)
    k_IF::Float64    = 1.0

    # virus
    p_V::Float64 = 1.5
    c_V::Float64 = 5.0
    D_V::Float64 = 1e-3
    mV::Float64  = 1.0

    # IFN reduces viral production via k_PV/(k_PV + F)
    k_PV::Float64 = 1.0

    # interferon
    p_F::Float64  = 1.0
    c_F::Float64  = 1.0
    D_F::Float64  = 1e-3
    mF::Float64   = 1.0

    # IFN positive feedback parameters
    # b(F) = 1 + a_F * F^2 / (K_F^2 + F^2)
    a_F::Float64  = 1.0    # max fold-increase in IFN production (1.0 means up to 2x)
    K_F::Float64  = 1.0    # half-max threshold for positive feedback

    # IFN-driven extra viral clearance
    delta_FV::Float64 = 1.0
    k_FV::Float64     = 1.0

    # diffusion smoothing
    eps_diff::Float64     = 1e-6
    delta_smooth::Float64 = 1e-6
end

# Cache
mutable struct Cache
    nx::Int
    ny::Int
    N::Int
    dx::Float64
    dy::Float64
    dA::Float64

    tissue::BitVector
    n_tissue::Int
    area_tissue::Float64

    p::Params
    inv_mV1::Float64
    inv_mF1::Float64

    # V diffusion buffers
    V_pos::Vector{Float64}
    wV::Vector{Float64}
    lapwV::Vector{Float64}

    # F diffusion buffers
    F_pos::Vector{Float64}
    wF::Vector{Float64}
    lapwF::Vector{Float64}
end

# Interface bits
nfields() = 6

function abstol(N::Int)
    return vcat(
        fill(1e-6,  N),  # T
        fill(1e-10, N),  # E1
        fill(1e-10, N),  # E2
        fill(1e-10, N),  # I
        fill(1e-10, N),  # V
        fill(1e-10, N)   # F
    )
end

# Initial conditions
function make_initial_conditions(nx::Int, ny::Int;
        T0=10.0,
        T_init::Union{Nothing,AbstractMatrix}=nothing,
        V_amp=1.0, sigma=0.03, center=(0.5, 0.5),
        E10=0.0,
        E20=0.0,
        I0=0.0,
        F0=0.0
    )
    x, y, dx, dy, dA, N = make_grid(nx, ny)

    T = if isnothing(T_init)
        fill(T0, nx, ny)
    else
        @assert size(T_init,1)==nx && size(T_init,2)==ny
        Array{Float64}(Float64.(T_init))
    end

    E1 = fill(E10, nx, ny)
    E2 = fill(E20, nx, ny)
    I  = fill(I0, nx, ny)

    cx, cy = center
    V = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        V[i,j] = V_amp * exp(-((x[i]-cx)^2 + (y[j]-cy)^2) / (2sigma^2))
    end

    F = fill(F0, nx, ny)

    y0 = vcat(vec(T), vec(E1), vec(E2), vec(I), vec(V), vec(F))
    return x, y, y0, dx, dy, dA, N
end

# IFN positive feedback function
# b(F) = 1 + a_F * F^2 / (K_F^2 + F^2)
@inline function ifn_feedback(FP::Float64, a_F::Float64, K_F::Float64)
    F2 = FP * FP
    K2 = K_F * K_F
    return 1.0 + a_F * F2 / (K2 + F2)
end

# RHS
function rhs!(dY, Y, cache::Cache, t)
    N = cache.N
    p = cache.p

    Tvec  = @view Y[1:N]
    E1vec = @view Y[N+1:2N]
    E2vec = @view Y[2N+1:3N]
    Ivec  = @view Y[3N+1:4N]
    Vvec  = @view Y[4N+1:5N]
    Fvec  = @view Y[5N+1:6N]

    dT  = @view dY[1:N]
    dE1 = @view dY[N+1:2N]
    dE2 = @view dY[2N+1:3N]
    dI  = @view dY[3N+1:4N]
    dV  = @view dY[4N+1:5N]
    dF  = @view dY[5N+1:6N]

    # Nonlinear diffusion for V
    @inbounds for i in 1:N
        cache.V_pos[i] = pospart(Vvec[i], p.delta_smooth)
    end
    @inbounds for i in 1:N
        cache.wV[i] = (cache.V_pos[i] + p.eps_diff)^(p.mV + 1.0)
    end
    laplacian_neumann_masked_2d!(cache.lapwV, cache.wV, cache.tissue,
                                 cache.nx, cache.ny, cache.dx, cache.dy)

    # Nonlinear diffusion for F
    @inbounds for i in 1:N
        cache.F_pos[i] = pospart(Fvec[i], p.delta_smooth)
    end
    @inbounds for i in 1:N
        cache.wF[i] = (cache.F_pos[i] + p.eps_diff)^(p.mF + 1.0)
    end
    laplacian_neumann_masked_2d!(cache.lapwF, cache.wF, cache.tissue,
                                 cache.nx, cache.ny, cache.dx, cache.dy)

    @inbounds for i in 1:N
        if !cache.tissue[i]
            dT[i]  = 0.0
            dE1[i] = 0.0
            dE2[i] = 0.0
            dI[i]  = 0.0
            dV[i]  = 0.0
            dF[i]  = 0.0
            continue
        end

        TP  = max(Tvec[i],  0.0)
        E1P = max(E1vec[i], 0.0)
        E2P = max(E2vec[i], 0.0)
        IP  = max(Ivec[i],  0.0)
        VP  = max(Vvec[i],  0.0)
        FP  = max(Fvec[i],  0.0)

        # IFN inhibition of infection rate
        inhib_inf = (FP <= 0.0) ? 1.0 : (p.k_IF / (p.k_IF + FP))

        # Infection term now feeds E1 (same as old T->I)
        inf = p.beta * inhib_inf * VP * TP

        # Erlang-2 eclipse progression with rate k_E
        dT[i]  = -inf
        dE1[i] =  inf - p.k_E * E1P
        dE2[i] =  p.k_E * E1P - p.k_E * E2P
        dI[i]  =  p.k_E * E2P - p.delta_I * IP

        # extra V clearance from IFN: delta_FV*(F/(F+k_FV))*V
        fracF = (FP <= 0.0) ? 0.0 : (FP / (FP + p.k_FV))
        extra_clear = p.delta_FV * FP * fracF * VP

        # IFN inhibition of viral production from infected cells
        inhib_prod = (FP <= 0.0) ? 1.0 : (p.k_PV / (p.k_PV + FP))
        prodV = p.p_V * inhib_prod * IP

        dV_diff = p.D_V * cache.inv_mV1 * cache.lapwV[i]
        dV[i] = dV_diff + prodV - p.c_V * VP - extra_clear

        # IFN equation with positive feedback
        # dF/dt = D_F * ΔF + p_F * I * b(F) - c_F * F
        # where b(F) = 1 + a_F * F^2 / (K_F^2 + F^2)
        bF = ifn_feedback(FP, p.a_F, p.K_F)
        dF_diff = p.D_F * cache.inv_mF1 * cache.lapwF[i]
        dF[i] = dF_diff + p.p_F * IP * bF - p.c_F * FP
    end

    return nothing
end

# Jacobian sparsity pattern
function jac_sparsity(nx::Int, ny::Int)
    N  = nx * ny
    sz = 6N
    J  = spzeros(Float64, sz, sz)

    T, E1, E2, I, V, F = 0, 1, 2, 3, 4, 5
    row(field, k) = field*N + k
    col(field, k) = field*N + k

    @inline function stencil5(k::Int, i::Int, j::Int)
        im = (i == 1  ? 2    : i - 1)
        ip = (i == nx ? nx-1 : i + 1)
        jm = (j == 1  ? 2    : j - 1)
        jp = (j == ny ? ny-1 : j + 1)
        return (k,
                idx2(im, j, nx), idx2(ip, j, nx),
                idx2(i, jm, nx), idx2(i, jp, nx))
    end

    for j in 1:ny, i in 1:nx
        k = idx2(i, j, nx)

        s1, s2, s3, s4, s5 = stencil5(k, i, j)

        # dT depends on T, V, F
        J[row(T,k), col(T,k)] = 1.0
        J[row(T,k), col(V,k)] = 1.0
        J[row(T,k), col(F,k)] = 1.0

        # dE1 depends on T, E1, V, F
        J[row(E1,k), col(T,k)]  = 1.0
        J[row(E1,k), col(E1,k)] = 1.0
        J[row(E1,k), col(V,k)]  = 1.0
        J[row(E1,k), col(F,k)]  = 1.0

        # dE2 depends on E1, E2
        J[row(E2,k), col(E1,k)] = 1.0
        J[row(E2,k), col(E2,k)] = 1.0

        # dI depends on E2, I
        J[row(I,k), col(E2,k)] = 1.0
        J[row(I,k), col(I,k)]  = 1.0

        # dV depends on I, F(local), and V(stencil)
        J[row(V,k), col(I,k)] = 1.0
        J[row(V,k), col(F,k)] = 1.0
        for kk in (s1, s2, s3, s4, s5)
            J[row(V,k), col(V,kk)] = 1.0
        end

        # dF depends on I(local), F(local for feedback), and F(stencil for diffusion)
        J[row(F,k), col(I,k)] = 1.0
        for kk in (s1, s2, s3, s4, s5)
            J[row(F,k), col(F,kk)] = 1.0
        end
    end

    return J
end

# Build problem
function make_problem(; nx=32, ny=32, tspan=(0.0, 11.0),
    pars::Params=Params(),
    T0=10.0, T_init=nothing, E10=0.0, E20=0.0, I0=0.0, F0=0.0,
    V_amp=1.0, sigma=0.03, center=(0.5,0.5)
)
    x, y, y0, dx, dy, dA, N = make_initial_conditions(nx, ny;
        T0=T0, T_init=T_init, E10=E10, E20=E20, I0=I0, F0=F0,
        V_amp=V_amp, sigma=sigma, center=center
    )

    Tvec0 = @view y0[1:N]
    tissue, n_tissue = make_tissue_mask(Tvec0)
    area_tissue = n_tissue * dA

    cache = Cache(nx, ny, N, dx, dy, dA,
                  tissue, n_tissue, area_tissue,
                  pars,
                  1.0/(pars.mV + 1.0),
                  1.0/(pars.mF + 1.0),
                  zeros(N), zeros(N), zeros(N),
                  zeros(N), zeros(N), zeros(N))

    Jpat = jac_sparsity(nx, ny)
    colorvec = matrix_colors(Jpat)

    f = ODEFunction(rhs!; jac_prototype=Jpat, colorvec=colorvec)
    prob = ODEProblem(f, y0, tspan, cache)

    return x, y, prob, N
end

# Observation helper (aggregates)
function observe(sol, N::Int)
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)
    idx_V  = (4N+1):(5N)
    idx_F  = (5N+1):(6N)

    Tmass  = [sum(u[idx_T])  for u in sol.u]
    E1mass = [sum(u[idx_E1]) for u in sol.u]
    E2mass = [sum(u[idx_E2]) for u in sol.u]
    Imass  = [sum(u[idx_I])  for u in sol.u]
    Vmass  = [sum(u[idx_V])  for u in sol.u]
    Fmass  = [sum(u[idx_F])  for u in sol.u]
    return (; t=sol.t, Tmass, E1mass, E2mass, Imass, Vmass, Fmass)
end

# Loss (V + F)
struct TIVFGPNLLLoss{P,GV,GF,A,FV,FF,C}
    prob_template::P
    N::Int
    gpV::GV
    gpF::GF
    alg::A
    reltol::Float64
    abstol::Vector{Float64}
    dtmax::Float64
    penalty::Float64
    obsV::FV
    obsF::FF
    params_in_log10::Bool
    callback::C
    wF::Float64
    # Infection containment penalty (nothing = disabled)
    # Penalizes if final aggregate T falls below this fraction of initial T
    min_T_fraction::Union{Nothing,Float64}
    infection_penalty_weight::Float64
    # Infection clearance penalty (nothing = disabled)
    # Penalizes if aggregate infection (E1+E2+I) at final time exceeds this threshold
    max_final_infection::Union{Nothing,Float64}
    clearance_penalty_weight::Float64
end

function default_obs_totalV_log10(sol, N::Int; eps::Float64=1e-12)
    idx_V = (4N + 1):(5N)
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

function default_obs_totalF_log10(sol, N::Int; eps::Float64=1e-12)
    idx_F = (5N + 1):(6N)
    y = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        u = sol.u[k]
        s = 0.0
        for j in idx_F
            f = u[j]
            s += (f > 0.0 ? f : 0.0)
        end
        y[k] = log10(s + eps)
    end
    return y
end

function _union_times(t1::AbstractVector, t2::AbstractVector; atol::Float64=1e-12)
    ts = sort!(vcat(collect(t1), collect(t2)))
    out = Float64[]
    for t in ts
        if isempty(out) || abs(t - out[end]) > atol
            push!(out, float(t))
        end
    end
    return out
end

function _indices_in_grid(t_all::AbstractVector, t_sub::AbstractVector; atol::Float64=1e-12)
    idxs = Vector{Int}(undef, length(t_sub))
    i = 1
    @inbounds for k in eachindex(t_sub)
        tk = t_sub[k]
        while i <= length(t_all) && t_all[i] < tk - atol
            i += 1
        end
        if i > length(t_all) || abs(t_all[i] - tk) > atol
            error("GP time tk=$(tk) not found in union grid within atol=$(atol)")
        end
        idxs[k] = i
    end
    return idxs
end

@inline function _take_at(y_all::AbstractVector{T}, idxs::AbstractVector{Int}) where {T}
    out = Vector{T}(undef, length(idxs))
    @inbounds for k in eachindex(idxs)
        out[k] = y_all[idxs[k]]
    end
    return out
end

"""
Set parameters in-place.

Supports:
- length 14: [beta, k_E, delta_I, p_V, c_V, D_V, k_PV, p_F, c_F, D_F, delta_FV, k_FV, k_IF, a_F]
- length 15: same as above plus [K_F] at the end

(If you pass length 14, K_F remains whatever is already in prob.p.p.K_F.)
"""
function set_params!(prob, x::AbstractVector{<:Real}; params_in_log10::Bool=false)
    cache = prob.p
    θ = params_in_log10 ? (10.0 .^ Float64.(x)) : Float64.(x)

    n = length(θ)
    (n == 14 || n == 15) || error("set_params!: expected 14 or 15 params, got $(n)")

    p = cache.p
    p.beta    = θ[1]
    p.k_E     = θ[2]
    p.delta_I = θ[3]

    p.p_V     = θ[4]
    p.c_V     = θ[5]
    p.D_V     = θ[6]
    p.k_PV    = θ[7]

    p.p_F     = θ[8]
    p.c_F     = θ[9]
    p.D_F     = θ[10]
    p.delta_FV = θ[11]
    p.k_FV    = θ[12]

    p.k_IF    = θ[13]
    p.a_F     = θ[14]

    if n == 15
        p.K_F = θ[15]
    end

    cache.inv_mV1 = 1.0 / (p.mV + 1.0)
    cache.inv_mF1 = 1.0 / (p.mF + 1.0)
    return nothing
end

function (L::TIVFGPNLLLoss)(x::AbstractVector{<:Real})
    prob = L.prob_template

    try
        set_params!(prob, x; params_in_log10=L.params_in_log10)
    catch
        return L.penalty
    end

    tV = vec(L.gpV.t_eval)
    tF = vec(L.gpF.t_eval)
    t_all = _union_times(tV, tF)

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
        return L.penalty
    end

    sol.retcode == ReturnCode.Success || return L.penalty

    yV_all = try
        L.obsV(sol, L.N)
    catch
        return L.penalty
    end
    yF_all = try
        L.obsF(sol, L.N)
    catch
        return L.penalty
    end

    idxV = try
        _indices_in_grid(t_all, tV)
    catch
        return L.penalty
    end
    idxF = try
        _indices_in_grid(t_all, tF)
    catch
        return L.penalty
    end

    yV = _take_at(yV_all, idxV)
    yF = _take_at(yF_all, idxF)

    @inbounds for i in eachindex(yV)
        isfinite(yV[i]) || return L.penalty
    end
    @inbounds for i in eachindex(yF)
        isfinite(yF[i]) || return L.penalty
    end

    nll = L.gpV(yV) + L.wF * L.gpF(yF)

    # Infection containment penalty
    # Penalize if final aggregate T drops below min_T_fraction of initial aggregate T
    if !isnothing(L.min_T_fraction)
        N = L.N
        idx_T = 1:N

        # Initial aggregate T (first saved time point)
        T_initial = sum(@view sol.u[1][idx_T])

        # Final aggregate T
        T_final = sum(@view sol.u[end][idx_T])

        # Fraction of target cells remaining
        frac_remaining = T_final / max(T_initial, 1e-30)

        # Penalize only if fraction remaining drops below the minimum
        if frac_remaining < L.min_T_fraction
            deficit = L.min_T_fraction - frac_remaining
            nll += L.infection_penalty_weight * deficit^2
        end
    end

    # Infection clearance penalty
    # Penalize if aggregate infection (E1+E2+I) at final time exceeds max_final_infection
    if !isnothing(L.max_final_infection)
        N = L.N
        idx_E1 = (N+1):(2N)
        idx_E2 = (2N+1):(3N)
        idx_I  = (3N+1):(4N)

        aggregate_infected = sum(@view sol.u[end][idx_E1]) +
                             sum(@view sol.u[end][idx_E2]) +
                             sum(@view sol.u[end][idx_I])

        if aggregate_infected > L.max_final_infection
            excess = log10(aggregate_infected) - log10(L.max_final_infection)
            nll += L.clearance_penalty_weight * excess^2
        end
    end

    return nll
end

function make_tivf_gp_loss(prob_template, N::Int;
        gp_path_V::AbstractString,
        gp_path_F::AbstractString,
        alg = QNDF(linsolve=KLUFactorization(), autodiff=false),
        reltol::Float64 = 1e-4,
        abstol::Vector{Float64},
        dtmax::Float64 = 0.05,
        penalty::Float64 = 1e20,
        obsV = default_obs_totalV_log10,
        obsF = default_obs_totalF_log10,
        params_in_log10::Bool = true,
        callback = nothing,
        wF::Float64 = 1.0,
        min_T_fraction::Union{Nothing,Float64} = nothing,
        infection_penalty_weight::Float64 = 100.0,
        max_final_infection::Union{Nothing,Float64} = nothing,
        clearance_penalty_weight::Float64 = 100.0
    )

    gpV = load_gp_likelihood_npz(gp_path_V)
    gpF = load_gp_likelihood_npz(gp_path_F)

    return TIVFGPNLLLoss(prob_template, N, gpV, gpF, alg, reltol, abstol, dtmax,
                         penalty, obsV, obsF, params_in_log10, callback, wF,
                         min_T_fraction, infection_penalty_weight,
                         max_final_infection, clearance_penalty_weight)
end

end

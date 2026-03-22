module GPLikelihood

using LinearAlgebra
using NPZ
using Random

export GPLikelihoodNPZ, load_gp_likelihood_npz, nll, sample_paths

"""
A multivariate normal likelihood over a vector y(t_eval).

Loss returned is negative log-likelihood:
  NLL(y) = 0.5 * [ (y-mu)' Σ^{-1} (y-mu) + logdet(Σ) + n*log(2π) ].

- t_eval: vector of times (length n)
- mu: mean vector (length n)
- cov: covariance matrix (nxn)
"""
struct GPLikelihoodNPZ
    t_eval::Vector{Float64}
    mu::Vector{Float64}
    cov::Matrix{Float64}
    chol::Cholesky{Float64, Matrix{Float64}}
    logdetΣ::Float64
end

# internal: build a stable Cholesky (adds jitter if needed)
function _stable_cholesky(cov::AbstractMatrix{<:Real}; jitter0=1e-10, max_tries=10)
    Σ = Symmetric(Matrix{Float64}(cov))
    jitter = jitter0
    for k in 1:max_tries
        try
            C = cholesky(Σ + jitter*I; check=true)
            return C, jitter
        catch
            jitter *= 10
        end
    end
    error("Could not cholesky covariance even after adding jitter up to $(jitter). " *
          "cov may not be SPD.")
end

"""
Load GP likelihood from NPZ file with keys:
- "mu": Vector length n
- "cov": Matrix nxn
- "t_eval": Vector length n (may be nx1; we vec() it)

Returns GPLikelihoodNPZ.
"""
function load_gp_likelihood_npz(path::AbstractString; jitter0=1e-10)
    data = NPZ.npzread(path)

    haskey(data, "mu")    || error("NPZ missing key 'mu'")
    haskey(data, "cov")   || error("NPZ missing key 'cov'")
    haskey(data, "t_eval")|| error("NPZ missing key 't_eval'")

    mu  = vec(Float64.(data["mu"]))
    cov = Matrix{Float64}(data["cov"])
    t   = vec(Float64.(data["t_eval"]))  # handles (n,1) or (n,)

    n = length(mu)
    size(cov,1) == n && size(cov,2) == n || error("cov size $(size(cov)) does not match length(mu)=$n")
    length(t) == n || error("t_eval length $(length(t)) does not match length(mu)=$n")

    chol, jitter_used = _stable_cholesky(cov; jitter0=jitter0)

    # logdet Σ from cholesky: Σ = L*L' => logdet = 2*sum(log(diag(L)))
    L = chol.L
    logdetΣ = 2.0 * sum(log, diag(L))

    if jitter_used > jitter0
        @warn "Added jitter to cov for SPD: jitter=$(jitter_used)"
    end

    return GPLikelihoodNPZ(t, mu, cov, chol, logdetΣ)
end

function nll(gp::GPLikelihoodNPZ, y::AbstractVector{<:Real})
    n = length(gp.mu)
    length(y) == n || error("y length $(length(y)) does not match n=$n")

    r = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r[i] = Float64(y[i]) - gp.mu[i]
    end

    # Solve Σ^{-1} r using cholesky
    α = gp.chol \ r
    quad = dot(r, α)

    return 0.5 * (quad + gp.logdetΣ + n * log(2π))
end

# gp(y) -> NLL
(gp::GPLikelihoodNPZ)(y::AbstractVector{<:Real}) = nll(gp, y)

"""
Draw nsamples sample paths from N(mu, cov).

Returns a matrix Y of size (n, nsamples), with n = length(t_eval).
"""
function sample_paths(gp::GPLikelihoodNPZ, nsamples::Int; rng::AbstractRNG=Random.default_rng())
    n = length(gp.mu)
    Y = Matrix{Float64}(undef, n, nsamples)
    L = gp.chol.L

    z = Vector{Float64}(undef, n)
    @inbounds for s in 1:nsamples
        # z ~ N(0, I)
        for i in 1:n
            z[i] = randn(rng)
        end
        # y = mu + L*z
        y = gp.mu .+ L*z
        @inbounds for i in 1:n
            Y[i, s] = y[i]
        end
    end
    return Y
end

end # module

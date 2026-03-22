module Losses

using NPZ

export AbstractLoss, FunctionLoss,
       GridLikelihoodNPZ, load_grid_likelihood_npz

abstract type AbstractLoss end

"""
Wrap a function f(θ)::Real as a loss object.
"""
struct FunctionLoss{F} <: AbstractLoss
    f::F
end
(loss::FunctionLoss)(θ) = loss.f(θ)

"""
Grid likelihood: multilinear interpolation on a rectangular grid.

- axes: Vector of 1D arrays (length D)
- loglik: D-dimensional array with size (length(axes[1]), ..., length(axes[D]))

Returns loss = -loglik(θ) by default.
"""
struct GridLikelihoodNPZ <: AbstractLoss
    axes::Vector{Vector{Float64}}
    loglik::Array{Float64}
    clamp::Bool
    neg::Bool
end

"""
Load a grid likelihood from an .npz file.

Expected keys:
- "axes_1", "axes_2", ..., "axes_D" as vectors
- "loglik" as an Array

Options:
- clamp=true: clamp θ to axis ranges (otherwise errors if out of range)
- neg=true: return -loglik (i.e., negative log-likelihood as a loss)
"""
function load_grid_likelihood_npz(path::AbstractString; clamp::Bool=true, neg::Bool=true)
    data = NPZ.npzread(path)
    haskey(data, "loglik") || error("NPZ missing key 'loglik'")
    loglik = Array{Float64}(data["loglik"])

    # discover D by reading axes_k keys
    axes = Vector{Vector{Float64}}()
    k = 1
    while true
        key = "axes_$(k)"
        if haskey(data, key)
            push!(axes, vec(Float64.(data[key])))
            k += 1
        else
            break
        end
    end
    length(axes) >= 1 || error("NPZ missing axes_1, axes_2, ...")
    D = length(axes)

    # sanity check sizes
    size(loglik) == Tuple(length(ax) for ax in axes) ||
        error("loglik size $(size(loglik)) does not match axes lengths $(map(length, axes)).")

    return GridLikelihoodNPZ(axes, loglik, clamp, neg)
end

GridLikelihoodNPZ(path::AbstractString; clamp::Bool=true, neg::Bool=true) =
    load_grid_likelihood_npz(path; clamp=clamp, neg=neg)

@inline function _find_bracket(ax::Vector{Float64}, x::Float64; clamp::Bool)
    n = length(ax)
    if x <= ax[1]
        return clamp ? (1, 1, 0.0) : error("x=$x below axis range")
    elseif x >= ax[n]
        return clamp ? (n, n, 0.0) : error("x=$x above axis range")
    end
    # find i with ax[i] <= x <= ax[i+1]
    i = searchsortedlast(ax, x)
    i == n && (i = n-1)
    x0 = ax[i]
    x1 = ax[i+1]
    t  = (x - x0) / (x1 - x0)
    return (i, i+1, t)
end

"""
Multilinear interpolation for loglik on axes at point θ.
Works for any D >= 1.

Returns interpolated loglik(θ).
"""
function interp_loglik(gl::GridLikelihoodNPZ, θ::AbstractVector{<:Real})
    D = length(gl.axes)
    length(θ) == D || error("θ length mismatch: got $(length(θ)), expected $D")

    # For each dim: lower index i0, upper index i1, weight t in [0,1]
    i0 = Vector{Int}(undef, D)
    i1 = Vector{Int}(undef, D)
    t  = Vector{Float64}(undef, D)

    @inbounds for d in 1:D
        a = gl.axes[d]
        ii0, ii1, tt = _find_bracket(a, Float64(θ[d]); clamp=gl.clamp)
        i0[d] = ii0
        i1[d] = ii1
        t[d]  = tt
    end

    # Sum over 2^D corners
    val = 0.0
    # We build an index tuple for each corner.
    # corner bit b_d=0 uses i0[d], weight (1-t[d]); b_d=1 uses i1[d], weight t[d].
    ncorner = 1 << D
    @inbounds for mask in 0:(ncorner-1)
        w = 1.0
        inds = Vector{Int}(undef, D)
        for d in 1:D
            if (mask >> (d-1)) & 0x1 == 1
                inds[d] = i1[d]
                w *= t[d]
            else
                inds[d] = i0[d]
                w *= (1.0 - t[d])
            end
        end
        # Index into D-dimensional array:
        val += w * gl.loglik[Tuple(inds)...]
    end
    return val
end

function (gl::GridLikelihoodNPZ)(θ::AbstractVector{<:Real})
    ll = interp_loglik(gl, θ)
    return gl.neg ? -ll : ll
end

end # module

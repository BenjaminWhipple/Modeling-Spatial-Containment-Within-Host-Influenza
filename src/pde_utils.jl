using DifferentialEquations

# Smooth positive-part
@inline pospart(x::Float64, δ::Float64) = 0.5 * (x + sqrt(x*x + δ*δ))

# Neumann Laplacian via edge-mirror (1D)
function laplacian_neumann_1d!(lap::AbstractVector{Float64},
                               z::AbstractVector{Float64},
                               dx::Float64)
    nx = length(z)
    invdx2 = 1.0 / (dx*dx)

    @inbounds begin
        lap[1]  = (z[2] - 2z[1] + z[1]) * invdx2
        for i in 2:nx-1
            lap[i] = (z[i+1] - 2z[i] + z[i-1]) * invdx2
        end
        lap[nx] = (z[nx] - 2z[nx] + z[nx-1]) * invdx2
    end
    return nothing
end

# Positivity callback: clamp all states >= 0
function positivity_callback_all(N::Int; nfields::Int=6)
    condition(u, t, integrator) = true

    function affect!(integrator)
        y = integrator.u
        @inbounds for k in 0:(nfields-1)
            off = k*N
            for i in 1:N
                idx = off + i
                if y[idx] < 0.0
                    y[idx] = 0.0
                end
            end
        end
        return nothing
    end

    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

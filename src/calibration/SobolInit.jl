module SobolInit

using QuasiMonteCarlo
using ..OptimTypes: Bounds

export sobol_init_population!

"""
Fill `pop` (DxNP) in-place with Sobol points mapped to bounds.

This uses the QuasiMonteCarlo.jl API:
    sample(n, lb, ub, SobolSample())

`sobol_skip` is implemented by sampling (NP + skip) points and dropping the first `skip`.
"""
function sobol_init_population!(pop::AbstractMatrix{T}, bounds::Bounds{T}; skip::Int=0) where {T<:Real}
    D, NP = size(pop)
    length(bounds.lower) == D || error("sobol_init_population!: bounds dimension mismatch.")

    # QuasiMonteCarlo maps directly into [lb, ub]
    # Returns D×(NP+skip) (typically Float64)
    X = QuasiMonteCarlo.sample(NP + skip, bounds.lower, bounds.upper, SobolSample())

    # Copy the last NP points into pop
    @inbounds for j in 1:NP, d in 1:D
        pop[d, j] = T(X[d, j + skip])
    end

    return pop
end

end # module

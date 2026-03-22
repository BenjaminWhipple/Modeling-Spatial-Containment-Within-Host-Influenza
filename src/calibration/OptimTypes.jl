module OptimTypes

using Random

export Bounds, ConstraintHandling, DEOptions, DEState,
       apply_bounds!, rand_distinct3, best_index

"""
Bounds for parameters θ in ℝ^D.
`lower` and `upper` must be length D.
"""
struct Bounds{T<:Real}
    lower::Vector{T}
    upper::Vector{T}
    function Bounds(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T<:Real}
        length(lower) == length(upper) || error("Bounds: lower/upper length mismatch.")
        any(upper .< lower) && error("Bounds: require upper[i] > lower[i] for all i.")
        new{T}(collect(lower), collect(upper))
    end
end

@enum ConstraintHandling begin
    Clamp
    Reflect
    Resample
end

"""
DE options.

- NP: population size
- F: mutation factor
- CR: crossover probability
- max_gens: number of generations
- constraint: Clamp/Reflect/Resample
- seed: RNG seed (optional)
- tol_f: stop if best_f <= tol_f
- stall_gens: stop if no improvement in best for this many gens
- verbose: print progress
"""
Base.@kwdef mutable struct DEOptions{T<:Real}
    NP::Int = 80
    F::T = T(0.7)
    CR::T = T(0.9)
    max_gens::Int = 200
    constraint::ConstraintHandling = Reflect
    seed::Union{Nothing,Int} = nothing
    tol_f::T = T(-Inf)
    stall_gens::Int = 50
    verbose::Bool = true
end

"""
DE state.

- pop: DxNP population (each column is a candidate vector)
- f: length NP fitness values (lower is better)
- best_x: best vector found so far
- best_f: best fitness value
- gen: current generation
- n_evals: number of function evaluations
"""
mutable struct DEState{T<:Real}
    pop::Matrix{T}       # DxNP
    f::Vector{T}         # NP
    best_x::Vector{T}    # D
    best_f::T
    gen::Int
    n_evals::Int
end

best_index(f::AbstractVector) = argmin(f)

"""
Pick r1,r2,r3 distinct from 1:NP and distinct from i.
"""
function rand_distinct3(rng::AbstractRNG, NP::Int, i::Int)
    NP >= 4 || error("Need NP >= 4 for DE/rand/1.")
    r1 = rand(rng, 1:NP)
    while r1 == i
        r1 = rand(rng, 1:NP)
    end
    r2 = rand(rng, 1:NP)
    while r2 == i || r2 == r1
        r2 = rand(rng, 1:NP)
    end
    r3 = rand(rng, 1:NP)
    while r3 == i || r3 == r1 || r3 == r2
        r3 = rand(rng, 1:NP)
    end
    return r1, r2, r3
end

"""
Apply bounds in-place to vector x.

- Clamp: clip to [lower, upper]
- Reflect: reflect back into bounds (with multiple reflections if needed)
- Resample: replace out-of-bounds components with uniform random within bounds
"""
function apply_bounds!(x::AbstractVector{T}, bounds::Bounds{T},
                       how::ConstraintHandling, rng::AbstractRNG) where {T<:Real}
    lo = bounds.lower
    hi = bounds.upper
    @inbounds for d in eachindex(x)
        xd = x[d]
        # Handle fixed parameters (lo == hi)
        if lo[d] == hi[d]
            x[d] = lo[d]
            continue
        end
        if lo[d] <= xd <= hi[d]
            continue
        end
        if how == Clamp
            x[d] = min(max(xd, lo[d]), hi[d])
        elseif how == Reflect
            # reflect across boundaries; handle far out by repeated reflection
            L = lo[d]
            U = hi[d]
            y = xd
            # bring into range by reflection on [L,U]
            # reflect algorithm: while out of range, mirror across nearest boundary
            while y < L || y > U
                if y < L
                    y = L + (L - y)
                elseif y > U
                    y = U - (y - U)
                end
            end
            x[d] = y
        elseif how == Resample
            x[d] = lo[d] + (hi[d] - lo[d]) * rand(rng)
        else
            error("Unknown constraint handling: $how")
        end
    end
    return x
end

end # module

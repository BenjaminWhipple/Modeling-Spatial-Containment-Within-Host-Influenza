module DE

using Random
using ..OptimTypes: Bounds, DEOptions, DEState, ConstraintHandling,
                    apply_bounds!, rand_distinct3, best_index
using ..SobolInit: sobol_init_population!

export de_optimize, DEMutationStrategy, Rand1, Best1, CurrentToBest1

# Mutation strategy definitions
@enum DEMutationStrategy begin
    Rand1          # v = x_r1 + F*(x_r2 - x_r3)
    Best1          # v = x_best + F*(x_r2 - x_r3)
    CurrentToBest1 # v = x_i + F*(x_best - x_i) + F*(x_r2 - x_r3)
end

"""
de_optimize(loss, bounds; opts=DEOptions(), sobol_skip=0, strategy=Rand1, synchronous=true)

- loss: callable (θ)->Real to minimize
- bounds: Bounds
- opts: DEOptions
- strategy: DEMutationStrategy
- synchronous=true: canonical DE (generate all trials from old population, then replace)
                   if false, does in-place/asynchronous replacement (your earlier behavior)

Returns (best_x, best_f, state)
"""
function de_optimize(loss, bounds::Bounds{T};
                     opts::DEOptions{T}=DEOptions{T}(),
                     sobol_skip::Int=0,
                     strategy::DEMutationStrategy=Rand1,
                     synchronous::Bool=true) where {T<:Real}

    D = length(bounds.lower)
    NP = opts.NP
    NP >= 4 || error("DE: require NP >= 4")

    rng = opts.seed === nothing ? Random.default_rng() : MersenneTwister(opts.seed)

    # Allocate state population + fitness
    pop = Matrix{T}(undef, D, NP)
    f   = Vector{T}(undef, NP)

    # Sobol init
    sobol_init_population!(pop, bounds; skip=sobol_skip)

    # Evaluate initial fitness
    @inbounds for j in 1:NP
        θ = @view pop[:, j]
        f[j] = T(loss(θ))
    end

    bi = best_index(f)
    best_x = Vector{T}(pop[:, bi])
    best_f = f[bi]

    state = DEState{T}(pop, f, best_x, best_f, 0, NP)

    # Working buffers
    trial  = Vector{T}(undef, D)
    mutant = Vector{T}(undef, D)

    # Frozen-best buffer (per generation)
    best_x_gen = similar(state.best_x)

    # For synchronous replacement
    newpop = synchronous ? similar(pop) : pop
    newf   = synchronous ? similar(f)   : f

    stall = 0
    if opts.verbose
        println("DE init: best_f = ", state.best_f, " | strategy=", strategy,
                " | synchronous=", synchronous)
    end

    for gen in 1:opts.max_gens
        state.gen = gen
        improved_this_gen = false

        # Freeze best for this generation
        copyto!(best_x_gen, state.best_x)

        @inbounds for i in 1:NP
            #println("DE Gen: ",gen," | Population: ", i)
            # pick r1,r2,r3 distinct from i
            r1, r2, r3 = rand_distinct3(rng, NP, i)

            # IMPORTANT:
            # use OLD population for generating trials if synchronous
            oldpop = pop
            oldf   = f

            x_r1 = @view oldpop[:, r1]
            x_r2 = @view oldpop[:, r2]
            x_r3 = @view oldpop[:, r3]
            x_i  = @view oldpop[:, i]

            # mutation
            if strategy == Rand1
                for d in 1:D
                    mutant[d] = x_r1[d] + opts.F * (x_r2[d] - x_r3[d])
                end
            elseif strategy == Best1
                for d in 1:D
                    mutant[d] = best_x_gen[d] + opts.F * (x_r2[d] - x_r3[d])
                end
            elseif strategy == CurrentToBest1
                for d in 1:D
                    mutant[d] = x_i[d] + opts.F * (best_x_gen[d] - x_i[d]) + opts.F * (x_r2[d] - x_r3[d])
                end
            else
                error("Unknown strategy: $strategy")
            end
            apply_bounds!(mutant, bounds, opts.constraint, rng)

            # binomial crossover
            jrand = rand(rng, 1:D)
            for d in 1:D
                if (rand(rng) < opts.CR) || (d == jrand)
                    trial[d] = mutant[d]
                else
                    trial[d] = x_i[d]
                end
            end
            apply_bounds!(trial, bounds, opts.constraint, rng)

            # selection
            ftrial = T(loss(trial))
            state.n_evals += 1

            if synchronous
                # Write winner into buffers (do NOT overwrite pop yet)
                if ftrial < oldf[i]
                    newf[i] = ftrial
                    for d in 1:D
                        newpop[d, i] = trial[d]
                    end
                else
                    newf[i] = oldf[i]
                    for d in 1:D
                        newpop[d, i] = x_i[d]
                    end
                end
            else
                # Asynchronous: overwrite immediately
                if ftrial < f[i]
                    f[i] = ftrial
                    for d in 1:D
                        pop[d, i] = trial[d]
                    end
                end
            end
        end

        # If synchronous, swap populations now (whole generation update)
        if synchronous
            pop, newpop = newpop, pop
            f,   newf   = newf,   f
            state.pop = pop
            state.f   = f
        end

        # Update global best (after generation)
        bi = best_index(state.f)
        if state.f[bi] < state.best_f
            state.best_f = state.f[bi]
            copyto!(state.best_x, @view state.pop[:, bi])
            improved_this_gen = true
        end

        if opts.verbose && (gen == 1 || gen % 1 == 0)
            println("gen $gen | best_f=$(state.best_f) | evals=$(state.n_evals)")
        end

        # stopping checks
        if state.best_f <= opts.tol_f
            opts.verbose && println("Stopping: reached tol_f at gen $gen")
            break
        end

        if improved_this_gen
            stall = 0
        else
            stall += 1
            if stall >= opts.stall_gens
                opts.verbose && println("Stopping: stalled for $(opts.stall_gens) generations.")
                break
            end
        end
    end

    return state.best_x, state.best_f, state
end

end # module

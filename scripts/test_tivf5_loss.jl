# scripts/test_tivf5_loss.jl
#
# Test script for TIVF5_2D model with IFN positive feedback.
# TIVF5 extends TIVF4 with:
#   - IFN production has positive feedback: p_F * I * b(F)
#     where b(F) = 1 + a_F * F^2 / (K_F^2 + F^2)
#
using Printf
using LinearAlgebra
using DifferentialEquations
using LinearSolve
using Plots
using Measures
gr()

# model (TIVF5: T,E1,E2,I,V,F with IFN positive feedback)
include("../src/models/TIVF5_2D.jl")
using .TIVF5_2D

# positivity clamp
include("../src/pde_utils.jl")

# Helpers for robust union time grid + indexing
function union_times(t1::AbstractVector, t2::AbstractVector; atol::Float64=1e-12)
    ts = sort!(vcat(collect(Float64.(t1)), collect(Float64.(t2))))
    out = Float64[]
    for t in ts
        if isempty(out) || abs(t - out[end]) > atol
            push!(out, t)
        end
    end
    return out
end

function indices_in_grid(t_all::AbstractVector, t_sub::AbstractVector; atol::Float64=1e-12)
    idxs = Vector{Int}(undef, length(t_sub))
    i = 1
    @inbounds for k in eachindex(t_sub)
        tk = Float64(t_sub[k])
        while i <= length(t_all) && t_all[i] < tk - atol
            i += 1
        end
        if i > length(t_all) || abs(t_all[i] - tk) > atol
            error("Time tk=$(tk) not found in union grid within atol=$(atol).")
        end
        idxs[k] = i
    end
    return idxs
end

# Area / extent utilities
"""
Count area above threshold over tissue.
"""
@inline function area_above_threshold(u_slice, tissue::BitVector, dA::Float64, thr::Float64)
    cnt = 0
    @inbounds for i in eachindex(u_slice)
        if tissue[i] && (u_slice[i] > thr)
            cnt += 1
        end
    end
    return cnt * dA
end

"""
Compute time series of area (number of cells x dA) above a fixed threshold.

Arguments:
- `sol`: solution object
- `idxrng`: index range for the field
- `tissue`: BitVector mask for tissue cells
- `dA`: area per cell
- `thr`: absolute threshold (default 1e-6, effectively zero)

Returns:
- `A`: vector of areas above threshold at each time point
"""
function area_timeseries(sol, idxrng, tissue::BitVector, dA::Float64; thr::Float64=1e-6)
    A = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        u = sol.u[k]
        us = @view u[idxrng]
        A[k] = area_above_threshold(us, tissue, dA, thr)
    end
    return A
end

function is_contained(sol, N::Int;
                      min_T_fraction::Float64=0.80,
                      max_final_infection::Float64=10.0)
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)

    T_initial = sum(@view sol.u[1][idx_T])
    T_final   = sum(@view sol.u[end][idx_T])
    frac_remaining = T_final / max(T_initial, 1e-30)

    aggregate_infected = sum(@view sol.u[end][idx_E1]) +
                         sum(@view sol.u[end][idx_E2]) +
                         sum(@view sol.u[end][idx_I])

    return (frac_remaining >= min_T_fraction) && (aggregate_infected <= max_final_infection)
end

function main()
    isdir("Images") || mkpath("Images")

    nx, ny = 32, 32

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

    # positivity clamp for all 6 fields
    cb = positivity_callback_all(N; nfields=TIVF5_2D.nfields())

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

    # Parameter vector for loss evaluation (15 params)
    θ0 = [
        pars.beta, pars.k_E, pars.delta_I,
        pars.p_V, pars.c_V, pars.D_V, pars.k_PV,
        pars.p_F, pars.c_F, pars.D_F, pars.delta_FV, pars.k_FV,
        pars.k_IF, pars.a_F, pars.K_F
    ]
    ϕ0 = log10.(θ0)

    nll = loss(ϕ0)
    @printf("NLL at baseline = %.6e\n", nll)

    # Forward sim on union GP grid (single solve)
    tV = vec(loss.gpV.t_eval)
    tF = vec(loss.gpF.t_eval)
    t_eval = union_times(tV, tF; atol=1e-12)

    TIVF5_2D.set_params!(prob, ϕ0; params_in_log10=true)
    prob2 = remake(prob; tspan=(minimum(t_eval), maximum(t_eval)))

    sol = solve(prob2, QNDF(linsolve=KLUFactorization(), autodiff=false);
        reltol=1e-4, abstol=abstol, dtmax=0.05,
        saveat=t_eval, dense=false, save_everystep=false,
        callback=cb
    )

    #println(is_contained(sol))

    println("Solve completed: ", sol.retcode)

    idxV = indices_in_grid(t_eval, tV; atol=1e-12)
    idxF = indices_in_grid(t_eval, tF; atol=1e-12)

    # Virus GP band + model curve (log10 total V)
    yV_all = TIVF5_2D.default_obs_totalV_log10(sol, N)
    yV_model = yV_all[idxV]

    muV_gp = vec(loss.gpV.mu)
    σV_gp  = sqrt.(max.(diag(loss.gpV.cov), 0.0))

    kband = 1.0
    pV = plot(tV, muV_gp; ribbon=(kband .* σV_gp, kband .* σV_gp),
              fillalpha=0.2, label="Virus GP μ±σ", xlabel="t", ylabel="log10(total V)")
    plot!(pV, tV, yV_model; lw=2, label="Model")
    savefig(pV, "Images/tivf5_gpV_compare.png")
    println("Saved: Images/tivf5_gpV_compare.png")

    # IFN GP band + model curve (log10 total F)
    yF_all = TIVF5_2D.default_obs_totalF_log10(sol, N)
    yF_model = yF_all[idxF]

    muF_gp = vec(loss.gpF.mu)
    σF_gp  = sqrt.(max.(diag(loss.gpF.cov), 0.0))

    pF = plot(tF, muF_gp; ribbon=(kband .* σF_gp, kband .* σF_gp),
              fillalpha=0.2, label="IFN GP μ±σ", xlabel="t", ylabel="log10(total F)")
    plot!(pF, tF, yF_model; lw=2, label="Model")
    savefig(pF, "Images/tivf5_gpF_compare.png")
    println("Saved: Images/tivf5_gpF_compare.png")

    # Aggregate trajectories (T,E1,E2,I,V,F) on union grid (log yscale)
    idx_T  = 1:N
    idx_E1 = (N+1):(2N)
    idx_E2 = (2N+1):(3N)
    idx_I  = (3N+1):(4N)
    idx_V  = (4N+1):(5N)
    idx_F  = (5N+1):(6N)

    eps = 1e-12
    Tmass  = [sum(@view u[idx_T])  + eps for u in sol.u]
    E1mass = [sum(@view u[idx_E1]) + eps for u in sol.u]
    E2mass = [sum(@view u[idx_E2]) + eps for u in sol.u]
    Imass  = [sum(@view u[idx_I])  + eps for u in sol.u]
    Vmass  = [sum(@view u[idx_V])  + eps for u in sol.u]
    Fmass  = [sum(@view u[idx_F])  + eps for u in sol.u]

    yticks_vals = 10.0 .^ (0:7)
    yticks_labs = ["$k" for k in 0:7]

    pAgg = plot(
        t_eval, max.(Tmass, eps);
        label="∫T dA",
        xlabel="t",
        ylabel="log10 mass",
        title="Aggregate State Trajectories",
        yscale=:log10,
        yticks=(yticks_vals, yticks_labs),
        ylims=(1.0, 1e7)
    )
    plot!(pAgg, t_eval, E1mass .+ 1.0; label="∫E1 dA")
    plot!(pAgg, t_eval, E2mass .+ 1.0; label="∫E2 dA")
    plot!(pAgg, t_eval, Imass  .+ 1.0; label="∫I dA")
    plot!(pAgg, t_eval, Vmass  .+ 1.0; label="∫V dA")
    plot!(pAgg, t_eval, Fmass  .+ 1.0; label="∫F dA")
    savefig(pAgg, "Images/tivf5_aggregate_masses_logscale.png")
    println("Saved: Images/tivf5_aggregate_masses_logscale.png")

    # Spatial extent (area above threshold)
    tissue = prob2.p.tissue
    dA     = prob2.p.dA / ((34.0/32.0)^2)

    # Fixed threshold for "effectively zero"
    thr = 0.99

    AT  = area_timeseries(sol, idx_T,  tissue, dA; thr=thr)
    AE1 = area_timeseries(sol, idx_E1, tissue, dA; thr=thr)
    AE2 = area_timeseries(sol, idx_E2, tissue, dA; thr=thr)
    AI  = area_timeseries(sol, idx_I,  tissue, dA; thr=thr)
    AV  = area_timeseries(sol, idx_V,  tissue, dA; thr=thr)
    AF  = area_timeseries(sol, idx_F,  tissue, dA; thr=thr)

    pA = plot(sol.t, AT;  lw=2, label="T",  xlabel="t", ylabel="Area",
              title="Spatial Extent")
    plot!(pA, sol.t, AE1; lw=2, label="E1")
    plot!(pA, sol.t, AE2; lw=2, label="E2")
    plot!(pA, sol.t, AI;  lw=2, label="I")
    plot!(pA, sol.t, AV;  lw=2, label="V")
    plot!(pA, sol.t, AF;  lw=2, label="F")
    savefig(pA, "Images/tivf5_area_extents.png")
    println("Saved: Images/tivf5_area_extents.png")

    # Heatmaps across time (snapshots)
    nS = 6
    snap_idxs = round.(Int, range(1, length(sol.t); length=nS))
    ts = sol.t[snap_idxs]

    function vmax_over_snapshots(idxrng)
        mx = -Inf
        @inbounds for kidx in snap_idxs
            u = sol.u[kidx]
            for j in idxrng
                v = u[j]
                if v > mx; mx = v; end
            end
        end
        return max(0.0, mx)
    end

    Tmax  = vmax_over_snapshots(idx_T)
    E1max = vmax_over_snapshots(idx_E1)
    E2max = vmax_over_snapshots(idx_E2)
    Imax  = vmax_over_snapshots(idx_I)
    Vmax  = vmax_over_snapshots(idx_V)
    Fmax  = vmax_over_snapshots(idx_F)

    function nice_ticks(vmax; nticks=4)
        vmax ≤ 0 && return ([], [])
        ticks = collect(range(0, vmax; length=nticks))
        return (ticks, string.(round.(ticks; digits=2)))
    end

    tT   = nice_ticks(Tmax)
    tE1  = nice_ticks(E1max)
    tE2  = nice_ticks(E2max)
    tI   = nice_ticks(Imax)
    tVtk = nice_ticks(Vmax)
    tFtk = nice_ticks(Fmax)

    @inline function make_field_panel(Uk::AbstractMatrix, x, y;
            clims::Tuple{Real,Real},
            title::String,
            show_y::Bool,
            row_label::String,
            show_x::Bool
        )
        heatmap(x, y, Uk';
            clims=clims,
            colorbar=false,
            aspect_ratio=:equal,
            framestyle=:box,
            title=title,
            xlabel = show_x ? "x" : "",
            ylabel = show_y ? row_label : "",
            xticks = show_x ? :auto : false,
            yticks = show_y ? :auto : false,
            margin=1mm,
        )
    end

    @inline function make_colorbar_panel(vmax; ticks=nothing)
        Z = [NaN NaN; NaN NaN]
        heatmap(Z;
            clims=(0.0, vmax),
            colorbar=true,
            colorbar_ticks=ticks,
            axis=nothing,
            framestyle=:none,
            title="",
            margin=1mm,
        )
    end

    panels = Plots.Plot[]

    function push_row!(label::String, idxrng, vmax, ticks)
        for (col, (kidx, t_snap)) in enumerate(zip(snap_idxs, ts))
            Uk = reshape(sol.u[kidx][idxrng], nx, ny)
            push!(panels, make_field_panel(Uk, x, y;
                clims=(0.0, vmax),
                title="",
                show_y=(col == 1),
                row_label=label,
                show_x=false
            ))
        end
        push!(panels, make_colorbar_panel(vmax; ticks=ticks))
        return nothing
    end

    push_row!("T",  idx_T,  Tmax,  tT)
    push_row!("E1", idx_E1, E1max, tE1)
    push_row!("E2", idx_E2, E2max, tE2)
    push_row!("I",  idx_I,  Imax,  tI)
    push_row!("V",  idx_V,  Vmax,  tVtk)
    push_row!("F",  idx_F,  Fmax,  tFtk)

    # Column titles only on first row
    for col in 1:nS
        title!(panels[col], "t=$(round(ts[col], digits=2))")
    end

    # x-axis labels only on bottom row (row 6)
    bottom_row_start = 5*(nS+1) + 1
    for col in 1:nS
        xlabel!(panels[bottom_row_start + (col-1)], "x")
        xticks!(panels[bottom_row_start + (col-1)], :auto)
    end

    panel_w = 190
    panel_h = 190
    cb_w    = 140

    fig_w = panel_w*nS + cb_w
    fig_h = panel_h*6

    widths = vcat(fill(1.0, nS), cb_w / panel_w)
    widths ./= sum(widths)

    fig = plot(panels...;
        layout = grid(6, nS+1; widths=widths),
        size   = (fig_w, fig_h),
        left_margin   = 6mm,
        right_margin  = 4mm,
        bottom_margin = 6mm,
        top_margin    = 6mm,
    )

    savefig(fig, "Images/tivf5_heatmaps.png")
    println("Saved: Images/tivf5_heatmaps.png")

    println("\n=== Done ===")
end

main()

module Domain2D
export idx2, make_grid, make_tissue_mask, laplacian_neumann_masked_2d!

@inline idx2(i, j, nx) = i + (j - 1) * nx

function make_grid(nx::Int, ny::Int)
    x = range(0.0, 1.0; length=nx)
    y = range(0.0, 1.0; length=ny)
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dA = dx * dy
    N  = nx * ny
    return x, y, dx, dy, dA, N
end

function make_tissue_mask(Tvec0::AbstractVector{<:Real})
    tissue = BitVector(Tvec0 .> 0.0)
    n_tissue = count(tissue)
    n_tissue == 0 && error("tissue mask has zero tissue cells (T_init/T0 all zero?)")
    return tissue, n_tissue
end

# 2D masked Neumann Laplacian (edge mirror + no-flux into void neighbors)
function laplacian_neumann_masked_2d!(Δu::AbstractVector{T}, u::AbstractVector{T},
                                     tissue::AbstractVector{Bool},
                                     nx::Int, ny::Int, dx::Float64, dy::Float64) where {T}
    @inbounds begin
        invdx2 = 1.0 / (dx * dx)
        invdy2 = 1.0 / (dy * dy)

        for j in 1:ny
            jm = (j == 1  ? 2     : j - 1)
            jp = (j == ny ? ny-1  : j + 1)

            for i in 1:nx
                im = (i == 1  ? 2    : i - 1)
                ip = (i == nx ? nx-1 : i + 1)

                c  = idx2(i,  j,  nx)

                if !tissue[c]
                    Δu[c] = zero(T)
                    continue
                end

                xm = idx2(im, j,  nx)
                xp = idx2(ip, j,  nx)
                ym = idx2(i,  jm, nx)
                yp = idx2(i,  jp, nx)

                uc  = u[c]
                uxM = tissue[xm] ? u[xm] : uc
                uxP = tissue[xp] ? u[xp] : uc
                uyM = tissue[ym] ? u[ym] : uc
                uyP = tissue[yp] ? u[yp] : uc

                Δu[c] = (uxP - 2uc + uxM) * invdx2 +
                        (uyP - 2uc + uyM) * invdy2
            end
        end
    end
    return nothing
end

end # module

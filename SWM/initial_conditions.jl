module InitialConditions

using LinearAlgebra
import ..LowRank

export get_lr_gaussian_hill, get_lr_geostrophic

function get_lr_gaussian_hill(N::Int, L::Real, dx::Real;
                              tol=1e-6, maxrank=typemax(Int), h_mean=10000.0,
                              h_amp=1000.0, alpha=400.0)

    xs = dx .* collect(1:N)
    ys = dx .* collect(1:N)

    gx = exp.(-alpha .* ((xs .- L / 2).^2 ./ L^2))
    gy = exp.(-alpha .* ((ys .- L / 2).^2 ./ L^2))

    h_const = LowRank.const_lr(h_mean, N, N)
    h_gauss = LowRank.from_factors(reshape(gx, :, 1),
                                   fill(Float64(h_amp), 1, 1),
                                   reshape(gy, :, 1);
                                   rtol=tol, maxrank=maxrank)

    h = LowRank.round_lr(h_const + h_gauss;
                         rtol=tol, maxrank=maxrank)

    u = LowRank.zero_lr(N, N)
    v = LowRank.zero_lr(N, N)

    return u, v, h
end

"""
    get_lr_geostrophic(N, dx; U0, grav, pcori) -> (u, v, h)

Geostrophically-balanced zonal-jet initial state from `neuro_swm.jl`
(lines 294–298), as low-rank `LRMat` fields on an N×N periodic grid with
`Lx = N*dx` and `2π*x/Lx = 2π*i/N`:

    u(x,y) = U0 * sin(2π y / Lx)                              (rank 1, y only)
    v(x,y) = 0.5 * cos(2π x / Lx) * sin(2π y / Lx)            (rank 1, separable)
    h(x,y) = H0 + (pcori/grav)*(U0*Lx/2π) * cos(2π y / Lx)    (rank 2)

These are exactly low-rank, so they are built directly from factors.
"""
function get_lr_geostrophic(N::Int, dx::Real;
                            U0=20.0, grav=9.80616, pcori=2*7.292e-5, H0=10e3)

    Lx = N * dx
    xs = dx .* collect(0:N-1)
    ys = dx .* collect(0:N-1)

    sx = sin.(2π .* xs ./ Lx)
    cx = cos.(2π .* xs ./ Lx)
    sy = sin.(2π .* ys ./ Lx)
    cy = cos.(2π .* ys ./ Lx)
    ones_col = ones(N)

    # u = U0 * 1_x ⊗ sin(2π y/Lx)         (factor U on x is constant)
    u = LowRank.from_factors(reshape(ones_col, :, 1),
                             fill(Float64(U0), 1, 1),
                             reshape(sy, :, 1))

    # v = 0.5 * cos(2π x/Lx) ⊗ sin(2π y/Lx)
    v = LowRank.from_factors(reshape(cx, :, 1),
                             fill(0.5, 1, 1),
                             reshape(sy, :, 1))

    # h = H0 (constant) + (pcori/grav)(U0 Lx/2π) * 1_x ⊗ cos(2π y/Lx)
    h_amp   = (pcori / grav) * (U0 * Lx / (2π))
    h_const = LowRank.const_lr(Float64(H0), N, N)
    h_wave  = LowRank.from_factors(reshape(ones_col, :, 1),
                                   fill(Float64(h_amp), 1, 1),
                                   reshape(cy, :, 1))
    h = LowRank.round_lr(h_const + h_wave)

    return u, v, h
end

end
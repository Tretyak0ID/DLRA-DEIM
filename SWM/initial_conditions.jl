module InitialConditions

using LinearAlgebra
import ..LowRank

export get_lr_gaussian_hill

function get_lr_gaussian_hill(N::Int, L::Real, dx::Real;
                              tol=1e-6, maxrank=typemax(Int), h_mean=10000.0,
                              h_amp=1000.0, alpha=400.0)

    xs = dx .* collect(1:N)
    ys = dx .* collect(1:N)

    gx = exp.(-alpha .* ((xs .- L / 2).^2 ./ L^2))
    gy = exp.(-alpha .* ((ys .- L / 2).^2 ./ L^2))

    h_const = LowRank.const_lr(h_mean, N, N)
    h_gauss = LowRank.from_factors(reshape(gx, :, 1), [Float64(h_amp)], reshape(gy, :, 1);
                                   rtol=tol, maxrank=maxrank)

    h = LowRank.round_lr(h_const + h_gauss;
                         rtol=tol, maxrank=maxrank)

    u = LowRank.zero_lr(N, N)
    v = LowRank.zero_lr(N, N)

    return u, v, h
end

end
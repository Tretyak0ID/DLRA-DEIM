using Test
using LinearAlgebra

include("../SWM/SWM.jl")
using .LowRank
using .SWERhs

# Dense periodic centered differences (reference).
# ∂x along rows (axis 1), ∂y along columns (axis 2) — matching swe_rhs.jl.
cdx(F, dx) = (circshift(F, (-1, 0)) .- circshift(F, (1, 0))) ./ (2dx)
cdy(F, dx) = (circshift(F, (0, -1)) .- circshift(F, (0, 1))) ./ (2dx)

function dense_rhs(u, v, h, p)
    dx, g, f = p.dx, p.grav, p.fcor
    KE = 0.5 .* (u .^ 2 .+ v .^ 2) .+ g .* h
    Q  = cdx(v, dx) .- cdy(u, dx) .+ f
    Gu = .-cdx(KE, dx) .+ Q .* v
    Gv = .-cdy(KE, dx) .- Q .* u
    Gh = .-(cdx(u .* h, dx) .+ cdy(v .* h, dx))
    return Gu, Gv, Gh
end

@testset "SWERhs" begin
    N = 16; dx = 0.5; g = 9.8; fcor = 1.2
    u0 = [sin(2π*i/N) * cos(2π*j/N)     for i in 0:N-1, j in 0:N-1]
    v0 = [cos(2π*i/N) + 0.3sin(4π*j/N)  for i in 0:N-1, j in 0:N-1]
    h0 = [10.0 + 0.5sin(2π*(i+j)/N)     for i in 0:N-1, j in 0:N-1]

    p = SWEParams(dx=dx, grav=g, fcor=fcor)
    G = swe_rhs(from_dense(u0), from_dense(v0), from_dense(h0), p)

    Gu, Gv, Gh = dense_rhs(u0, v0, h0, p)
    refs = (Gu, Gv, Gh)

    I = [2, 5, 9, 14]
    J = [1, 4, 8, 16]

    @testset "getrows matches dense (field $fi)" for fi in 1:3
        R = SWERhs.getrows(G, fi, I, 0.0)
        @test size(R) == (length(I), N)
        @test norm(R - refs[fi][I, :]) < 1e-10
    end

    @testset "getcols matches dense (field $fi)" for fi in 1:3
        C = SWERhs.getcols(G, fi, J, 0.0)
        @test size(C) == (N, length(J))
        @test norm(C - refs[fi][:, J]) < 1e-10
    end

    @testset "getblock matches dense (field $fi)" for fi in 1:3
        B = SWERhs.getblock(G, fi, I, J, 0.0)
        @test size(B) == (length(I), length(J))
        @test norm(B - refs[fi][I, J]) < 1e-10
    end

    @testset "update_state swaps only one field" begin
        u1 = from_dense(2 .* u0)
        G2 = SWERhs.update_state(G, 1, u1)
        # field u changed
        @test norm(dense(G2.u) - 2 .* u0) < 1e-10
        # fields v, h unchanged
        @test norm(dense(G2.v) - v0) < 1e-10
        @test norm(dense(G2.h) - h0) < 1e-10
    end

    @testset "rank-deficient fields still correct" begin
        # rank-1 fields
        ur = [sin(2π*i/N)        for i in 0:N-1, j in 0:N-1]   # depends on i only
        vr = [cos(2π*j/N)        for i in 0:N-1, j in 0:N-1]   # depends on j only
        hr = fill(5.0, N, N)
        Gr = swe_rhs(from_dense(ur), from_dense(vr), from_dense(hr), p)
        ru, rv, rh = dense_rhs(ur, vr, hr, p)
        @test norm(SWERhs.getrows(Gr, 1, I, 0.0) - ru[I, :]) < 1e-10
        @test norm(SWERhs.getcols(Gr, 2, J, 0.0) - rv[:, J]) < 1e-10
        @test norm(SWERhs.getblock(Gr, 3, I, J, 0.0) - rh[I, J]) < 1e-10
    end
end

using Test
using LinearAlgebra

include("../SWM/SWM.jl")
using .LowRank
using .InitialConditions

@testset "InitialConditions" begin

    @testset "geostrophic matches dense reference" begin
        N = 64
        grav = 9.80616; omega = 7.292e-5; pcori = 2*omega
        H0 = 10e3; radz = 6371.22e3; Lx = 2π*radz; dx = Lx/N
        U0 = 20.0

        xs = collect(0:N-1) .* dx
        u_ref = [U0*sin(2π*y/Lx)                              for x in xs, y in xs]
        v_ref = [0.5*cos(2π*x/Lx)*sin(2π*y/Lx)               for x in xs, y in xs]
        h_ref = [H0 + (pcori/grav)*(U0*Lx/(2π))*cos(2π*y/Lx) for x in xs, y in xs]

        u, v, h = get_lr_geostrophic(N, dx; U0=U0, grav=grav, pcori=pcori, H0=H0)

        @test size(u) == (N, N)
        @test norm(dense(u) - u_ref) < 1e-8
        @test norm(dense(v) - v_ref) < 1e-8
        @test norm(dense(h) - h_ref) < 1e-6

        # exact low-rank structure
        @test LowRank.rank(u) == 1
        @test LowRank.rank(v) == 1
    end

    @testset "gaussian hill builds and is low rank" begin
        N = 32; dx = 1.0; L = N * dx
        u, v, h = get_lr_gaussian_hill(N, L, dx)
        @test size(h) == (N, N)
        @test LowRank.rank(u) == 0      # zero velocity
        @test LowRank.rank(v) == 0
        @test LowRank.rank(h) <= 2      # const + separable gaussian
        @test all(isfinite, dense(h))
    end

end

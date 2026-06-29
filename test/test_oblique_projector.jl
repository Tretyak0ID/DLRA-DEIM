using Test
using Random
using LinearAlgebra

include("../DLRA/oblique_projector.jl")

function random_orthonormal_cols(n, m; seed=42)
    Q, _ = qr(randn(MersenneTwister(seed), n, m))
    return Matrix(Q)[:, 1:m]
end

@testset "ObliqueProjector" begin

    @testset "construction" begin
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 4, MersenneTwister(1))

        @test length(P.S) == 4
        @test allunique(P.S)
        @test all(1 .<= P.S .<= 20)
        @test size(P.U) == (20, 4)
    end

    @testset "k > m uses k indices" begin
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 8, MersenneTwister(1))

        @test length(P.S) == 8
        @test allunique(P.S)
    end

    @testset "reproducible with same seed" begin
        U = random_orthonormal_cols(20, 4)
        P1 = ObliqueProjector(U, 6, MersenneTwister(7))
        P2 = ObliqueProjector(U, 6, MersenneTwister(7))

        @test P1.S == P2.S
    end

    @testset "left multiply: projector identity on U (k == m)" begin
        # P*U should be I_m, so U*(P*U) == U
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 4, MersenneTwister(1))

        @test norm(U * (P * U) - U) < 1e-10
    end

    @testset "left multiply: projector identity on U (k > m)" begin
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 8, MersenneTwister(1))

        @test norm(U * (P * U) - U) < 1e-10
    end

    @testset "left multiply: output size" begin
        n, m, k, p = 30, 5, 10, 7
        U = random_orthonormal_cols(n, m)
        P = ObliqueProjector(U, k, MersenneTwister(1))
        A = randn(MersenneTwister(2), n, p)

        C = P * A
        @test size(C) == (m, p)
    end

    @testset "right multiply: projector identity on U' (k == m)" begin
        # (U'*P)*U' == U', i.e. right projection recovers row space of U'
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 4, MersenneTwister(1))

        @test norm((U' * P) * U' - U') < 1e-10
    end

    @testset "right multiply: projector identity on U' (k > m)" begin
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 8, MersenneTwister(1))

        @test norm((U' * P) * U' - U') < 1e-10
    end

    @testset "right multiply: output size" begin
        n, m, k, p = 30, 5, 10, 7
        U = random_orthonormal_cols(n, m)
        P = ObliqueProjector(U, k, MersenneTwister(1))
        A = randn(MersenneTwister(2), p, n)

        C = A * P
        @test size(C) == (p, m)
    end

    @testset "left and right are transposes of each other on U" begin
        # (P * U) == (U' * P)' since both equal I_m
        U = random_orthonormal_cols(20, 4)
        P = ObliqueProjector(U, 8, MersenneTwister(1))

        @test norm(P * U - (U' * P)') < 1e-10
    end

end

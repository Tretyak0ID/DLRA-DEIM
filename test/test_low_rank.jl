using Test
using LinearAlgebra

include("../DLRA/low_rank.jl")
using .LowRank

@testset "LowRank" begin

    @testset "from_dense roundtrip" begin
        A = randn(10, 8)
        B = from_dense(A)
        @test size(B) == (10, 8)
        @test norm(dense(B) - A) < 1e-10
        @test size(B.S) == (LowRank.rank(B), LowRank.rank(B))
    end

    @testset "from_dense rank truncation" begin
        # rank-3 matrix embedded in 10×8
        U = Matrix(qr(randn(10, 3)).Q)
        V = Matrix(qr(randn(8,  3)).Q)
        S = Diagonal([5.0, 2.0, 0.5])
        A = U * S * V'
        B = from_dense(A; rtol=1e-10)
        @test LowRank.rank(B) == 3
        @test norm(dense(B) - A) < 1e-8
    end

    @testset "zero_lr" begin
        Z = zero_lr(6, 5)
        @test size(Z) == (6, 5)
        @test LowRank.rank(Z) == 0
        @test norm(dense(Z)) == 0
    end

    @testset "const_lr" begin
        C = const_lr(3.0, 5, 7)
        @test size(C) == (5, 7)
        @test norm(dense(C) - fill(3.0, 5, 7)) < 1e-12
    end

    @testset "addition" begin
        A = from_dense(randn(8, 6))
        B = from_dense(randn(8, 6))
        C = A + B
        @test size(C) == (8, 6)
        @test norm(dense(C) - (dense(A) + dense(B))) < 1e-10
    end

    @testset "subtraction" begin
        A = from_dense(randn(8, 6))
        B = from_dense(randn(8, 6))
        @test norm(dense(A - B) - (dense(A) - dense(B))) < 1e-10
    end

    @testset "scalar multiply" begin
        A = from_dense(randn(8, 6))
        @test norm(dense(3.0 * A) - 3.0 * dense(A)) < 1e-10
        @test norm(dense(A * 0.5) - 0.5 * dense(A)) < 1e-10
        @test LowRank.rank(0.0 * A) == 0
    end

    @testset "negation" begin
        A = from_dense(randn(8, 6))
        @test norm(dense(-A) + dense(A)) < 1e-12
    end

    @testset "round_lr preserves matrix, reduces rank" begin
        # sum of two rank-3 matrices sharing the same column space -> rank ≤ 3
        U = Matrix(qr(randn(10, 3)).Q)
        V = Matrix(qr(randn(8,  3)).Q)
        A = from_dense(U * Diagonal([3.0, 2.0, 1.0]) * V')
        B = from_dense(U * Diagonal([1.0, 0.5, 0.1]) * V')
        C = A + B   # internally reorth'd, but let's also test round_lr explicitly
        D = round_lr(C; rtol=1e-10)
        @test norm(dense(D) - dense(C)) < 1e-8
        @test LowRank.rank(D) <= LowRank.rank(C)
    end

    @testset "hadamard_lr" begin
        A = from_dense(randn(8, 6))
        B = from_dense(randn(8, 6))
        H = hadamard_lr(A, B)
        @test size(H) == (8, 6)
        @test norm(dense(H) - dense(A) .* dense(B)) < 1e-8
    end

    @testset "from_factors with non-orthogonal inputs" begin
        # pass non-orthonormal U, V — reorth should fix it
        U = randn(10, 3)
        V = randn(8,  3)
        S = [2.0 0.1 0.0; 0.0 1.0 0.0; 0.0 0.0 0.5]
        A = from_factors(U, S, V)
        @test norm(dense(A) - U * S * V') < 1e-10
        @test norm(A.U' * A.U - I) < 1e-12
        @test norm(A.V' * A.V - I) < 1e-12
    end

    @testset "S is square after construction" begin
        A = from_dense(randn(7, 5))
        r = LowRank.rank(A)
        @test size(A.S) == (r, r)
    end

end

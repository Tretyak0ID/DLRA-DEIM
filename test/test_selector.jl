using Test
using Random
using LinearAlgebra

include("../DLRA/selector.jl")

# Build a random m×n matrix with orthonormal rows
function random_orthonormal_rows(m, n; seed=42)
    rng = MersenneTwister(seed)
    A = randn(rng, n, m)
    Q, _ = qr(A)
    return Matrix(Q)'   # m×n, orthonormal rows
end

@testset "select_subset" begin

    @testset "output shape and valid indices" begin
        X = random_orthonormal_rows(3, 10)
        rng = MersenneTwister(1)
        idx = select_subset(X, 5, rng)

        @test length(idx) == 5
        @test allunique(idx)
        @test all(1 .<= idx .<= 10)
    end

    @testset "k == m returns exactly m indices" begin
        X = random_orthonormal_rows(4, 8)
        rng = MersenneTwister(2)
        idx = select_subset(X, 4, rng)

        @test length(idx) == 4
        @test allunique(idx)
    end

    @testset "k == n returns all column indices" begin
        X = random_orthonormal_rows(3, 6)
        rng = MersenneTwister(3)
        idx = select_subset(X, 6, rng)

        @test sort(idx) == 1:6
    end

    @testset "selected m columns span the row space" begin
        # X[:,S] must be full rank (rank m) — the whole point of DEIM selection
        X = random_orthonormal_rows(4, 20)
        rng = MersenneTwister(4)
        idx = select_subset(X, 4, rng)   # k == m

        selected = X[:, idx]
        @test rank(selected) == 4
    end

    @testset "selected submatrix is well-conditioned" begin
        # Condition number of the m×m submatrix should be modest
        X = random_orthonormal_rows(5, 30)
        rng = MersenneTwister(5)
        idx = select_subset(X, 5, rng)

        cond_num = cond(X[:, idx])
        @test cond_num < 100.0
    end

    @testset "reproducible with same seed" begin
        X = random_orthonormal_rows(3, 15)
        idx1 = select_subset(X, 7, MersenneTwister(99))
        idx2 = select_subset(X, 7, MersenneTwister(99))

        @test idx1 == idx2
    end

    @testset "different seeds give different results" begin
        X = random_orthonormal_rows(3, 15)
        results = [select_subset(X, 5, MersenneTwister(s)) for s in 1:20]

        @test length(unique(results)) > 1
    end

end

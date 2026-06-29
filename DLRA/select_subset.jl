module SelectSubset

using LinearAlgebra
using Random

export select_subset

"""
    select_subset(X, k, rng=Random.default_rng())

Select `k` columns from matrix `X` (m×n, with orthonormal rows) using the
Adaptive Randomized Pivoting (ARP) algorithm for the first `m` columns, then
uniform random sampling for the remaining `k - m` columns.

Returns a vector of `k` 1-based column indices.

Reference: Cortinovis and Cressner (2026), "Adaptive Randomized Pivoting for
Column Subset Selection, DEIM, and Low-Rank Approximation".
"""
function select_subset(X::AbstractMatrix{T}, k::Int,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    m, n = size(X)
    @assert k <= n "k must be <= number of columns n"
    @assert k >= m "k must be >= number of rows m"

    indices = arp_algorithm(X, rng)   # selects first m, rest in tail

    # Uniformly shuffle the tail (positions m+1:n) and keep k-m of them
    if k > m
        tail = @view indices[m+1:n]
        shuffle!(rng, tail)
    end

    return indices[1:k]
end


"""
    arp_algorithm(X, rng)

Adaptive Randomized Pivoting: selects m columns from the m×n matrix `X`
(assumed to have orthonormal rows) by iteratively sampling and accepting a
column proportional to its squared residual norm after projecting out the
already-selected directions.

Returns a length-n index vector where the first m entries are the selected
column indices (1-based) and the remaining n-m entries are the rest.
"""
function arp_algorithm(X::AbstractMatrix{T},
                       rng::AbstractRNG) where {T<:AbstractFloat}
    m, n = size(X)

    # Q = X (rows already orthonormal, no LQ needed)
    Q = copy(X)   # m×n; we permute columns in-place alongside indices

    indices = collect(1:n)

    # Growing orthonormal basis for the selected subspace (m×m pre-allocated)
    B = Matrix{T}(undef, m, m)

    for t in 1:m
        # Sample until acceptance
        j = t  # will be overwritten
        accepted = false
        while !accepted
            j = rand(rng, t:n)

            # Residual: project Q[:,j] onto complement of span(B[:,1:t-1])
            q_ort = Q[:, j]
            for i in 1:t-1
                q_ort -= dot(B[:, i], q_ort) * B[:, i]
            end

            # Accept with probability ||q_ort||^2  (≤ 1 since rows of X are orthonormal)
            p = sum(abs2, q_ort)
            accepted = rand(rng) < p
        end

        # Recompute orthogonal component cleanly for the accepted column
        q_ort = Q[:, j]
        for i in 1:t-1
            q_ort -= dot(B[:, i], q_ort) * B[:, i]
        end
        B[:, t] = q_ort / norm(q_ort)

        # Swap accepted column into position t
        indices[t], indices[j] = indices[j], indices[t]
        Q[:, t], Q[:, j] = Q[:, j], copy(Q[:, t])
    end

    return indices
end

end # module SelectSubset

module ObliqueProjectors

using LinearAlgebra
using Random

# Depends on the SelectSubset module (select_subset). Loaded by the DLRA entry
# file (DLRA/DLRA.jl) in dependency order; do not include it here.
using ..SelectSubset

export ObliqueProjector, getrows, getcols

"""
    ObliqueProjector{T}

Stabilized DEIM oblique projector P = U (P_S U)† P_S, where:
- `U`  is an n×m matrix with orthonormal columns (the basis),
- `S`  is a set of k ≥ m row indices selected by ARP,
- `†`  denotes the pseudoinverse (least-squares solve when k > m).

Multiplication returns the m×… coefficient matrix C = (P_S U)† P_S A in the
U-basis. Reconstruct the full projection as `P.U * (P * A)`.

When k == m this reduces to the standard DEIM projector with an exact solve.

# Construction
    ObliqueProjector(U, k, rng=Random.default_rng())
"""
struct ObliqueProjector{T<:AbstractFloat, QRType}
    U  :: Matrix{T}    # n×m, orthonormal columns
    S  :: Vector{Int}  # k selected row indices (1-based), k ≥ m
    QR :: QRType       # QR factorisation of P_S U (k×m)
end

function ObliqueProjector(U::Matrix{T}, k::Int,
                          rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n, m = size(U)
    @assert k >= m "k must be >= rank of the basis m"
    @assert k <= n "k must be <= ambient dimension n"

    # ARP operates on a matrix with orthonormal rows; U' is m×n.
    S = select_subset(U', k, rng)

    PSU = U[S, :]          # k×m
    F   = qr(PSU, ColumnNorm())   # pivoted QR for stable least-squares

    ObliqueProjector{T, typeof(F)}(U, S, F)
end


# ── duck-typed row-extraction interface ───────────────────────────────────────

"""
    getrows(A, S)

Extract rows indexed by `S` from `A`. Specialise for custom matrix types
(e.g. TTtensor) by adding a method.
"""
getrows(A::AbstractMatrix, S::Vector{Int}) = A[S, :]
getcols(A::AbstractMatrix, S::Vector{Int}) = A[:, S]


# ── multiplication ────────────────────────────────────────────────────────────

"""
    *(P, A)

Left-multiply: returns the m×… coefficient matrix C = (P_S U)† P_S A in the
U-basis. Reconstruct the full projection as `P.U * (P * A)`.
"""
function Base.:*(P::ObliqueProjector, A)
    PSA = getrows(A, P.S)   # k×(cols of A)
    return P.QR \ PSA       # (P_S U)† (P_S A) — m×(cols of A)
end

"""
    *(A, P)

Right-multiply: returns the …×m coefficient matrix C = A P_S^T (P_S U)†^T in
the U-basis. Reconstruct the full projection as `(A * P) * P.U'`.
"""
function Base.:*(A, P::ObliqueProjector)
    APS = getcols(A, P.S)   # (rows of A)×k
    return (P.QR \ APS')'   # (P_S U)†^T (P_S A^T)^T — (rows of A)×m
end

end # module ObliqueProjectors

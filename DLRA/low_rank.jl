module LowRank

using LinearAlgebra
import LinearAlgebra: rank

export LRMat, dense, zero_lr, from_dense, from_factors, const_lr, round_lr, hadamard_lr

# ------------------------------------------------------------
# Low-rank matrix in factored form:
#
#   A ≈ U * S * V'
#
#   U'U = I  (n1×r, orthonormal columns)
#   V'V = I  (n2×r, orthonormal columns)
#   S        (r×r, dense, nondegenerate)
# ------------------------------------------------------------

struct LRMat{T<:AbstractFloat}
    U :: Matrix{T}
    S :: Matrix{T}
    V :: Matrix{T}

    function LRMat(U::AbstractMatrix{T}, S::AbstractMatrix{T},
                   V::AbstractMatrix{T}) where {T<:AbstractFloat}
        r = size(S, 1)
        size(S, 2) == r       || throw(DimensionMismatch("S must be square"))
        size(U, 2) == r       || throw(DimensionMismatch("size(U,2) must equal size(S,1)"))
        size(V, 2) == r       || throw(DimensionMismatch("size(V,2) must equal size(S,1)"))
        new{T}(Matrix{T}(U), Matrix{T}(S), Matrix{T}(V))
    end
end

Base.size(A::LRMat) = (size(A.U, 1), size(A.V, 1))
rank(A::LRMat)      = size(A.S, 1)

dense(A::LRMat) = A.U * A.S * A.V'

function zero_lr(::Type{T}, m::Int, n::Int) where {T<:AbstractFloat}
    return LRMat(zeros(T, m, 0), zeros(T, 0, 0), zeros(T, n, 0))
end
zero_lr(m::Int, n::Int) = zero_lr(Float64, m, n)

function choose_rank(s::AbstractVector;
                     rtol=1e-10, atol=0.0, maxrank=typemax(Int))
    isempty(s) && return 0
    threshold = max(eltype(s)(atol), eltype(s)(rtol) * s[1])
    return min(count(σ -> σ > threshold, s), maxrank, length(s))
end

# Reorthogonalise U*S*V' (U, V not necessarily orthonormal) and truncate.
function reorth(U::AbstractMatrix{T}, S::AbstractMatrix{T}, V::AbstractMatrix{T};
                rtol=T(1e-12), atol=T(0), maxrank=typemax(Int)) where {T<:AbstractFloat}
    m, n = size(U, 1), size(V, 1)
    r    = size(S, 1)
    @assert size(U, 2) == r && size(V, 2) == r

    r == 0 && return zero_lr(T, m, n)

    FU = qr(U);  FV = qr(V)
    ku = min(size(U)...);  kv = min(size(V)...)
    Qu = Matrix(FU.Q)[:, 1:ku];  Ru = Matrix(FU.R)[1:ku, :]
    Qv = Matrix(FV.Q)[:, 1:kv];  Rv = Matrix(FV.R)[1:kv, :]

    F  = svd(Ru * S * Rv')
    k  = choose_rank(F.S; rtol=rtol, atol=atol, maxrank=maxrank)
    k == 0 && return zero_lr(T, m, n)

    return LRMat(Qu * F.U[:, 1:k],
                 Diagonal(F.S[1:k]) |> Matrix,
                 Qv * F.V[:, 1:k])
end

function from_dense(A::AbstractMatrix{T};
                    rtol=T(1e-12), atol=T(0), maxrank=typemax(Int)) where {T<:AbstractFloat}
    F = svd(A)
    k = choose_rank(F.S; rtol=rtol, atol=atol, maxrank=maxrank)
    k == 0 && return zero_lr(T, size(A, 1), size(A, 2))
    return LRMat(F.U[:, 1:k], Diagonal(F.S[1:k]) |> Matrix, F.V[:, 1:k])
end
from_dense(A::AbstractMatrix) = from_dense(Matrix{Float64}(A))

function from_factors(U::AbstractMatrix{T}, S::AbstractMatrix{T},
                      V::AbstractMatrix{T};
                      rtol=T(1e-12), atol=T(0), maxrank=typemax(Int)) where {T<:AbstractFloat}
    return reorth(U, S, V; rtol=rtol, atol=atol, maxrank=maxrank)
end

function const_lr(c::T, m::Int, n::Int) where {T<:AbstractFloat}
    U = ones(T, m, 1) ./ T(sqrt(m))
    V = ones(T, n, 1) ./ T(sqrt(n))
    S = fill(c * T(sqrt(m)) * T(sqrt(n)), 1, 1)
    return LRMat(U, S, V)
end
const_lr(c::Real, m::Int, n::Int) = const_lr(Float64(c), m, n)

# ------------------------------------------------------------
# Basic algebra
# ------------------------------------------------------------

function Base.:+(A::LRMat{T}, B::LRMat{T}) where {T}
    @assert size(A) == size(B)
    U = hcat(A.U, B.U)                          # n1×(rA+rB)
    V = hcat(A.V, B.V)                          # n2×(rA+rB)
    S = cat(A.S, B.S; dims=(1,2))               # (rA+rB)×(rA+rB) block-diagonal
    return reorth(U, S, V)
end

Base.:-(A::LRMat) = LRMat(A.U, -A.S, A.V)
Base.:-(A::LRMat, B::LRMat) = A + (-B)

function Base.:*(α::Real, A::LRMat{T}) where {T}
    α == 0 && return zero_lr(T, size(A)...)
    return LRMat(A.U, T(α) .* A.S, A.V)
end
Base.:*(A::LRMat, α::Real) = α * A

function round_lr(A::LRMat{T};
                  rtol=T(1e-12), atol=T(0), maxrank=typemax(Int)) where {T}
    return reorth(A.U, A.S, A.V; rtol=rtol, atol=atol, maxrank=maxrank)
end

function hadamard_lr(A::LRMat{T}, B::LRMat{T};
                     rtol=T(1e-10), atol=T(0), maxrank=typemax(Int)) where {T}
    @assert size(A) == size(B)
    m, n = size(A)
    rA, rB = rank(A), rank(B)
    (rA == 0 || rB == 0) && return zero_lr(T, m, n)

    rc = rA * rB
    U  = zeros(T, m, rc)
    V  = zeros(T, n, rc)
    Sv = zeros(T, rc)

    k = 0
    for i in 1:rA, j in 1:rB
        k += 1
        U[:, k] .= A.U[:, i] .* B.U[:, j]
        V[:, k] .= A.V[:, i] .* B.V[:, j]
        # S is diagonal after reorth, so A.S[i,i] * B.S[j,j]
        Sv[k] = A.S[i, i] * B.S[j, j]
    end

    return reorth(U, Diagonal(Sv) |> Matrix, V; rtol=rtol, atol=atol, maxrank=maxrank)
end

end # module LowRank

module LowRank

using LinearAlgebra
import LinearAlgebra: rank

export LRMat, dense, zero_lr, from_dense, from_factors, const_lr, round_lr, hadamard_lr

# ------------------------------------------------------------
# Low-rank matrix in canonical SVD-like form:
#
# A ≈ U * Diagonal(S) * V'
#
# U'U = I
# V'V = I
# S[i] >= 0
# ------------------------------------------------------------

struct LRMat
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}

    function LRMat(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix)
        size(U, 2) == length(S) || throw(DimensionMismatch("size(U,2) must equal length(S)"))
        size(V, 2) == length(S) || throw(DimensionMismatch("size(V,2) must equal length(S)"))
        all(σ -> σ >= 0, S) || throw(ArgumentError("singular values must be nonnegative"))

        return new(Matrix{Float64}(U), Vector{Float64}(S), Matrix{Float64}(V))
    end
end

Base.size(A::LRMat) = (size(A.U, 1), size(A.V, 1))
rank(A::LRMat) = length(A.S)

function dense(A::LRMat)
    return A.U * Diagonal(A.S) * A.V'
end

function zero_lr(m::Int, n::Int)
    return LRMat(zeros(m, 0), Float64[], zeros(n, 0))
end

function choose_rank(s::AbstractVector; 
                     rtol=1e-10, atol=0.0, maxrank=typemax(Int))

    isempty(s) && return 0
    threshold = max(atol, rtol * s[1])
    k = count(σ -> σ > threshold, s)
    return min(k, maxrank, length(s))
end

function reorth(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix;
                rtol=1e-12, atol=0.0, maxrank=typemax(Int))

    m = size(U, 1)
    n = size(V, 1)
    r = length(S)

    @assert size(U, 2) == r
    @assert size(V, 2) == r

    if r == 0
        return zero_lr(m, n)
    end

    # QR-разложения неортогональных факторов
    FU = qr(U)
    FV = qr(Matrix{Float64}(V))

    ku = min(size(U)...)
    kv = min(size(V)...)

    Qu = Matrix(FU.Q[:, 1:ku])
    Qv = Matrix(FV.Q[:, 1:kv])

    Ru = Matrix(FU.R[1:ku, :])
    Rv = Matrix(FV.R[1:kv, :])

    # A = Qu * (Ru * Diagonal(S) * Rv') * Qv'
    M = Ru * Diagonal(S) * Rv'
    F = svd(M)
    k = choose_rank(F.S; rtol=rtol, atol=atol, maxrank=maxrank)

    if k == 0
        return zero_lr(m, n)
    end

    Unew = Qu * F.U[:, 1:k]
    Snew = F.S[1:k]
    Vnew = Qv * F.V[:, 1:k]

    return LRMat(Unew, Snew, Vnew)
end

function from_dense(A::AbstractMatrix; 
                    rtol=1e-12, atol=0.0, maxrank=typemax(Int))

    F = svd(Matrix{Float64}(A))
    k = choose_rank(F.S; rtol=rtol, atol=atol, maxrank=maxrank)

    if k == 0
        return zero_lr(size(A, 1), size(A, 2))
    end

    return LRMat(F.U[:, 1:k], F.S[1:k], F.V[:, 1:k])
end

function from_factors(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix;
                      rtol=1e-12, atol=0.0, maxrank=typemax(Int))

    return reorth(U, S, V;
                rtol=rtol,
                atol=atol,
                maxrank=maxrank)
end

function const_lr(c::Real, m::Int, n::Int)
    U = reshape(Float64.(ones(m)), :, 1)
    V = reshape(Float64.(ones(n)), :, 1)
    S = [Float64(c)]
    return from_factors(U, S, V)
end

# ------------------------------------------------------------
# Basic algebra
# ------------------------------------------------------------
function Base.:+(A::LRMat, B::LRMat)
    @assert size(A) == size(B)

    U = hcat(A.U, B.U)
    V = hcat(A.V, B.V)
    S = vcat(A.S, B.S)

    return reorth(U, S, V)
end

function Base.:-(A::LRMat)
    return LRMat(-A.U, A.S, A.V)
end

function Base.:-(A::LRMat, B::LRMat)
    return A + (-B)
end

function Base.:*(α::Real, A::LRMat)
    α = Float64(α)

    if α == 0.0
        m, n = size(A)
        return zero_lr(m, n)
    elseif α > 0.0
        return LRMat(A.U, α .* A.S, A.V)
    else
        return LRMat(-A.U, (-α) .* A.S, A.V)
    end
end

Base.:*(A::LRMat, α::Real) = α * A

function round_lr(A::LRMat;
                  rtol=1e-12, atol=0.0, maxrank=typemax(Int))

    return reorth(A.U, A.S, A.V;
                  rtol=rtol, atol=atol, maxrank=maxrank)
end

function hadamard_lr(A::LRMat, B::LRMat;
                     rtol=1e-10, atol=0.0, maxrank=typemax(Int))

    @assert size(A) == size(B)
    m, n = size(A)
    rA = rank(A)
    rB = rank(B)

    if rA == 0 || rB == 0
        return zero_lr(m, n)
    end

    U = zeros(Float64, m, rA * rB)
    V = zeros(Float64, n, rA * rB)
    S = zeros(Float64, rA * rB)

    k = 0

    for i in 1:rA
        for j in 1:rB
            k += 1

            U[:, k] .= A.U[:, i] .* B.U[:, j]
            V[:, k] .= A.V[:, i] .* B.V[:, j]
            S[k] = A.S[i] * B.S[j]
        end
    end

    return reorth(U, S, V;
                  rtol=rtol,
                  atol=atol,
                  maxrank=maxrank)
end

end
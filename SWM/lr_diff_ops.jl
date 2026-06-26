module LRDiffOps

using LinearAlgebra
using ..LowRank

export periodic_cdiff_columns, diff_x, diff_y, diff_lr

function periodic_cdiff_columns(X::AbstractMatrix, dx::Real)
    N, r = size(X)
    Y = zeros(Float64, N, r)

    if N <= 1 || r == 0
        return Y
    end

    inv2dx = 1.0 / (2.0 * Float64(dx))

    @inbounds for k in 1:r
        Y[1, k] = (X[2, k] - X[N, k]) * inv2dx
        
        for i in 2:N-1
            Y[i, k] = (X[i+1, k] - X[i-1, k]) * inv2dx
        end
        
        Y[N, k] = (X[1, k] - X[N-1, k]) * inv2dx
    end

    return Y
end

function diff_x(A::LowRank.LRMat, dx::Real;
                rtol=1e-12, atol=0.0, maxrank=typemax(Int))

    dU = periodic_cdiff_columns(A.U, dx)

    return LowRank.reorth(dU, A.S, A.V;
                          rtol=rtol, atol=atol, maxrank=maxrank)
end

function diff_y(A::LowRank.LRMat, dy::Real;
                rtol=1e-10, atol=0.0, maxrank=typemax(Int))

    dV = periodic_cdiff_columns(A.V, dy)

    return LowRank.canonicalize(A.U, A.S, dV;
                                rtol=rtol, atol=atol, maxrank=maxrank)
end

function diff_(A::LowRank.LRMat, dir::AbstractString, d::Real;
               rtol=1e-10, atol=0.0, maxrank=typemax(Int))

    if dir == "x"
        return diff_x(A, d; rtol=rtol, atol=atol, maxrank=maxrank)
    elseif dir == "y"
        return diff_y(A, d; rtol=rtol, atol=atol, maxrank=maxrank)
    else
        throw(ArgumentError("dir must be x or y direction!"))
    end
end

end
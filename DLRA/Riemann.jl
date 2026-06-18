export tangent_components, tangent_proj, Retraction, riemann_grad, get_element, PQ

function tangent_components(U::CoreCell, V::CoreCell, PQZ::Vector{Float64}, Q::Vector{Vector{Int}})
    d  = length(U)
    dU = CoreCell(undef, d)
    C  = CoreCell(undef, d)
    UL = MatrixCell(undef, d)
    k  = 1

    for mu = 1 : d
        C[mu] = zeros(size(U[mu]))
    end

    for j in Q
        UL[1] = U[1][:, j[1], :]
        for mu = 2 : d - 1
            UL[mu] = UL[mu - 1] * U[mu][:, j[mu], :]
        end

        VR = V[d][:, j[d], :]
        C[d][:, j[d], :] = C[d][:, j[d], :] + PQZ[k] * (UL[d - 1]')
        for mu = d - 1 : -1 : 2
            C[mu][:, j[mu], :] = C[mu][:, j[mu], :] + PQZ[k] * (UL[mu - 1]') * (VR')
            VR = V[mu][:, j[mu], :] * VR
        end
        C[1][:, j[1], :] = C[1][:, j[1], :] + PQZ[k] * (VR')
        k = k + 1

        dU[d] = C[d]
        for mu = 1 : d - 1
            CLmu = reshape(C[mu], size(C[mu], 1)*size(C[mu], 2), size(C[mu], 3))
            ULmu = reshape(U[mu], size(U[mu], 1)*size(U[mu], 2), size(U[mu], 3))
            dU[mu] = reshape(CLmu - ULmu * (ULmu') * CLmu, size(U[mu],1), size(U[mu],2), size(U[mu], 3))
        end
    end

    return dU
end

#calculate of projection on tangent space (sparse format)
function tangent_proj(X::TTtensor, PQY::Vector{<:Float64}, Q::Vector{Vector{Int}})
    d  = length(size(X))
    U  = reorth(X, "left").cores
    V  = reorth(X, "right").cores
    dU = tangent_components(U, V, PQY, Q)
    W  = CoreCell(undef, d)

    W[1] = cat(dU[1], U[1], dims=3)
    W[d] = cat(V[d], dU[d], dims=1)
    for k = 2 : d - 1
        top  = cat(V[k], zeros(size(V[k])), dims=3)
        bot  = cat(dU[k], U[k], dims=3)
        W[k] = cat(top, bot, dims=1)
    end

    return TTtensor(W)
end

#retruction on manyfold for tt-rank(X)
function Retraction(X::TTtensor, xi::TTtensor; tolerance=1e-14)
    return TTsvd(X + xi; tol_rel = tolerance)
end

function riemann_grad(X, Grad, Q)
    return tangent_proj(X, Grad, Q)
end

#compute element in full tensor
function get_element(Z::TTtensor, j::Vector{Int64})
    W = Z.cores
    d = length(size(Z))
    elem = W[1][:, j[1], :]
    for mu = 2 : d
        elem = elem * W[mu][:, j[mu], :]
    end
    return elem[1]
end

#projection on target set  (sparse format)
function PQ(X::TTtensor, Q::Vector{Vector{Int}})
    PQX = zeros(length(Q))
    k = 1
    for j in Q
        PQX[k] = get_element(X, j)
        k = k + 1
    end
    return PQX
end

function vector_transport(X::TTtensor, xi::TTtensor, Q::Vector{Vector{Int}})
    PQxi = PQ(xi, Q)
    return tangent_proj(X, PQxi, Q)
end
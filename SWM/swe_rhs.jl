module SWERhs

using LinearAlgebra
using ..LowRank
# Field-indexed RHS accessors consumed by the KSL solver.
import ..KSLMatrixSolver: getrows, getcols, getblock, update_state

export SWEParams, SWEState, swe_rhs

# Field indices into the state [u, v, h].
const IU = 1
const IV = 2
const IH = 3

"""
    SWEParams

Parameters for the vector-invariant shallow water equations on a doubly-periodic
N×N grid with uniform spacing `dx`.

- `dx`   : grid spacing (same in x and y)
- `grav` : gravitational acceleration g
- `fcor` : Coriolis parameter f (constant)
"""
struct SWEParams{T<:AbstractFloat}
    dx   :: T
    grav :: T
    fcor :: T
end
SWEParams(; dx, grav, fcor) = SWEParams(promote(float(dx), float(grav), float(fcor))...)

"""
    SWEState

RHS of the vector-invariant shallow water equations (cf. neuro_swm.jl lines
312–320):

    du/dt = -∂x(KE) + Q·v
    dv/dt = -∂y(KE) - Q·u
    dh/dt = -( ∂x(u·h) + ∂y(v·h) )

with  KE = ½(u² + v²) + g·h  and  Q = ∂x(v) - ∂y(u) + f.

Here ∂x is the periodic centered difference along the rows (axis 1, the U
factor) and ∂y along the columns (axis 2, the V factor) — matching `dx_i` /
`dy_j` in the reference and `diff_x` / `diff_y` in `lr_diff_ops.jl`.

Holds the current snapshot of the three fields as low-rank matrices. The field
being integrated is refreshed via `update_state(G, f, Y_f)`; the others stay
frozen for the duration of a timestep (frozen-snapshot coupling).
"""
struct SWEState{T<:AbstractFloat}
    u :: LRMat{T}
    v :: LRMat{T}
    h :: LRMat{T}
    p :: SWEParams{T}
end

swe_rhs(u::LRMat{T}, v::LRMat{T}, h::LRMat{T}, p::SWEParams{T}) where {T} =
    SWEState{T}(u, v, h, p)

# update_state: replace the working value of field f, leave the others frozen.
function update_state(G::SWEState{T}, f::Int, Y::LRMat{T}) where {T}
    f == IU && return SWEState{T}(Y,   G.v, G.h, G.p)
    f == IV && return SWEState{T}(G.u, Y,   G.h, G.p)
    f == IH && return SWEState{T}(G.u, G.v, Y,   G.p)
    throw(ArgumentError("field index must be 1 (u), 2 (v) or 3 (h)"))
end

# ── densify-on-demand primitives ─────────────────────────────────────────────

# Rows `I` (all n columns) of a low-rank field:  A[I, :] = U[I,:] * S * V'
rows_of(A::LRMat, I) = A.U[I, :] * A.S * A.V'          # |I|×n
# Cols `J` (all n rows) of a low-rank field:  A[:, J] = U * S * V[J,:]'
cols_of(A::LRMat, J) = A.U * A.S * A.V[J, :]'          # n×|J|

# Periodic neighbour indices (i-1, i+1) mod n.
@inline nbr(i, n) = (mod(i - 2, n) + 1, mod(i, n) + 1)

# Periodic centered difference of a dense matrix along axis 1 (rows) / 2 (cols).
function cdiff1(X::AbstractMatrix{T}, dx::T) where {T}
    n = size(X, 1)
    return (X[[mod(i, n) + 1     for i in 1:n], :] .-
            X[[mod(i - 2, n) + 1 for i in 1:n], :]) ./ (2dx)
end
function cdiff2(X::AbstractMatrix{T}, dx::T) where {T}
    n = size(X, 2)
    return (X[:, [mod(j, n) + 1     for j in 1:n]] .-
            X[:, [mod(j - 2, n) + 1 for j in 1:n]]) ./ (2dx)
end

# ── RHS evaluation over a ROW set (rows I, all columns) → |I|×n ───────────────
#
# ∂x at rows I needs rows I±1; ∂y stays within the extracted rows.

function rhs_rows(G::SWEState{T}, f::Int, I) where {T}
    dx, g, fc = G.p.dx, G.p.grav, G.p.fcor
    n = size(G.u, 1)
    Im = [nbr(i, n)[1] for i in I]
    Ip = [nbr(i, n)[2] for i in I]

    if f == IU
        # -∂x KE + Q·v
        KE_m = ke_rows(G, Im); KE_p = ke_rows(G, Ip)
        dxKE = (KE_p .- KE_m) ./ (2dx)
        Q    = (rows_of(G.v, Ip) .- rows_of(G.v, Im)) ./ (2dx) .-   # ∂x v
               cdiff2(rows_of(G.u, I), dx) .+ fc                    # -∂y u + f
        return .-dxKE .+ Q .* rows_of(G.v, I)

    elseif f == IV
        # -∂y KE - Q·u
        dyKE = cdiff2(ke_rows(G, I), dx)
        Q    = (rows_of(G.v, Ip) .- rows_of(G.v, Im)) ./ (2dx) .-
               cdiff2(rows_of(G.u, I), dx) .+ fc
        return .-dyKE .- Q .* rows_of(G.u, I)

    elseif f == IH
        # -( ∂x(u·h) + ∂y(v·h) )
        uh_m = rows_of(G.u, Im) .* rows_of(G.h, Im)
        uh_p = rows_of(G.u, Ip) .* rows_of(G.h, Ip)
        dx_uh = (uh_p .- uh_m) ./ (2dx)
        dy_vh = cdiff2(rows_of(G.v, I) .* rows_of(G.h, I), dx)
        return .-(dx_uh .+ dy_vh)
    else
        throw(ArgumentError("field index must be 1 (u), 2 (v) or 3 (h)"))
    end
end

ke_rows(G::SWEState, rowset) = begin
    u = rows_of(G.u, rowset); v = rows_of(G.v, rowset); h = rows_of(G.h, rowset)
    0.5 .* (u .^ 2 .+ v .^ 2) .+ G.p.grav .* h
end

# ── RHS evaluation over a COLUMN set (cols J, all rows) → n×|J| ───────────────
#
# ∂y at cols J needs cols J±1; ∂x stays within the extracted columns.

function rhs_cols(G::SWEState{T}, f::Int, J) where {T}
    dx, g, fc = G.p.dx, G.p.grav, G.p.fcor
    n = size(G.u, 2)
    Jm = [nbr(j, n)[1] for j in J]
    Jp = [nbr(j, n)[2] for j in J]

    if f == IU
        dxKE = cdiff1(ke_cols(G, J), dx)
        Q    = cdiff1(cols_of(G.v, J), dx) .-                       # ∂x v
               (cols_of(G.u, Jp) .- cols_of(G.u, Jm)) ./ (2dx) .+   # -∂y u
               fc
        return .-dxKE .+ Q .* cols_of(G.v, J)

    elseif f == IV
        KE_m = ke_cols(G, Jm); KE_p = ke_cols(G, Jp)
        dyKE = (KE_p .- KE_m) ./ (2dx)
        Q    = cdiff1(cols_of(G.v, J), dx) .-
               (cols_of(G.u, Jp) .- cols_of(G.u, Jm)) ./ (2dx) .+ fc
        return .-dyKE .- Q .* cols_of(G.u, J)

    elseif f == IH
        dx_uh = cdiff1(cols_of(G.u, J) .* cols_of(G.h, J), dx)
        vh_m = cols_of(G.v, Jm) .* cols_of(G.h, Jm)
        vh_p = cols_of(G.v, Jp) .* cols_of(G.h, Jp)
        dy_vh = (vh_p .- vh_m) ./ (2dx)
        return .-(dx_uh .+ dy_vh)
    else
        throw(ArgumentError("field index must be 1 (u), 2 (v) or 3 (h)"))
    end
end

ke_cols(G::SWEState, colset) = begin
    u = cols_of(G.u, colset); v = cols_of(G.v, colset); h = cols_of(G.h, colset)
    0.5 .* (u .^ 2 .+ v .^ 2) .+ G.p.grav .* h
end

# ── KSL accessor interface ───────────────────────────────────────────────────

getrows(G::SWEState,  f::Int, I, t::Real)    = rhs_rows(G, f, I)      # |I|×n
getcols(G::SWEState,  f::Int, J, t::Real)    = rhs_cols(G, f, J)      # n×|J|
getblock(G::SWEState, f::Int, I, J, t::Real) = rhs_rows(G, f, I)[:, J]  # |I|×|J|

end # module SWERhs

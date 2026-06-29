module KSLMatrixSolver

using LinearAlgebra
using ..LowRank
using ..ObliqueProjectors

export Euler, RK4, LieTrotter, Strang, ksl_step
export getrows, getcols, getblock, update_state

# ── RHS interface ─────────────────────────────────────────────────────────────
#
# The state is a vector of fields  Y = [Y_1, …, Y_F], each Y_f ≈ U_f S_f V_f'.
# Field f's equation  dY_f/dt = G_f(Y, t)  may depend on every field.
#
# A concrete RHS object G holds the current snapshot of all fields and exposes,
# for each field index f:
#
#   getrows(G, f, I, t)      -> n1×|I|   rows I of G_f(Y, t),    I: row indices
#   getcols(G, f, J, t)      -> n2×|J|   cols J of G_f(Y, t),    J: col indices
#   getblock(G, f, I, J, t)  -> |I|×|J|  submatrix (I,J) of G_f(Y, t)
#   update_state(G, f, Y_f)  -> G′       set field f's working value to Y_f
#
# Coupling is frozen-snapshot per timestep: all fields are advanced from the
# same snapshot; update_state only refreshes the field currently integrating,
# while the other fields stay frozen.
#
# The accessors are generic functions with no built-in methods; concrete RHS
# types add methods for them. They are declared here so other modules can import
# and extend them.
function getrows  end
function getcols  end
function getblock end

# The default update_state is a no-op for stateless RHS objects.
function update_state end
update_state(G, f, Y) = G

# ── Inner steppers (plain matrix ODE  dX/dt = F(X,t)) ────────────────────────

struct Euler end
struct RK4   end

function plain_step(::Euler, F, X0::Matrix, t0::Real, dt::Real)
    return X0 + dt * F(X0, t0)
end

function plain_step(::RK4, F, X0::Matrix, t0::Real, dt::Real)
    k1 = F(X0,               t0)
    k2 = F(X0 + dt/2 * k1,  t0 + dt/2)
    k3 = F(X0 + dt/2 * k2,  t0 + dt/2)
    k4 = F(X0 + dt    * k3, t0 + dt)
    return X0 + dt/6 * (k1 + 2k2 + 2k3 + k4)
end

# ── Outer splittings ──────────────────────────────────────────────────────────

struct LieTrotter end
struct Strang     end

# ── KSL substeps (operate on a single field f) ───────────────────────────────

# K-step (eq. 9): integrate  dK/dt = G_f(:, J, t) * [V(J,:)]^{-T}
# P_J = ObliqueProjector(V0, k_K): selects J (column indices of Y_f, row indices of V).
# J is fixed for the whole step; inner ODE re-evaluates G_f at current K*V0'.
# Returns (U1, R1) from QR of K(t1).
function _k_step(Y::LRMat{T}, G, f::Int, t0::Real, dt::Real,
                 inner, P_J::ObliqueProjector) where {T}
    J    = P_J.S                             # column indices of Y_f (row indices of V)
    VJ   = Y.V[J, :]                         # k_K×r
    VJf  = qr(VJ, ColumnNorm())              # factorised V[J,:] for right solve

    K0   = Y.U * Y.S                         # n1×r

    F_K = (K, t) -> begin
        G′  = update_state(G, f, LRMat(K, Matrix{T}(I, size(K,2), size(K,2)), Y.V))
        GJ  = getcols(G′, f, J, t)           # n1×k_K: cols J of G_f(K*V0', t)
        Matrix((VJf \ GJ')')                 # n1×r: G_f(:,J) * (V[J,:])^{-T}
    end

    K1  = plain_step(inner, F_K, K0, t0, dt)
    F1  = qr(K1)
    return Matrix(F1.Q), Matrix(F1.R)        # U1 (n1×r), R1 (r×r)
end

# S-step (eq. 10): integrate  dS̃/dt = -[U1(I,:)]^{-1} * G_f(I,J,t) * [V0(J,:)]^{-T}
# P_I = ObliqueProjector(U1, k_K): selects I (row indices of Y_f).
# P_J = ObliqueProjector(V0, k_L): selects J (column indices of Y_f).
# Both fixed; inner ODE re-evaluates G_f at U1*S̃*V0'.
# Returns S̃(t1).
function _s_step(U1::Matrix{T}, R1::Matrix{T}, V0::Matrix{T},
                 G, f::Int, t0::Real, dt::Real,
                 inner, P_I::ObliqueProjector, P_J::ObliqueProjector) where {T}
    I    = P_I.S                             # row indices of Y_f
    J    = P_J.S                             # column indices of Y_f
    U1I  = U1[I, :]                          # k_K×r
    V0J  = V0[J, :]                          # k_L×r
    U1If = qr(U1I, ColumnNorm())
    V0Jf = qr(V0J, ColumnNorm())

    F_S = (S, t) -> begin
        G′  = update_state(G, f, LRMat(U1, S, V0))
        GB  = getblock(G′, f, I, J, t)       # k_K×k_L
        lhs = U1If \ GB                      # r×k_L
        -Matrix((V0Jf \ lhs')')              # r×r
    end

    return plain_step(inner, F_S, R1, t0, dt)   # S̃1, r×r
end

# L-step (eq. 11): integrate  dL/dt = G_f(I,:,t)^T * [U1(I,:)]^{-T}
# P_I = ObliqueProjector(U1, k_L): selects I (row indices of Y_f).
# I is fixed; inner ODE re-evaluates G_f at U1*L'.
# Returns (V1, S1) from QR of L(t1),  L1 = V1*R1  =>  Y_f = U1*R1^T*V1'.
function _l_step(U1::Matrix{T}, S̃1::Matrix{T}, V0::Matrix{T},
                 G, f::Int, t0::Real, dt::Real,
                 inner, P_I::ObliqueProjector) where {T}
    I    = P_I.S                             # row indices of Y_f
    U1I  = U1[I, :]                          # k_L×r
    U1If = qr(U1I, ColumnNorm())

    L0   = V0 * S̃1'                         # n2×r

    F_L = (L, t) -> begin
        # current field is Y_f = U1 * L'  =  U1 * I_r * L', factored U=U1, S=I, V=L
        Ir  = Matrix{T}(LinearAlgebra.I, size(L,2), size(L,2))
        G′  = update_state(G, f, LRMat(U1, Ir, L))
        GI  = getrows(G′, f, I, t)          # k_L×n2: rows I of G_f(U1*L', t)
        Matrix((U1If \ GI)')                # n2×r: G_f(I,:)^T * (U1[I,:])^{-T}
    end

    L1  = plain_step(inner, F_L, L0, t0, dt)
    F1  = qr(L1)
    V1  = Matrix(F1.Q)
    S1  = Matrix(F1.R)'                     # L1 = V1*R1  =>  Y_f = U1*S1*V1', S1 = R1^T
    return V1, S1
end

# ── Single-field splittings ───────────────────────────────────────────────────

# Lie-Trotter: K → S → L
function _lie_trotter(Y::LRMat, G, f::Int, t::Real, dt::Real, inner, k_K::Int, k_L::Int)
    P_J  = ObliqueProjector(Y.V, k_K)       # column projector, fixed from V0

    U1, R1 = _k_step(Y, G, f, t, dt, inner, P_J)

    P_I    = ObliqueProjector(U1, k_K)      # row projector from updated U1
    P_J_L  = ObliqueProjector(Y.V, k_L)    # column projector for S- and L-steps

    S̃1     = _s_step(U1, R1, Y.V, G, f, t, dt, inner, P_I, P_J_L)
    V1, S1 = _l_step(U1, S̃1, Y.V, G, f, t, dt, inner, P_I)

    return LRMat(U1, S1, V1)
end

# Strang: K/2 → S/2 → L → S/2 → K/2  (symmetric, 2nd-order)
function _strang(Y::LRMat, G, f::Int, t::Real, dt::Real, inner, k_K::Int, k_L::Int)
    h = dt / 2

    # Forward half-sweep
    P_J0   = ObliqueProjector(Y.V, k_K)
    U1, R1 = _k_step(Y, G, f, t, h, inner, P_J0)

    P_I1    = ObliqueProjector(U1, k_K)
    P_J0_L  = ObliqueProjector(Y.V, k_L)
    S̃1      = _s_step(U1, R1, Y.V, G, f, t, h, inner, P_I1, P_J0_L)
    V1, S1  = _l_step(U1, S̃1, Y.V, G, f, t, h, inner, P_I1)

    Ymid = LRMat(U1, S1, V1)
    G    = update_state(G, f, Ymid)

    # Backward half-sweep (symmetric)
    P_I1_S  = ObliqueProjector(U1, k_K)
    P_J1_L  = ObliqueProjector(V1, k_L)
    S̃2      = _s_step(U1, S1, V1, G, f, t + h, h, inner, P_I1_S, P_J1_L)
    V2, S2  = _l_step(U1, S̃2, V1, G, f, t + h, h, inner, P_I1_S)

    Ymid2 = LRMat(U1, S2, V2)
    G     = update_state(G, f, Ymid2)

    P_J2   = ObliqueProjector(V2, k_K)
    U2, R2 = _k_step(LRMat(U1, S2, V2), G, f, t + h, h, inner, P_J2)

    P_I2   = ObliqueProjector(U2, k_L)
    V3, S3 = _l_step(U2, R2, V2, G, f, t + h, h, inner, P_I2)

    return LRMat(U2, S3, V3)
end

_step_field(::LieTrotter, Y, G, f, t, dt, inner, k_K, k_L) =
    _lie_trotter(Y, G, f, t, dt, inner, k_K, k_L)
_step_field(::Strang, Y, G, f, t, dt, inner, k_K, k_L) =
    _strang(Y, G, f, t, dt, inner, k_K, k_L)

# ── Public entry point ────────────────────────────────────────────────────────

"""
    ksl_step(Ys, G, t, dt, outer, inner, k_K, k_L) -> Vector{LRMat}

Advance a vector of low-rank fields `Ys = [Y_1, …, Y_F]`, each `Y_f ≈ U_f*S_f*V_f'`,
by one timestep `dt` using the interpolatory projector-splitting (KSL) integrator.

Fields are advanced independently from a frozen snapshot of the whole state
taken at the start of the step (frozen-snapshot coupling). The RHS `G` couples
the fields and is queried per field.

Arguments:
- `G`    : RHS object holding the current snapshot; must implement
           `getrows(G,f,I,t)`, `getcols(G,f,J,t)`, `getblock(G,f,I,J,t)`.
           Optionally implement `update_state(G,f,Y_f)`.
           Convention: I are row indices, J are column indices of field f.
- `outer`: `LieTrotter()` (1st order) or `Strang()` (2nd order).
- `inner`: plain-matrix ODE stepper — `Euler()` or `RK4()`.
- `k_K`  : column index-set size for K/S-steps. Scalar (shared) or per-field vector (≥ rank).
- `k_L`  : row index-set size for L/S-steps. Scalar (shared) or per-field vector (≥ rank).
"""
function ksl_step(Ys::AbstractVector{<:LRMat}, G, t::Real, dt::Real,
                  outer, inner,
                  k_K::Union{Int,AbstractVector{Int}},
                  k_L::Union{Int,AbstractVector{Int}})
    F   = length(Ys)
    kKs = k_K isa Int ? fill(k_K, F) : k_K
    kLs = k_L isa Int ? fill(k_L, F) : k_L

    out = Vector{eltype(Ys)}(undef, F)
    for f in 1:F
        out[f] = _step_field(outer, Ys[f], G, f, t, dt, inner, kKs[f], kLs[f])
    end
    return out
end

end # module KSLMatrixSolver

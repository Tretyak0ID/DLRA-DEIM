module KSLMatrixSolver

using LinearAlgebra
using ..ObliqueProjectors

export KSLState, Euler, RK4, LieTrotter, Strang, ksl_step
export getrows, getcols, getblock, update_state

# ── State ─────────────────────────────────────────────────────────────────────

"""
    KSLState{T}

Low-rank matrix Y ≈ U * S * V' in factored form.

- `U`: n1×r, orthonormal columns
- `S`: r×r, dense (not necessarily diagonal)
- `V`: n2×r, orthonormal columns
"""
struct KSLState{T<:AbstractFloat}
    U :: Matrix{T}
    S :: Matrix{T}
    V :: Matrix{T}
end

# ── RHS interface ─────────────────────────────────────────────────────────────
#
# Implement for a concrete G representing G(Y, t):
#
#   getrows(G, I, t)       -> n1×|I|   rows I of G(Y, t),      I: row indices
#   getcols(G, J, t)       -> n2×|J|   cols J of G(Y, t),      J: col indices
#   getblock(G, I, J, t)   -> |I|×|J|  submatrix at (I, J) of G(Y, t)
#   update_state(G, Y)     -> G′       return RHS with internal state set to Y
#
# The default update_state is a no-op for stateless RHS objects.

update_state(G, ::KSLState) = G

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

# ── KSL substeps ─────────────────────────────────────────────────────────────

# K-step (eq. 9): integrate  dK/dt = G_K(:, J, t) * [V(J,:)]^{-T}
# P_J = ObliqueProjector(V0, k_K): selects J (column indices of Y, row indices of V).
# J is fixed for the whole step; inner ODE re-evaluates G at current K*V0'.
# Returns (U1, R1) from QR of K(t1).
function _k_step(Y::KSLState{T}, G, t0::Real, dt::Real,
                 inner, P_J::ObliqueProjector) where {T}
    J   = P_J.S                              # column indices of Y (row indices of V)
    VJ  = Y.V[J, :]                          # k_K×r
    VJf = qr(VJ, ColumnNorm())               # factorised V[J,:] for right solve

    K0  = Y.U * Y.S                          # n1×r

    F_K = (K, t) -> begin
        G′  = update_state(G, KSLState(K, Matrix{T}(I, size(K,2), size(K,2)), Y.V))
        GJ  = getcols(G′, J, t)              # n1×k_K: cols J of G(K*V0', t)
        Matrix((VJf \ GJ')')                 # n1×r: G(:,J) * (V[J,:])^{-T}
    end

    K1  = plain_step(inner, F_K, K0, t0, dt)
    F1  = qr(K1)
    return Matrix(F1.Q), Matrix(F1.R)        # U1 (n1×r), R1 (r×r)
end

# S-step (eq. 10): integrate  dS̃/dt = -[U1(I,:)]^{-1} * G_S(I,J,t) * [V0(J,:)]^{-T}
# P_I = ObliqueProjector(U1, k_K): selects I (row indices of Y, row indices of U).
# P_J = ObliqueProjector(V0, k_L): selects J (column indices of Y, row indices of V).
# Both fixed; inner ODE re-evaluates G at U1*S̃*V0'.
# Returns S̃(t1).
function _s_step(U1::Matrix{T}, R1::Matrix{T}, V0::Matrix{T},
                 G, t0::Real, dt::Real,
                 inner, P_I::ObliqueProjector, P_J::ObliqueProjector) where {T}
    I   = P_I.S                              # row indices of Y
    J   = P_J.S                              # column indices of Y
    U1I = U1[I, :]                           # k_K×r
    V0J = V0[J, :]                           # k_L×r
    U1If = qr(U1I, ColumnNorm())
    V0Jf = qr(V0J, ColumnNorm())

    F_S = (S, t) -> begin
        G′ = update_state(G, KSLState(U1, S, V0))
        GB = getblock(G′, I, J, t)           # k_K×k_L
        lhs = U1If \ GB                      # r×k_L
        -Matrix((V0Jf \ lhs')')              # r×r
    end

    return plain_step(inner, F_S, R1, t0, dt)   # S̃1, r×r
end

# L-step (eq. 11): integrate  dL/dt = G_L(I,:,t)^T * [U1(I,:)]^{-T}
# P_I = ObliqueProjector(U1, k_L): selects I (row indices of Y).
# I is fixed; inner ODE re-evaluates G at U1*L'.
# Returns (V1, S1) from QR of L(t1),  L1 = V1*R1  =>  Y = U1*R1^T*V1'.
function _l_step(U1::Matrix{T}, S̃1::Matrix{T}, V0::Matrix{T},
                 G, t0::Real, dt::Real,
                 inner, P_I::ObliqueProjector) where {T}
    I    = P_I.S                             # row indices of Y
    U1I  = U1[I, :]                          # k_L×r
    U1If = qr(U1I, ColumnNorm())

    L0   = V0 * S̃1'                         # n2×r

    F_L = (L, t) -> begin
        G′  = update_state(G, KSLState(U1, L', V0))
        GI  = getrows(G′, I, t)             # k_L×n2: rows I of G(U1*L', t)
        Matrix((U1If \ GI)')                # n2×r: G(I,:)^T * (U1[I,:])^{-T}
    end

    L1  = plain_step(inner, F_L, L0, t0, dt)
    F1  = qr(L1)
    V1  = Matrix(F1.Q)
    S1  = Matrix(F1.R)'                     # L1 = V1*R1  =>  Y = U1*S1*V1', S1 = R1^T
    return V1, S1
end

# ── Lie-Trotter: K → S → L ───────────────────────────────────────────────────

function _lie_trotter(Y::KSLState, G, t::Real, dt::Real, inner, k_K::Int, k_L::Int)
    P_J      = ObliqueProjector(Y.V, k_K)   # column projector, fixed from V0
    P_I_S    = ObliqueProjector(Y.U, k_K)   # placeholder; recomputed after K-step

    U1, R1   = _k_step(Y, G, t, dt, inner, P_J)

    P_I      = ObliqueProjector(U1, k_K)    # row projector from updated U1
    P_J_S    = ObliqueProjector(Y.V, k_L)   # column projector for S- and L-step

    S̃1       = _s_step(U1, R1, Y.V, G, t, dt, inner, P_I, P_J_S)
    V1, S1   = _l_step(U1, S̃1, Y.V, G, t, dt, inner, P_I)

    return KSLState(U1, S1, V1)
end

# ── Strang: K/2 → S/2 → L → S/2 → K/2  (symmetric, 2nd-order) ───────────────

function _strang(Y::KSLState, G, t::Real, dt::Real, inner, k_K::Int, k_L::Int)
    h = dt / 2

    # Forward half-sweep
    P_J0     = ObliqueProjector(Y.V, k_K)
    U1, R1   = _k_step(Y, G, t, h, inner, P_J0)

    P_I1     = ObliqueProjector(U1, k_K)
    P_J0_L   = ObliqueProjector(Y.V, k_L)
    S̃1       = _s_step(U1, R1, Y.V, G, t, h, inner, P_I1, P_J0_L)
    V1, S1   = _l_step(U1, S̃1, Y.V, G, t, h, inner, P_I1)

    Ymid = KSLState(U1, S1, V1)
    G    = update_state(G, Ymid)

    # Backward half-sweep (symmetric)
    P_J1     = ObliqueProjector(V1, k_K)
    P_I1_S   = ObliqueProjector(U1, k_K)
    P_J1_L   = ObliqueProjector(V1, k_L)
    S̃2       = _s_step(U1, S1, V1, G, t + h, h, inner, P_I1_S, P_J1_L)
    V2, S2   = _l_step(U1, S̃2, V1, G, t + h, h, inner, P_I1_S)

    Ymid2 = KSLState(U1, S2, V2)
    G     = update_state(G, Ymid2)

    P_J2     = ObliqueProjector(V2, k_K)
    U2, R2   = _k_step(KSLState(U1, S2, V2), G, t + h, h, inner, P_J2)

    P_I2     = ObliqueProjector(U2, k_L)
    V3, S3   = _l_step(U2, R2, V2, G, t + h, h, inner, P_I2)

    return KSLState(U2, S3, V3)
end

# ── Public entry point ────────────────────────────────────────────────────────

"""
    ksl_step(Y, G, t, dt, outer, inner, k_K, k_L) -> KSLState

Advance the low-rank state `Y ≈ U*S*V'` by one timestep `dt` using the
interpolatory projector-splitting (KSL) integrator.

Arguments:
- `G`   : RHS object; must implement `getrows(G,I,t)`, `getcols(G,J,t)`,
          `getblock(G,I,J,t)`. Optionally implement `update_state(G,Y)`.
          Convention: I are row indices of Y, J are column indices of Y.
- `outer`: `LieTrotter()` (1st order) or `Strang()` (2nd order).
- `inner`: plain-matrix ODE stepper — `Euler()` or `RK4()`.
- `k_K` : size of column index set J for K- and S-steps (≥ r).
- `k_L` : size of row index set I for L- and S-steps (≥ r).
"""
function ksl_step(Y::KSLState, G, t::Real, dt::Real,
                  outer::LieTrotter, inner, k_K::Int, k_L::Int)
    return _lie_trotter(Y, G, t, dt, inner, k_K, k_L)
end

function ksl_step(Y::KSLState, G, t::Real, dt::Real,
                  outer::Strang, inner, k_K::Int, k_L::Int)
    return _strang(Y, G, t, dt, inner, k_K, k_L)
end

end # module KSLMatrixSolver

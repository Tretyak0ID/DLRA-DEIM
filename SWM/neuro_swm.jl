# =====================================================================
# 2D shallow water (vector-invariant, leapfrog) in QTT format. Julia.
# Dependency-free: only the LinearAlgebra stdlib.
#
# 1:1 port of a NumPy version validated against dense references.
# Layout: field F[i,j] on N x N periodic grid (N = 2^L) stored as an
# L-core QTT, physical mode size 4 per level (level k carries bit k of
# i AND bit k of j, mode = 2*bit_i + bit_j), LSB-first.
#
# The QTT-specific piece is the periodic centered difference: the cyclic
# shift is a rank-2 QTT matrix built from a "+1 with carry" automaton
# (NOT a one-core circshift). selftest() checks every primitive against
# dense references on startup -- run it before trusting the dynamics.
# =====================================================================

using LinearAlgebra
using Printf

# Cores: vector core = Array{Float64,3} (rl,n,rr); matrix core = Array{Float64,4} (rl,nout,nin,rr).

# ---------- full N x N  <->  4-mode QTT tensor (explicit, convention-safe) ----------
function full_to_qtt_tensor(F::AbstractMatrix, L::Int)
    T = Array{Float64}(undef, ntuple(_->4, L))
    @inbounds for idx in CartesianIndices(T)
        bi = 0; bj = 0
        for k in 1:L
            m = idx[k] - 1            # 0..3
            bi += (m >> 1) << (k-1)   # high bit of mode -> i
            bj += (m & 1)  << (k-1)   # low  bit of mode -> j
        end
        T[idx] = F[bi+1, bj+1]
    end
    return T
end

function qtt_tensor_to_full(T::AbstractArray, L::Int)
    N = 2^L
    F = Array{Float64}(undef, N, N)
    @inbounds for idx in CartesianIndices(T)
        bi = 0; bj = 0
        for k in 1:L
            m = idx[k] - 1
            bi += (m >> 1) << (k-1)
            bj += (m & 1)  << (k-1)
        end
        F[bi+1, bj+1] = T[idx]
    end
    return F
end

# ---------- rank truncation helper ----------
function trunc_rank(S::Vector{Float64}, delta::Float64, rmax::Int)
    cum = 0.0; r = 1
    for i in length(S):-1:1
        cum += S[i]^2
        if cum > delta^2
            r = i; break
        end
    end
    return max(1, min(r, rmax))
end

# ---------- TT-SVD: full tensor -> cores ----------
function tt_svd(T::AbstractArray, eps::Float64, rmax::Int)
    shp = size(T); d = length(shp)
    cores = Vector{Array{Float64,3}}()
    r = 1
    C = reshape(copy(T), 1, :)
    nrm = norm(T); delta = eps / sqrt(max(d-1,1)) * (nrm > 0 ? nrm : 1.0)
    for k in 1:d-1
        n = shp[k]
        C = reshape(C, r*n, :)
        F = svd(C)
        rr = trunc_rank(F.S, delta, rmax)
        push!(cores, reshape(F.U[:, 1:rr], r, n, rr))
        C = Diagonal(F.S[1:rr]) * F.Vt[1:rr, :]
        r = rr
    end
    push!(cores, reshape(C, r, shp[d], 1))
    return cores
end

# ---------- rounding ----------
function tt_round(cores::Vector{Array{Float64,3}}, eps::Float64, rmax::Int)
    d = length(cores)
    cores = [copy(c) for c in cores]
    d == 1 && return cores
    # right-to-left orthogonalization
    for k in d:-1:2
        rl, n, rr = size(cores[k])
        M = reshape(cores[k], rl, n*rr)
        F = qr(Matrix(M'))                 # M' = Q R
        s = min(n*rr, rl)
        Qt = Matrix(F.Q)[:, 1:s]
        Rt = Matrix(F.R)[1:s, :]
        cores[k] = reshape(Matrix(Qt'), s, n, rr)
        rlm, nm, _ = size(cores[k-1])
        cores[k-1] = reshape(reshape(cores[k-1], rlm*nm, :) * Matrix(Rt'), rlm, nm, s)
    end
    nrm = norm(cores[1]); delta = eps / sqrt(d-1) * (nrm > 0 ? nrm : 1.0)
    # left-to-right SVD truncation
    for k in 1:d-1
        rl, n, rr = size(cores[k])
        F = svd(reshape(cores[k], rl*n, rr))
        rr2 = trunc_rank(F.S, delta, rmax)
        cores[k] = reshape(F.U[:, 1:rr2], rl, n, rr2)
        SV = Diagonal(F.S[1:rr2]) * F.Vt[1:rr2, :]      # rr2 x rr
        rk1l, nk1, rk1r = size(cores[k+1])
        cores[k+1] = reshape(SV * reshape(cores[k+1], rk1l, nk1*rk1r), rr2, nk1, rk1r)
    end
    return cores
end

# ---------- arithmetic ----------
function tt_full(cores::Vector{Array{Float64,3}})
    C = cores[1]                                   # (1,n1,r1)
    acc = reshape(C, size(C,2), size(C,3))         # (n1, r1)
    dims = [size(cores[1],2)]
    for k in 2:length(cores)
        rl, n, rr = size(cores[k])
        acc = reshape(acc, :, rl) * reshape(cores[k], rl, n*rr)  # (prod x n*rr)
        acc = reshape(acc, :, rr)
        push!(dims, n)
    end
    return reshape(acc, ntuple(i->dims[i], length(dims)))
end

function had(a, b)                                 # Hadamard product
    out = Vector{Array{Float64,3}}()
    for (A, B) in zip(a, b)
        ra, n, rb = size(A); rc, _, rd = size(B)
        H = Array{Float64}(undef, ra*rc, n, rb*rd)
        for i in 1:n
            H[:, i, :] = kron(A[:, i, :], B[:, i, :])
        end
        push!(out, H)
    end
    return out
end

function add(a, b)
    d = length(a); out = Vector{Array{Float64,3}}()
    for k in 1:d
        A = a[k]; B = b[k]; ra, n, rb = size(A); rc, _, rd = size(B)
        if k == 1
            C = zeros(1, n, rb+rd); C[:, :, 1:rb] = A; C[:, :, rb+1:end] = B
        elseif k == d
            C = zeros(ra+rc, n, 1); C[1:ra, :, :] = A; C[ra+1:end, :, :] = B
        else
            C = zeros(ra+rc, n, rb+rd); C[1:ra, :, 1:rb] = A; C[ra+1:end, :, rb+1:end] = B
        end
        push!(out, C)
    end
    return out
end

smul(a, c::Float64) = (out = [copy(x) for x in a]; out[1] = out[1] .* c; out)
axpby(al, x, be, y, eps, rmax) = tt_round(add(smul(x, al), smul(y, be)), eps, rmax)
ones_qtt(L::Int) = [ones(1, 4, 1) for _ in 1:L]
add_const(x, c, eps, rmax) = tt_round(add(x, smul(ones_qtt(length(x)), c)), eps, rmax)

function tt_dot(a, b)
    Lm = ones(1, 1)
    for (A, B) in zip(a, b)
        rAl, n, rAr = size(A); rBl, _, rBr = size(B)
        tmp = zeros(rAr, rBr)
        for i in 1:n
            tmp += A[:, i, :]' * Lm * B[:, i, :]
        end
        Lm = tmp
    end
    return Lm[1, 1]
end

# ---------- TT-matrix x TT-vector ----------
function matvec(Ac, x)
    out = Vector{Array{Float64,3}}()
    for (A, X) in zip(Ac, x)
        Ra, no, ni, Rb = size(A); ra, _, rb = size(X)
        Y = zeros(Ra*ra, no, Rb*rb)
        for o in 1:no
            acc = zeros(Ra*ra, Rb*rb)
            for ii in 1:ni
                acc += kron(A[:, o, ii, :], X[:, ii, :])
            end
            Y[:, o, :] = acc
        end
        push!(out, Y)
    end
    return out
end

# ---------- periodic shift as rank-2 QTT matrices ----------
function succ_auto(L::Int)                          # "+1" successor (LSB-first)
    cores = Vector{Array{Float64,4}}()
    G = zeros(1, 2, 2, 2)                            # LSB: carry-in = 1
    for old in 0:1
        G[1, ((old+1)%2)+1, old+1, old+1] = 1.0
    end
    push!(cores, G)
    for _ in 2:L-1
        G = zeros(2, 2, 2, 2)
        for cin in 0:1, old in 0:1
            G[cin+1, ((old+cin)%2)+1, old+1, (old & cin)+1] = 1.0
        end
        push!(cores, G)
    end
    G = zeros(2, 2, 2, 1)                            # MSB: drop carry (periodic)
    for cin in 0:1, old in 0:1
        G[cin+1, ((old+cin)%2)+1, old+1, 1] = 1.0
    end
    push!(cores, G)
    return cores
end

transpose_op(cores) = [permutedims(c, (1, 3, 2, 4)) for c in cores]

function lift_i(ac)                                 # automaton on i (high bit), id on j (low bit)
    out = Vector{Array{Float64,4}}()
    for A in ac
        cl, _, _, cr = size(A)
        G = zeros(cl, 4, 4, cr)
        for c in 1:cl, cp in 1:cr, bio in 0:1, bii in 0:1, bj in 0:1
            G[c, 2*bio+bj+1, 2*bii+bj+1, cp] = A[c, bio+1, bii+1, cp]
        end
        push!(out, G)
    end
    return out
end

function lift_j(ac)                                 # id on i (high bit), automaton on j (low bit)
    out = Vector{Array{Float64,4}}()
    for A in ac
        cl, _, _, cr = size(A)
        G = zeros(cl, 4, 4, cr)
        for c in 1:cl, cp in 1:cr, bi in 0:1, bjo in 0:1, bji in 0:1
            G[c, 2*bi+bjo+1, 2*bi+bji+1, cp] = A[c, bjo+1, bji+1, cp]
        end
        push!(out, G)
    end
    return out
end

struct Diff
    dx::Float64
    i_succ; i_pred; j_succ; j_pred
end
function Diff(L::Int, dx::Float64)
    succ = succ_auto(L); pred = transpose_op(succ)
    Diff(dx, lift_i(succ), lift_i(pred), lift_j(succ), lift_j(pred))
end
dx_i(D::Diff, x, eps, rmax) = axpby( 1/(2D.dx), matvec(D.i_pred, x),
                                    -1/(2D.dx), matvec(D.i_succ, x), eps, rmax)
dy_j(D::Diff, x, eps, rmax) = axpby( 1/(2D.dx), matvec(D.j_pred, x),
                                    -1/(2D.dx), matvec(D.j_succ, x), eps, rmax)

maxrank(c) = length(c) > 1 ? maximum(size(c[k],1) for k in 2:length(c)) : 1
nparams(c) = sum(length(x) for x in c)

# ---------- self-test against dense references ----------
function selftest()
    L = 6; N = 2^L; dx = 1.0
    xs = collect(0:N-1)
    F = [sin(2π*i/N)*cos(2π*j/N) + 0.7 for i in xs, j in xs]
    c = tt_round(tt_svd(full_to_qtt_tensor(F, L), 1e-12, 10^9), 1e-12, 10^9)
    e1 = norm(qtt_tensor_to_full(tt_full(c), L) - F)
    D = Diff(L, dx)
    tofull(z) = qtt_tensor_to_full(tt_full(z), L)
    diq = tofull(dx_i(D, c, 1e-12, 10^9))
    dref = (circshift(F, (-1,0)) - circshift(F, (1,0))) / (2dx)
    e2 = norm(diq - dref)
    djq = tofull(dy_j(D, c, 1e-12, 10^9))
    dref2 = (circshift(F, (0,-1)) - circshift(F, (0,1))) / (2dx)
    e3 = norm(djq - dref2)
    e4 = norm(tofull(tt_round(had(c, c), 1e-12, 10^9)) - F .* F)
    e5 = abs(tt_dot(c, c) - sum(F .* F))
    @printf("selftest  reshape=%.1e  d/dx=%.1e  d/dy=%.1e  hadamard=%.1e  dot=%.1e\n", e1, e2, e3, e4, e5)
    ok = maximum((e1, e2, e3, e4, e5)) < 1e-9
    println(ok ? "selftest PASSED" : "selftest FAILED -- do not trust the run")
    return ok
end

# ---------- shallow water driver ----------
function main()
    selftest() || return

    L = 8; N = 2^L
    radz = 6371.22e3; grav = 9.80616; omega = 7.292e-5; pcori = 2*omega
    H0 = 10e3; Lx = 2π*radz; dx = Lx/N
    dt = 0.45 * dx / sqrt(grav*H0)
    eps = 1e-4; rmax = 64
    nsteps = 8415; report_every = 50

    xs = collect(0:N-1) .* dx
    U0 = 20.0
    u0 = [U0*sin(2π*y/Lx)                         for x in xs, y in xs]
    v0 = [0.5*cos(2π*x/Lx)*sin(2π*y/Lx)           for x in xs, y in xs]
    h0 = [H0 + (pcori/grav)*(U0*Lx/(2π))*cos(2π*y/Lx) for x in xs, y in xs]

    toq(F)  = tt_round(tt_svd(full_to_qtt_tensor(F, L), eps, rmax), eps, rmax)
    tofull(c) = qtt_tensor_to_full(tt_full(c), L)
    D = Diff(L, dx); R(c) = tt_round(c, eps, rmax)

    u = toq(u0); v = toq(v0); h = toq(h0)
    uo, vo, ho = u, v, h
    ones_v = ones_qtt(L); mass0 = tt_dot(h, ones_v)

    @printf("N=%d  dt=%.1fs  eps=%g  rmax=%d  steps=%d\n", N, dt, eps, rmax, nsteps)
    @printf("%6s %9s %9s %9s %9s %7s %9s\n",
            "step","qtt_rmax","qtt_par","mat_rank","mat_par","ratio","d_mass")
    t0 = time()
    for s in 1:nsteps
        uu = R(had(u, u)); vv = R(had(v, v))
        KE = R(add(smul(R(add(uu, vv)), 0.5), smul(h, grav)))
        Q  = add_const(R(add(dx_i(D, v, eps, rmax), smul(dy_j(D, u, eps, rmax), -1.0))), pcori, eps, rmax)
        qv = R(had(Q, v)); qu = R(had(Q, u)); uh = R(had(u, h)); vh = R(had(v, h))

        un = axpby(1.0, uo, 2dt, R(add(smul(dx_i(D, KE, eps, rmax), -1.0), qv)), eps, rmax)
        vn = axpby(1.0, vo, 2dt, R(add(smul(dy_j(D, KE, eps, rmax), -1.0), smul(qu, -1.0))), eps, rmax)
        hn = axpby(1.0, ho, -2dt, R(add(dx_i(D, uh, eps, rmax), dy_j(D, vh, eps, rmax))), eps, rmax)

        uo, vo, ho = u, v, h
        u, v, h = un, vn, hn

        if report_every != 0 && s % report_every == 0
            qrm = max(maxrank(h), maxrank(u), maxrank(v))
            qpar = nparams(h)
            Ff = tofull(h)
            sv = svdvals(Ff)
            rmat = count(>(eps*sv[1]), sv)
            mpar = 2*N*rmat
            dmass = abs(tt_dot(h, ones_v) - mass0) / abs(mass0)
            @printf("%6d %9d %9d %9d %9d %6.2fx %9.1e\n",
                    s, qrm, qpar, rmat, mpar, mpar/qpar, dmass)
            if !all(isfinite, Ff)
                println("  ** non-finite -- unstable, stopping **"); break
            end
        end
    end
    @printf("done in %.1fs\n", time() - t0)
end

main()
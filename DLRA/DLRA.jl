# =====================================================================
# DLRA entry point.
#
# Pattern A: this is the single place that `include`s the DLRA source
# files, in dependency order. Include this file once (from a session or
# from a higher-level entry file such as SWM/SWM.jl); the individual
# source files do not include each other.
#
#   include(joinpath(@__DIR__, "..", "DLRA", "DLRA.jl"))
#   using .LowRank            # the module defined in low_rank.jl
#
# Each file is included only once here, so the modules/functions they
# define exist as a single copy (no duplicate-type pitfalls).
# =====================================================================

# Each file defines one module. After including this file, reach an API with
# `using .ModuleName` (one dot); from a sibling module use `..ModuleName`.
# Order matters: ObliqueProjectors uses `..SelectSubset`, so SelectSubset is
# included first.
include(joinpath(@__DIR__, "low_rank.jl"))           # module LowRank
include(joinpath(@__DIR__, "select_subset.jl"))      # module SelectSubset
include(joinpath(@__DIR__, "oblique_projector.jl"))  # module ObliqueProjectors

# NOTE: Riemann.jl / riemann_ops.jl are QTT-tensor code with undefined
# dependencies (CoreCell, MatrixCell, a `reorth` returning `.cores`) and
# are not wired up yet. Add them here once their deps are in place.

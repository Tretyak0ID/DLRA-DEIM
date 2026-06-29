# =====================================================================
# SWM entry point.
#
# Pattern A: includes the DLRA entry file first (which loads `LowRank`
# and the DLRA helpers), then the SWM source files, in dependency order.
# Include this file once:
#
#   include(joinpath(@__DIR__, "SWM.jl"))
#   using .LowRank, .LRDiffOps, .InitialConditions
#
# Because everything is included into one enclosing scope, the SWM modules
# can reach LowRank as a sibling module via `..LowRank` (see lr_diff_ops.jl
# and initial_conditions.jl). The individual files do not include each
# other.
# =====================================================================

# DLRA dependencies (defines `module LowRank` + helpers as siblings).
include(joinpath(@__DIR__, "..", "DLRA", "DLRA.jl"))

# SWM modules. lr_diff_ops and initial_conditions both use `..LowRank`.
include(joinpath(@__DIR__, "lr_diff_ops.jl"))
include(joinpath(@__DIR__, "initial_conditions.jl"))

# NOTE: swe_rhs.jl is currently empty; neuro_swm.jl is a standalone QTT
# script (no module). Add them here once they are ready.

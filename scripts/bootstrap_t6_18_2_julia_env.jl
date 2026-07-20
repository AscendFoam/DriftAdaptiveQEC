#!/usr/bin/env julia

using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OFFICIAL_REPO = joinpath(REPO_ROOT, "third_party", "LatticeAlgorithms.jl")

isdir(OFFICIAL_REPO) || error("Pinned official repository is missing: $(OFFICIAL_REPO)")
Pkg.develop(path=OFFICIAL_REPO)
# The official Project declares PythonCall 0.9.10 compatibility but ships no
# Manifest.  Lock the lower compatible release: 0.9.35 fails under the
# required --compiled-modules=no path on Julia 1.10 with a world-age error.
Pkg.add(Pkg.PackageSpec(name="PythonCall", version="0.9.10"))
Pkg.instantiate(; allow_autoprecomp=false)

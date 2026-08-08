#!/usr/bin/env julia

using Pkg

Pkg.test("LatticeAlgorithms"; coverage=false)

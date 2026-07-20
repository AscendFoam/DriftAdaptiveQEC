using Random
using Test

if length(ARGS) < 2
    error("usage: run_lattice_algorithms_seeded_tests.jl SEED TEST_FILE [TEST_FILE ...]")
end

seed = parse(Int, ARGS[1])
files = ARGS[2:end]
repo_root = normpath(joinpath(@__DIR__, ".."))
official_test_root = joinpath(repo_root, "third_party", "LatticeAlgorithms.jl", "test")

Random.seed!(seed)
for file in files
    path = joinpath(official_test_root, file)
    isfile(path) || error("missing official test file: $(path)")
    @testset "official seeded $(file) seed=$(seed)" begin
        include(path)
    end
end

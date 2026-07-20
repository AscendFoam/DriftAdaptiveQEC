using Dates
using JLD2
using JSON
using LinearAlgebra
using PythonCall
using Random
using SHA
using Statistics

using LatticeAlgorithms

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OFFICIAL_ROOT = joinpath(REPO_ROOT, "third_party", "LatticeAlgorithms.jl")
const OFFICIAL_COMMIT = "01f9bf1f6970b3e229b43aac9da3325c75518db8"
const DEFAULT_SEEDS = collect(61820001:61820032)
const DEFAULT_DISTANCES = [3, 5, 7]
const DEFAULT_SIGMAS = collect(0.56:0.01:0.64)

# PythonCall installs some conversion rules lazily.  The pinned package cannot
# be precompiled on Julia 1.10 because it includes utilities.jl twice, so calls
# that cross the Julia/Python boundary must enter the newest world explicitly.
official_surface_decode(x, xstabs, zstabs) =
    Base.invokelatest(LatticeAlgorithms.decode_surface_code, x, xstabs, zstabs)
official_mwpm(graph, syndrome) =
    Base.invokelatest(LatticeAlgorithms.mwpm, graph, syndrome)

function parse_args(args)
    opts = Dict{String,String}(
        "mode" => "all",
        "output" => joinpath(REPO_ROOT, "docs", "t6_18_2_julia_raw.json"),
        "trials-per-seed" => "2000",
        "bootstrap-resamples" => "2000",
        "pilot-trials" => "32",
    )
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected positional argument: $(args[i])")
        key = args[i][3:end]
        i == length(args) && error("missing value for --$(key)")
        opts[key] = args[i + 1]
        i += 2
    end
    return opts
end

sha256_file(path) = bytes2hex(open(sha256, path))

function git_head(path)
    return strip(read(`git -C $path rev-parse HEAD`, String))
end

function provenance()
    project = joinpath(REPO_ROOT, "configs", "literature", "t6_18_2_julia_env", "Project.toml")
    manifest = joinpath(REPO_ROOT, "configs", "literature", "t6_18_2_julia_env", "Manifest.toml")
    license = joinpath(OFFICIAL_ROOT, "LICENSE")
    notebook = joinpath(
        OFFICIAL_ROOT, "examples", "papers",
        "Closest_lattice_point_decoding_for_multimode_GKP_codes", "Fig_5.ipynb",
    )
    data = joinpath(
        OFFICIAL_ROOT, "examples", "papers",
        "Closest_lattice_point_decoding_for_multimode_GKP_codes", "data", "surface_codes",
        "fidelity_time_surf_3_29_1.0e7_with_std.jld2",
    )
    return Dict(
        "official_repository" => "https://github.com/amazon-science/LatticeAlgorithms.jl",
        "official_head" => git_head(OFFICIAL_ROOT),
        "expected_head" => OFFICIAL_COMMIT,
        "head_matches" => git_head(OFFICIAL_ROOT) == OFFICIAL_COMMIT,
        "license" => "Apache-2.0",
        "license_sha256" => sha256_file(license),
        "project_sha256" => sha256_file(project),
        "manifest_sha256" => sha256_file(manifest),
        "fig5_notebook_sha256" => sha256_file(notebook),
        "fig5_data_sha256" => sha256_file(data),
        "julia_version" => string(VERSION),
        "compiled_modules_disabled_required" => true,
    )
end

function brute_force_cvp(x::Vector{Float64}, basis::Matrix{Float64})
    # Row-basis convention: lattice points are B' * z.  A Babai candidate
    # supplies a finite upper radius; sigma_min(B) then proves a finite box
    # contains every point that can beat that candidate.
    z0 = round.(Int, transpose(basis) \ x)
    y0 = transpose(basis) * z0
    radius = norm(x - y0)
    sigma_min = minimum(svdvals(basis))
    sigma_min > 1e-10 || error("ill-conditioned exact-check basis")
    coord_bound = ceil(Int, (norm(x) + radius + 1e-12) / sigma_min)
    best_distance = radius
    best = copy(y0)
    ranges = ntuple(_ -> (-coord_bound:coord_bound), length(x))
    for tuple_z in Iterators.product(ranges...)
        z = collect(tuple_z)
        y = transpose(basis) * z
        distance = norm(x - y)
        if distance < best_distance - 1e-12
            best_distance = distance
            best = y
        end
    end
    return best, best_distance, coord_bound
end

function single_mode_ci(x::Float64)
    return sqrt(pi) * LatticeAlgorithms.closest_integer(x / sqrt(pi))
end

function exact_correctness_checks()
    rng = MersenneTwister(61820001)
    rows = Vector{Dict{String,Any}}()
    mismatches = 0
    max_distance_gap = 0.0
    max_box_bound = 0

    # Deterministic, well-conditioned integer bases.  Distances, rather than
    # a particular vector, are compared so exact Voronoi ties are accepted.
    samples_by_dimension = Dict(1 => 96, 2 => 96, 3 => 72, 4 => 48)
    for n in 1:4
        for sample in 1:samples_by_dimension[n]
            basis_int = Matrix{Int}(I, n, n)
            for i in 1:n, j in 1:n
                basis_int[i, j] += rand(rng, -1:1)
            end
            basis_int += n * Matrix{Int}(I, n, n)
            basis = Float64.(basis_int)
            x = 1.25 .* randn(rng, n)
            # The official generic sphere decoder indexes layer 2 after the
            # one-dimensional terminal layer and therefore does not support
            # n=1.  Use the package's exact scaled-Z implementation for that
            # declared small-dimension case; n=2..4 exercise closest_point.
            official = n == 1 ?
                LatticeAlgorithms.closest_point_scaled_Zn(x, basis[1, 1]) :
                LatticeAlgorithms.closest_point(x, basis)
            _, oracle_distance, box_bound = brute_force_cvp(x, basis)
            official_distance = norm(x - official)
            gap = abs(official_distance - oracle_distance)
            mismatch = gap > 1e-9
            mismatches += mismatch
            max_distance_gap = max(max_distance_gap, gap)
            max_box_bound = max(max_box_bound, box_bound)
            push!(rows, Dict(
                "dimension" => n,
                "sample" => sample,
                "official_distance" => official_distance,
                "oracle_distance" => oracle_distance,
                "distance_gap" => gap,
                "mismatch" => mismatch,
                "certified_integer_box_bound" => box_bound,
            ))
        end
    end

    boundary_offsets = [-2.0, -1.5, -1.0, -0.5, nextfloat(-0.5), prevfloat(0.5), 0.5, 1.0, 1.5, 2.0]
    single_mode_mismatches = 0
    single_mode_rows = Vector{Dict{String,Any}}()
    for offset in boundary_offsets
        x = offset * sqrt(pi)
        official = LatticeAlgorithms.closest_point_scaled_Zn([x], sqrt(pi))[1]
        ci = single_mode_ci(x)
        mismatch = !isapprox(official, ci; atol=1e-12, rtol=0.0)
        single_mode_mismatches += mismatch
        push!(single_mode_rows, Dict("offset" => offset, "official" => official, "ci" => ci, "mismatch" => mismatch))
    end

    # The fast surface decoder must agree with generic exact CVP at d=3.
    d = 3
    matrix = LatticeAlgorithms.surface_code_M(d)
    dual = LatticeAlgorithms.GKP_logical_operator_generator(matrix)
    basis = sqrt(2pi) * dual
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    surface_mismatches = 0
    surface_max_gap = 0.0
    for _ in 1:64
        x = 0.65 .* randn(rng, 2d^2)
        exact = LatticeAlgorithms.closest_point(x, basis)
        fast, _ = official_surface_decode(x, xstabs, zstabs)
        gap = norm(exact - fast)
        surface_max_gap = max(surface_max_gap, gap)
        surface_mismatches += gap > 1e-8
    end

    return Dict(
        "generic_samples" => length(rows),
        "generic_mismatches" => mismatches,
        "generic_max_distance_gap" => max_distance_gap,
        "generic_max_certified_box_bound" => max_box_bound,
        "single_mode_samples" => length(single_mode_rows),
        "single_mode_mismatches" => single_mode_mismatches,
        "surface_d3_samples" => 64,
        "surface_d3_mismatches" => surface_mismatches,
        "surface_d3_max_vector_gap" => surface_max_gap,
        "passed" => mismatches == 0 && single_mode_mismatches == 0 && surface_mismatches == 0,
        "generic_rows" => rows,
        "single_mode_rows" => single_mode_rows,
    )
end

function crossing(x1, x2, y1, y2, y3, y4)
    numerator_delta = y4 - y1 - y3 + y2
    return (x2 / (x2 - x1) * numerator_delta + y1 - y2) / (numerator_delta / (x2 - x1))
end

function adjacent_crossing(sigmas, curves, lower_index)
    difference = [curve[lower_index] - curve[lower_index + 1] for curve in curves]
    positive = findfirst(>(0), difference)
    if positive === nothing || positive == 1
        return nothing
    end
    x1, x2 = sigmas[positive - 1], sigmas[positive]
    y1 = curves[positive][lower_index + 1]
    y2 = curves[positive][lower_index]
    y3 = curves[positive - 1][lower_index]
    y4 = curves[positive - 1][lower_index + 1]
    return crossing(x1, x2, y1, y2, y3, y4)
end

function official_data_reanalysis()
    data_path = joinpath(
        OFFICIAL_ROOT, "examples", "papers",
        "Closest_lattice_point_decoding_for_multimode_GKP_codes", "data", "surface_codes",
        "fidelity_time_surf_3_29_1.0e7_with_std.jld2",
    )
    data = JLD2.load(data_path)
    sigmas = Float64.(collect(data["σrange"]))
    distances = Int.(collect(data["drange"]))
    exact_dict = data["p0list_exact"]
    analog_dict = data["p0list_non_exact"]
    exact_curves = [Float64.(exact_dict[sigma]) for sigma in sort(collect(keys(exact_dict)))]
    analog_curves = [Float64.(analog_dict[sigma]) for sigma in sort(collect(keys(analog_dict)))]
    sorted_sigmas = sort(sigmas)
    exact_crossings = [adjacent_crossing(sorted_sigmas, exact_curves, index) for index in 1:length(distances)-1]
    analog_crossings = [adjacent_crossing(sorted_sigmas, analog_curves, index) for index in 1:length(distances)-1]
    any(isnothing, exact_crossings) && error("missing exact crossing in official aggregate")
    any(isnothing, analog_crossings) && error("missing analog crossing in official aggregate")
    exact_values = Float64[value for value in exact_crossings]
    analog_values = Float64[value for value in analog_crossings]
    # Paper/notebook average crossings for lower distance d>13, i.e. index 6:end.
    exact_tail = exact_values[6:end]
    analog_tail = analog_values[6:end]
    exact_mean, exact_std = mean(exact_tail), std(exact_tail)
    analog_mean, analog_std = mean(analog_tail), std(analog_tail)
    return Dict(
        "evidence_class" => "OFFICIAL_AGGREGATE_DATA_REANALYSIS_NOT_INDEPENDENT_MONTE_CARLO",
        "data_path" => relpath(data_path, REPO_ROOT),
        "data_sha256" => sha256_file(data_path),
        "distances" => distances,
        "sigma_min" => minimum(sorted_sigmas),
        "sigma_max" => maximum(sorted_sigmas),
        "sigma_count" => length(sorted_sigmas),
        "declared_samples_per_point" => 10_000_000,
        "cpd_crossings" => exact_values,
        "analog_crossings" => analog_values,
        "tail_lower_distance_gt_13" => true,
        "cpd_threshold_mean" => exact_mean,
        "cpd_threshold_std" => exact_std,
        "analog_threshold_mean" => analog_mean,
        "analog_threshold_std" => analog_std,
        "notebook_anchor_cpd" => 0.6024563484296794,
        "notebook_anchor_analog" => 0.5995937637028759,
        "anchor_max_abs_gap" => max(abs(exact_mean - 0.6024563484296794), abs(analog_mean - 0.5995937637028759)),
        "passed" => max(abs(exact_mean - 0.6024563484296794), abs(analog_mean - 0.5995937637028759)) < 1e-12,
    )
end

function logsumexp(values::Vector{Float64})
    maximum_value = maximum(values)
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function conditional_logical_error_probability(residual::Float64, sigma::Float64; truncation::Int=12)
    scale = sqrt(pi)
    odd_terms = Float64[]
    all_terms = Float64[]
    for integer in -truncation:truncation
        term = -((residual - integer * scale)^2) / (2sigma^2)
        push!(all_terms, term)
        if isodd(integer)
            push!(odd_terms, term)
        end
    end
    return exp(logsumexp(odd_terms) - logsumexp(all_terms))
end

function analog_weight_validation()
    maximum_difference = 0.0
    minimum_probability = 1.0
    maximum_probability = 0.0
    for sigma in [0.56, 0.60, 0.64]
        for residual in range(-sqrt(pi) / 2, sqrt(pi) / 2; length=1001)
            p12 = conditional_logical_error_probability(residual, sigma; truncation=12)
            p20 = conditional_logical_error_probability(residual, sigma; truncation=20)
            maximum_difference = max(maximum_difference, abs(p12 - p20))
            minimum_probability = min(minimum_probability, p12)
            maximum_probability = max(maximum_probability, p12)
        end
    end
    return Dict(
        "equation" => "Noh-Chamberland-2020 Eq.(11); MWPM weight=-log2(p) as Appendix B",
        "truncation" => 12,
        "reference_truncation" => 20,
        "grid_points" => 3003,
        "max_probability_difference" => maximum_difference,
        "minimum_probability" => minimum_probability,
        "maximum_probability" => maximum_probability,
        "passed" => maximum_difference < 1e-14 && minimum_probability > 0.0 && maximum_probability <= 0.5 + 1e-12,
    )
end

function decode_analog_subspace(x::Vector{Float64}, stabilizers::Dict{Int64,Vector{Int64}}, sigma::Float64)
    scaled = x ./ sqrt(pi)
    closest = LatticeAlgorithms.closest_integer.(scaled)
    second = LatticeAlgorithms.second_closest_integer.(scaled)
    residuals = x .- closest .* sqrt(pi)
    weights = [-log2(conditional_logical_error_probability(residual, sigma)) for residual in residuals]

    real_vertices = length(stabilizers)
    boundary = real_vertices + 1
    keys_stabilizers = collect(keys(stabilizers))
    values_stabilizers = collect(values(stabilizers))
    edge_to_mode = Dict{Set{Int64},Tuple{Int64,Float64}}()
    for qubit in eachindex(x)
        membership = findall(qubit .∈ values_stabilizers)
        length(membership) > 2 && error("unsupported non-matching topology")
        isempty(membership) && continue
        vertices = length(membership) == 2 ?
            (keys_stabilizers[membership[1]], keys_stabilizers[membership[2]]) :
            (keys_stabilizers[membership[1]], boundary)
        key = Set(Int64[vertices[1], vertices[2]])
        if !haskey(edge_to_mode, key) || weights[qubit] < edge_to_mode[key][2]
            edge_to_mode[key] = (qubit, weights[qubit])
        end
    end

    syndrome = zeros(Int, boundary)
    for (index, stabilizer) in stabilizers
        syndrome[index] = mod(sum(closest[stabilizer]), 2)
    end
    syndrome[boundary] = mod(sum(syndrome), 2)

    graph = zeros(Float64, boundary, boundary)
    for (vertices, (_, weight)) in edge_to_mode
        i, j = collect(vertices)
        graph[i, j] = weight
        graph[j, i] = weight
    end
    paths = official_mwpm(graph, syndrome)
    correction = zeros(Int, length(x))
    for (i, j) in paths
        correction[edge_to_mode[Set(Int64[i, j])][1]] += 1
    end
    selected = [correction[i] == 1 ? second[i] : closest[i] for i in eachindex(x)]
    final = correction - mod.(closest, 2)
    return selected .* sqrt(pi), final
end

function decode_analog_surface(x::Vector{Float64}, d::Int, sigma::Float64)
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    yq, finalq = decode_analog_subspace(x[1:2:end], zstabs, sigma)
    yp, finalp = decode_analog_subspace(x[2:2:end], xstabs, sigma)
    y = zeros(Float64, length(x))
    final = zeros(Int, length(x))
    y[1:2:end], y[2:2:end] = yq, yp
    final[1:2:end], final[2:2:end] = finalq, finalp
    return y, final
end

function logical_bits_from_final(final, d)
    qfinal = final[1:2:end]
    pfinal = final[2:2:end]
    xbit = mod(sum(qfinal[LatticeAlgorithms.surface_code_Z_logicals(d)[1]]), 2)
    zbit = mod(sum(pfinal[LatticeAlgorithms.surface_code_X_logicals(d)[1]]), 2)
    return Int(xbit), Int(zbit)
end

function logical_bits_from_lattice_point(y, logical_transform)
    coordinates = logical_transform * y
    return mod(round(Int, coordinates[1]), 2), mod(round(Int, coordinates[2]), 2)
end

function final_list_validation()
    rng = MersenneTwister(61820002)
    mismatches = 0
    samples = 0
    for d in [3, 5]
        matrix = LatticeAlgorithms.surface_code_M(d)
        canonical_dual = LatticeAlgorithms.GKP_logical_operator_generator_canonical(matrix)
        transform = inv(transpose(sqrt(2pi) * canonical_dual))
        xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
        zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
        for sigma in [0.56, 0.60, 0.64], _ in 1:32
            x = sigma .* randn(rng, 2d^2)
            y, final = official_surface_decode(x, xstabs, zstabs)
            mismatches += logical_bits_from_final(final, d) != logical_bits_from_lattice_point(y, transform)
            ya, finala = decode_analog_surface(x, d, sigma)
            mismatches += logical_bits_from_final(finala, d) != logical_bits_from_lattice_point(ya, transform)
            samples += 2
        end
    end
    return Dict("samples" => samples, "mismatches" => mismatches, "passed" => mismatches == 0)
end

function simulate_cell(seed::Int, d::Int, sigma::Float64, trials::Int)
    # A stable cell-specific seed avoids Julia hash randomization and gives
    # paired CPD/analog instances without coupling different distances.
    cell_seed = seed * 100_000 + d * 100 + round(Int, 100sigma)
    rng = MersenneTwister(cell_seed)
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    cpd_errors = 0
    analog_errors = 0
    cpd_time = 0.0
    analog_time = 0.0
    cpd_allocated = 0
    analog_allocated = 0
    for trial in 1:trials
        x = sigma .* randn(rng, 2d^2)
        local cpd_y, cpd_final, analog_y, analog_final
        cpd_time += @elapsed cpd_y, cpd_final = official_surface_decode(x, xstabs, zstabs)
        analog_time += @elapsed analog_y, analog_final = decode_analog_surface(x, d, sigma)
        if trial == 1
            cpd_allocated = @allocated official_surface_decode(x, xstabs, zstabs)
            analog_allocated = @allocated decode_analog_surface(x, d, sigma)
        end
        cpd_errors += logical_bits_from_final(cpd_final, d) != (0, 0)
        analog_errors += logical_bits_from_final(analog_final, d) != (0, 0)
    end
    return Dict(
        "seed" => seed,
        "distance" => d,
        "sigma" => sigma,
        "trials" => trials,
        "cpd_errors" => cpd_errors,
        "analog_errors" => analog_errors,
        "cpd_ler" => cpd_errors / trials,
        "analog_ler" => analog_errors / trials,
        "cpd_runtime_seconds" => cpd_time,
        "analog_runtime_seconds" => analog_time,
        "cpd_seconds_per_decode" => cpd_time / trials,
        "analog_seconds_per_decode" => analog_time / trials,
        "cpd_allocated_bytes_first_measured_decode" => cpd_allocated,
        "analog_allocated_bytes_first_measured_decode" => analog_allocated,
        "paired_error_difference" => cpd_errors - analog_errors,
    )
end

function threshold_simulation(trials_per_seed::Int; pilot::Bool=false, pilot_trials::Int=32)
    trials = pilot ? pilot_trials : trials_per_seed
    seeds = pilot ? DEFAULT_SEEDS[1:2] : DEFAULT_SEEDS
    rows = Vector{Dict{String,Any}}()
    started = time()
    for d in DEFAULT_DISTANCES, sigma in DEFAULT_SIGMAS, seed in seeds
        push!(rows, simulate_cell(seed, d, sigma, trials))
    end
    return Dict(
        "evidence_class" => pilot ? "PILOT_RUNTIME_NOT_THRESHOLD_EVIDENCE" : "INDEPENDENT_PREREGISTERED_MONTE_CARLO",
        "distances" => DEFAULT_DISTANCES,
        "sigmas" => DEFAULT_SIGMAS,
        "seeds" => seeds,
        "trials_per_seed" => trials,
        "total_paired_trials" => sum(row["trials"] for row in rows),
        "wall_clock_seconds" => time() - started,
        "rows" => rows,
    )
end

function main()
    opts = parse_args(ARGS)
    mode = opts["mode"]
    allowed = Set(["correctness", "official-data", "pilot", "threshold", "all"])
    mode in allowed || error("unknown mode $(mode)")
    result = Dict{String,Any}(
        "schema_version" => "t6.18.2-julia-raw-v1",
        "task_id" => "T6.18.2",
        "generated_at" => string(now()),
        "requested_mode" => mode,
        "provenance" => provenance(),
    )
    if mode in ["correctness", "all"]
        result["exact_correctness"] = exact_correctness_checks()
        result["analog_weight_validation"] = analog_weight_validation()
        result["final_list_validation"] = final_list_validation()
    end
    if mode in ["official-data", "all"]
        result["official_data_reanalysis"] = official_data_reanalysis()
    end
    if mode == "pilot"
        result["pilot"] = threshold_simulation(parse(Int, opts["trials-per-seed"]); pilot=true, pilot_trials=parse(Int, opts["pilot-trials"]))
    elseif mode in ["threshold", "all"]
        result["threshold_simulation"] = threshold_simulation(parse(Int, opts["trials-per-seed"]))
    end
    output = normpath(opts["output"])
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, result, 2)
        write(io, "\n")
    end
    println(JSON.json(Dict(
        "task_id" => "T6.18.2",
        "mode" => mode,
        "output" => output,
        "completed_sections" => sort(collect(keys(result))),
    )))
end

main()

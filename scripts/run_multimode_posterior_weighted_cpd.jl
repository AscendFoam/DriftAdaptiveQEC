using Dates
using JSON
using LinearAlgebra
using PythonCall
using Random
using SHA
using Statistics

using LatticeAlgorithms

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_CONFIG = joinpath(REPO_ROOT, "configs", "literature", "t6_18_3_multimode_drift.json")
const SQRT_PI = sqrt(pi)

official_mwpm(graph, syndrome) = Base.invokelatest(LatticeAlgorithms.mwpm, graph, syndrome)

function parse_args(args)
    opts = Dict{String,String}(
        "config" => DEFAULT_CONFIG,
        "mode" => "formal",
        "seed-index-start" => "1",
        "seed-index-end" => "32",
        "cycles" => "0",
        "output" => joinpath(REPO_ROOT, "docs", "t6_18_3_multimode_drift_shard.json"),
    )
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected positional argument $(args[i])")
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

function coordinate_signs(d::Int)
    signs = zeros(Int, 2d^2)
    for mode in 1:d^2
        sign = isodd(mode) ? 1 : -1
        signs[2mode - 1] = sign
        signs[2mode] = -sign
    end
    @assert count(==(1), signs) == count(==(-1), signs)
    return signs
end

metric_precision(theta::Float64, signs::Vector{Int}) = cosh(theta) .* exp.(-signs .* theta)

function sigma_vector(base_sigma::Float64, theta::Float64, signs::Vector{Int})
    variance = base_sigma^2 .* exp.(signs .* theta) ./ cosh(theta)
    @assert isapprox(mean(variance), base_sigma^2; atol=1e-12, rtol=1e-12)
    return sqrt.(variance)
end

function logsumexp(values)
    maximum_value = maximum(values)
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function build_transition(grid::Vector{Float64}, cfg)
    n = length(grid)
    identity_mass = Float64(cfg["transition_identity_mass"])
    local_mass = Float64(cfg["transition_local_mass"])
    uniform_mass = Float64(cfg["transition_uniform_mass"])
    local_sigma = Float64(cfg["transition_local_sigma"])
    @assert isapprox(identity_mass + local_mass + uniform_mass, 1.0; atol=1e-12)
    transition = zeros(Float64, n, n)
    for i in 1:n
        local_kernel = exp.(-0.5 .* ((grid .- grid[i]) ./ local_sigma).^2)
        local_kernel ./= sum(local_kernel)
        transition[i, :] .= local_mass .* local_kernel .+ uniform_mass / n
        transition[i, i] += identity_mass
        transition[i, :] ./= sum(transition[i, :])
    end
    return transition
end

function periodic_residual_logdensity(residual::Float64, sigma::Float64, terms::Int)
    values = Float64[]
    for integer in -terms:terms
        push!(values, -0.5 * ((SQRT_PI * (residual + integer)) / sigma)^2)
    end
    return log(SQRT_PI) - log(sigma) - 0.5log(2pi) + logsumexp(values)
end

function build_likelihood_tables(grid::Vector{Float64}, base_sigma::Float64, bins::Int, terms::Int)
    centers = collect(range(-0.5 + 0.5 / bins, 0.5 - 0.5 / bins; length=bins))
    plus = zeros(Float64, length(grid), bins)
    minus = zeros(Float64, length(grid), bins)
    for (state, theta) in enumerate(grid)
        sigma_plus = sigma_vector(base_sigma, theta, [1, -1])[1]
        sigma_minus = sigma_vector(base_sigma, theta, [1, -1])[2]
        for bin in 1:bins
            plus[state, bin] = periodic_residual_logdensity(centers[bin], sigma_plus, terms)
            minus[state, bin] = periodic_residual_logdensity(centers[bin], sigma_minus, terms)
        end
    end
    return plus, minus
end

mutable struct CausalPosteriorFilter
    grid::Vector{Float64}
    transition::Matrix{Float64}
    loglik_plus::Matrix{Float64}
    loglik_minus::Matrix{Float64}
    posterior::Vector{Float64}
    block_loglik::Vector{Float64}
    block_count::Int
    cadence::Int
    bins::Int
end

function build_filter(config, base_sigma::Float64)
    cfg = config["posterior_filter"]
    grid = Float64.(cfg["theta_grid"])
    transition = build_transition(grid, cfg)
    bins = Int(cfg["residual_bins"])
    plus, minus = build_likelihood_tables(grid, base_sigma, bins, Int(cfg["periodic_gaussian_terms"]))
    nominal = Float64(config["drift"]["nominal_theta"])
    posterior = exp.(-0.5 .* ((grid .- nominal) ./ 0.05).^2)
    posterior ./= sum(posterior)
    return CausalPosteriorFilter(
        grid, transition, plus, minus, posterior, zeros(length(grid)), 0,
        Int(cfg["update_cadence_cycles"]), bins,
    )
end

function predictive_precision(filter::CausalPosteriorFilter, signs::Vector{Int})
    plus = sum(filter.posterior .* (cosh.(filter.grid) .* exp.(-filter.grid)))
    minus = sum(filter.posterior .* (cosh.(filter.grid) .* exp.(filter.grid)))
    return [sign == 1 ? plus : minus for sign in signs]
end

posterior_theta(filter::CausalPosteriorFilter) = sum(filter.posterior .* filter.grid)

function residual_bin(residual::Float64, bins::Int)
    return clamp(floor(Int, (residual + 0.5) * bins) + 1, 1, bins)
end

function observe!(filter::CausalPosteriorFilter, residuals::Vector{Float64}, signs::Vector{Int})
    for (residual, sign) in zip(residuals, signs)
        bin = residual_bin(residual, filter.bins)
        if sign == 1
            filter.block_loglik .+= @view filter.loglik_plus[:, bin]
        else
            filter.block_loglik .+= @view filter.loglik_minus[:, bin]
        end
    end
    filter.block_count += 1
    if filter.block_count == filter.cadence
        logposterior = log.(filter.posterior .+ eps()) .+ filter.block_loglik
        logposterior .-= maximum(logposterior)
        filtered = exp.(logposterior)
        filtered ./= sum(filtered)
        filter.posterior .= transpose(filter.transition) * filtered
        filter.posterior ./= sum(filter.posterior)
        fill!(filter.block_loglik, 0.0)
        filter.block_count = 0
        return true
    end
    return false
end

struct PreparedSubspace
    closest::Vector{Int}
    second::Vector{Int}
    delta::Vector{Float64}
    residuals::Vector{Float64}
    syndrome::Vector{Int}
end

function prepare_subspace(x::Vector{Float64}, stabilizers::Dict{Int64,Vector{Int64}})
    scaled = x ./ SQRT_PI
    closest = LatticeAlgorithms.closest_integer.(scaled)
    second = LatticeAlgorithms.second_closest_integer.(scaled)
    delta = (second .- scaled).^2 .- (closest .- scaled).^2
    residuals = scaled .- closest
    boundary = length(stabilizers) + 1
    syndrome = zeros(Int, boundary)
    for (index, stabilizer) in stabilizers
        syndrome[index] = mod(sum(closest[stabilizer]), 2)
    end
    syndrome[boundary] = mod(sum(syndrome), 2)
    return PreparedSubspace(closest, second, delta, residuals, syndrome)
end

function decode_prepared(prepared::PreparedSubspace, stabilizers::Dict{Int64,Vector{Int64}}, precision::Vector{Float64})
    boundary = length(stabilizers) + 1
    keys_stabilizers = collect(keys(stabilizers))
    values_stabilizers = collect(values(stabilizers))
    edge_to_mode = Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    weights = prepared.delta .* precision
    for qubit in eachindex(prepared.closest)
        membership = findall(qubit .∈ values_stabilizers)
        length(membership) > 2 && error("unsupported non-matching topology")
        isempty(membership) && continue
        vertices = length(membership) == 2 ?
            (keys_stabilizers[membership[1]], keys_stabilizers[membership[2]]) :
            (keys_stabilizers[membership[1]], boundary)
        key = minmax(vertices[1], vertices[2])
        if !haskey(edge_to_mode, key) || weights[qubit] < edge_to_mode[key][2]
            edge_to_mode[key] = (qubit, weights[qubit])
        end
    end
    graph = zeros(Float64, boundary, boundary)
    for ((i, j), (_, weight)) in edge_to_mode
        graph[i, j] = weight
        graph[j, i] = weight
    end
    paths = official_mwpm(graph, prepared.syndrome)
    correction = zeros(Int, length(prepared.closest))
    for (i, j) in paths
        key = minmax(Int(i), Int(j))
        correction[edge_to_mode[key][1]] += 1
    end
    return correction - mod.(prepared.closest, 2)
end

function logical_error(qfinal, pfinal, d::Int)
    xbit = mod(sum(qfinal[LatticeAlgorithms.surface_code_Z_logicals(d)[1]]), 2)
    zbit = mod(sum(pfinal[LatticeAlgorithms.surface_code_X_logicals(d)[1]]), 2)
    return (xbit != 0) || (zbit != 0)
end

function theta_schedule(family::String, cycle::Int, seed::Int, config, telegraph_state)
    drift = config["drift"]
    if family == "smooth"
        cfg = drift["smooth"]
        theta = Float64(cfg["center"]) + Float64(cfg["amplitude"]) * sin(2pi * (cycle - 1) / Int(cfg["period_cycles"]))
        return theta, telegraph_state
    elseif family == "calibration_shift"
        cfg = drift["calibration_shift"]
        onset = Int(cfg["onset_min"]) + mod((seed - 61830001) * Int(cfg["onset_seed_multiplier"]), Int(cfg["onset_span"]))
        theta = cycle < onset ? Float64(cfg["before"]) : Float64(cfg["after"])
        return theta, merge(telegraph_state, Dict("registered_onset" => onset))
    elseif family == "telegraph"
        if cycle == telegraph_state["next_transition"]
            telegraph_state["theta"] = telegraph_state["theta"] > 0 ? Float64(drift["telegraph"]["low"]) : Float64(drift["telegraph"]["high"])
            push!(telegraph_state["transitions"], cycle)
            raw_dwell = round(Int, randexp(telegraph_state["rng"]) * Int(drift["telegraph"]["mean_dwell_cycles"]))
            dwell = clamp(raw_dwell, Int(drift["telegraph"]["minimum_dwell_cycles"]), Int(drift["telegraph"]["maximum_dwell_cycles"]))
            telegraph_state["next_transition"] = cycle + dwell
        end
        return Float64(telegraph_state["theta"]), telegraph_state
    end
    error("unknown family $family")
end

function initial_telegraph_state(seed::Int, config)
    rng = MersenneTwister(seed * 1009 + 83)
    cfg = config["drift"]["telegraph"]
    raw_dwell = round(Int, randexp(rng) * Int(cfg["mean_dwell_cycles"]))
    dwell = clamp(raw_dwell, Int(cfg["minimum_dwell_cycles"]), Int(cfg["maximum_dwell_cycles"]))
    return Dict{String,Any}(
        "rng" => rng,
        "theta" => Float64(cfg["high"]),
        "next_transition" => dwell + 1,
        "transitions" => Int[],
    )
end

function transition_lags(true_trace, estimate_trace, update_cycles, transition_cycles, stable_updates::Int)
    lags = Union{Nothing,Int}[]
    for transition in transition_cycles
        before_index = findlast(cycle -> cycle < transition, update_cycles)
        after_index = findfirst(cycle -> cycle >= transition, update_cycles)
        if before_index === nothing || after_index === nothing
            push!(lags, nothing)
            continue
        end
        before = true_trace[before_index]
        target = true_trace[after_index]
        midpoint = 0.5 * (before + target)
        direction = target > before ? 1 : -1
        found = nothing
        for index in after_index:max(after_index, length(update_cycles) - stable_updates + 1)
            last_index = index + stable_updates - 1
            last_index > length(update_cycles) && break
            crossed = all(direction * (estimate_trace[j] - midpoint) >= 0 for j in index:last_index)
            if crossed
                found = update_cycles[index] - transition
                break
            end
        end
        push!(lags, found)
    end
    return lags
end

function smooth_lag(true_trace, estimate_trace, cadence::Int)
    isempty(true_trace) && return nothing
    maximum_lag_updates = min(64, length(true_trace) ÷ 4)
    best_lag = 0
    best_mse = Inf
    for lag in 0:maximum_lag_updates
        count = length(true_trace) - lag
        count < 8 && continue
        mse = mean((estimate_trace[(lag + 1):end] .- true_trace[1:count]).^2)
        if mse < best_mse
            best_mse = mse
            best_lag = lag
        end
    end
    return best_lag * cadence
end

function simulate_cluster(seed::Int, family::String, cycles::Int, config)
    d = Int(config["structured_family"]["distance"])
    base_sigma = Float64(config["structured_family"]["base_sigma"])
    window_cycles = Int(config["formal"]["window_cycles"])
    nominal_theta = Float64(config["drift"]["nominal_theta"])
    signs = coordinate_signs(d)
    qsigns, psigns = signs[1:2:end], signs[2:2:end]
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    filter = build_filter(config, base_sigma)
    static_precision = metric_precision(nominal_theta, signs)
    methods = String.(config["formal"]["comparators"])
    errors = Dict(method => 0 for method in methods)
    runtimes = Dict(method => 0.0 for method in methods)
    allocated = Dict(method => 0 for method in methods)
    current_window = Dict(method => 0 for method in methods)
    window_errors = Dict(method => Int[] for method in methods)
    rng = MersenneTwister(seed * 1009 + sum(codeunits(family)))
    telegraph_state = initial_telegraph_state(seed, config)
    update_cycles = Int[]
    true_trace = Float64[]
    estimate_trace = Float64[]
    posterior_entropy_trace = Float64[]
    started = time()

    for cycle in 1:cycles
        theta, telegraph_state = theta_schedule(family, cycle, seed, config, telegraph_state)
        sigmas = sigma_vector(base_sigma, theta, signs)
        x = sigmas .* randn(rng, length(signs))
        prepared_q = prepare_subspace(x[1:2:end], zstabs)
        prepared_p = prepare_subspace(x[2:2:end], xstabs)
        adaptive_precision = predictive_precision(filter, signs)
        oracle_precision = metric_precision(theta, signs)
        precision_by_method = Dict(
            "static_euclidean" => ones(length(signs)),
            "weighted_static" => static_precision,
            "observed_only_posterior_predictive_weighted" => adaptive_precision,
            "oracle_metric_upper_bound" => oracle_precision,
        )
        for method in methods
            precision = precision_by_method[method]
            local qfinal, pfinal
            if cycle == 1
                allocated[method] = @allocated begin
                    qfinal = decode_prepared(prepared_q, zstabs, precision[1:2:end])
                    pfinal = decode_prepared(prepared_p, xstabs, precision[2:2:end])
                end
            end
            runtimes[method] += @elapsed begin
                qfinal = decode_prepared(prepared_q, zstabs, precision[1:2:end])
                pfinal = decode_prepared(prepared_p, xstabs, precision[2:2:end])
            end
            error = logical_error(qfinal, pfinal, d)
            errors[method] += error
            current_window[method] += error
        end
        residuals = zeros(Float64, length(signs))
        residuals[1:2:end] .= prepared_q.residuals
        residuals[2:2:end] .= prepared_p.residuals
        updated = observe!(filter, residuals, signs)
        if updated
            push!(update_cycles, cycle + 1)
            next_cycle = min(cycle + 1, cycles)
            next_theta, _ = theta_schedule(family, next_cycle, seed, config, telegraph_state)
            push!(true_trace, next_theta)
            push!(estimate_trace, posterior_theta(filter))
            entropy = -sum(filter.posterior .* log.(filter.posterior .+ eps()))
            push!(posterior_entropy_trace, entropy)
        end
        if mod(cycle, window_cycles) == 0
            for method in methods
                push!(window_errors[method], current_window[method])
                current_window[method] = 0
            end
        end
    end

    registered_transitions = family == "calibration_shift" ?
        [Int(telegraph_state["registered_onset"])] :
        (family == "telegraph" ? Int.(telegraph_state["transitions"]) : Int[])
    stable_updates = Int(config["statistics"]["lag_midpoint_stable_updates"])
    lags = family == "smooth" ? Union{Nothing,Int}[smooth_lag(true_trace, estimate_trace, filter.cadence)] :
        transition_lags(true_trace, estimate_trace, update_cycles, registered_transitions, stable_updates)
    return Dict(
        "seed" => seed,
        "family" => family,
        "cycles" => cycles,
        "full_windows" => div(cycles, window_cycles),
        "unwindowed_tail_cycles" => mod(cycles, window_cycles),
        "errors" => errors,
        "window_errors" => window_errors,
        "runtime_seconds" => runtimes,
        "allocated_bytes_first_measured_decode" => allocated,
        "cluster_wall_clock_seconds" => time() - started,
        "posterior_update_cadence_cycles" => filter.cadence,
        "posterior_updates" => length(update_cycles),
        "posterior_theta_mean" => mean(estimate_trace),
        "posterior_theta_rmse" => sqrt(mean((estimate_trace .- true_trace).^2)),
        "posterior_entropy_mean" => mean(posterior_entropy_trace),
        "registered_transition_cycles" => registered_transitions,
        "adaptation_lag_cycles" => lags,
        "censored_lag_count" => count(isnothing, lags),
        "trace_sample" => [Dict(
            "cycle" => update_cycles[index],
            "true_theta" => true_trace[index],
            "posterior_theta" => estimate_trace[index],
            "posterior_entropy" => posterior_entropy_trace[index],
        ) for index in 1:32:length(update_cycles)],
    )
end

function correctness_checks(config)
    d = Int(config["structured_family"]["distance"])
    base_sigma = Float64(config["structured_family"]["base_sigma"])
    signs = coordinate_signs(d)
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    rng = MersenneTwister(61839999)
    exact_mismatches = 0
    scale_mismatches = 0
    samples = 512
    for _ in 1:samples
        x = base_sigma .* randn(rng, 2d^2)
        prepared_q = prepare_subspace(x[1:2:end], zstabs)
        prepared_p = prepare_subspace(x[2:2:end], xstabs)
        qfinal = decode_prepared(prepared_q, zstabs, ones(d^2))
        pfinal = decode_prepared(prepared_p, xstabs, ones(d^2))
        _, official_final = Base.invokelatest(LatticeAlgorithms.decode_surface_code, x, xstabs, zstabs)
        exact_mismatches += (qfinal != Int.(official_final[1:2:end])) || (pfinal != Int.(official_final[2:2:end]))
        scale = 0.2 + 3rand(rng)
        qscaled = decode_prepared(prepared_q, zstabs, fill(scale, d^2))
        pscaled = decode_prepared(prepared_p, xstabs, fill(scale, d^2))
        scale_mismatches += (qscaled != qfinal) || (pscaled != pfinal)
    end
    variance_errors = Float64[]
    for theta in Float64.(config["posterior_filter"]["theta_grid"])
        sigmas = sigma_vector(base_sigma, theta, signs)
        push!(variance_errors, abs(mean(sigmas.^2) - base_sigma^2))
    end
    # Future mutation audit: predictions emitted before the mutated suffix must
    # remain bit-identical.  A later divergence is also required so the check
    # cannot pass because the filter silently ignores observations.
    filter_a = build_filter(config, base_sigma)
    filter_b = build_filter(config, base_sigma)
    causal_rng = MersenneTwister(61839998)
    prefix_cycles = 64
    prefix_mismatches = 0
    suffix_divergence_detected = false
    normalization_max_error = 0.0
    minimum_precision = Inf
    for cycle in 1:128
        precision_a = predictive_precision(filter_a, signs)
        precision_b = predictive_precision(filter_b, signs)
        minimum_precision = min(minimum_precision, minimum(precision_a), minimum(precision_b))
        if cycle <= prefix_cycles
            prefix_mismatches += precision_a != precision_b
        elseif maximum(abs.(precision_a .- precision_b)) > 1e-9
            suffix_divergence_detected = true
        end
        residual_a = rand(causal_rng, length(signs)) .- 0.5
        residual_b = cycle <= prefix_cycles ? copy(residual_a) : fill(0.49, length(signs))
        observe!(filter_a, residual_a, signs)
        observe!(filter_b, residual_b, signs)
        normalization_max_error = max(
            normalization_max_error,
            abs(sum(filter_a.posterior) - 1.0),
            abs(sum(filter_b.posterior) - 1.0),
        )
    end
    return Dict(
        "official_euclidean_samples" => samples,
        "official_euclidean_final_list_mismatches" => exact_mismatches,
        "positive_scale_invariance_samples" => samples,
        "positive_scale_invariance_mismatches" => scale_mismatches,
        "max_mean_variance_error" => maximum(variance_errors),
        "strict_causal_prefix_cycles" => prefix_cycles,
        "strict_causal_prefix_mismatches" => prefix_mismatches,
        "mutated_suffix_divergence_detected" => suffix_divergence_detected,
        "posterior_normalization_max_error" => normalization_max_error,
        "minimum_predictive_precision" => minimum_precision,
        "passed" => exact_mismatches == 0 && scale_mismatches == 0 &&
            maximum(variance_errors) < 1e-12 && prefix_mismatches == 0 &&
            suffix_divergence_detected && normalization_max_error < 1e-12 && minimum_precision > 0.0,
    )
end

function main()
    opts = parse_args(ARGS)
    config_path = normpath(opts["config"])
    config = JSON.parsefile(config_path)
    mode = opts["mode"]
    mode in ["pilot", "formal", "correctness"] || error("unknown mode $mode")
    formal_seeds = Int.(config["formal"]["seeds"])
    seeds = mode == "pilot" ? Int.(config["pilot"]["seeds"]) : formal_seeds
    start_index = parse(Int, opts["seed-index-start"])
    end_index = min(parse(Int, opts["seed-index-end"]), length(seeds))
    selected_seeds = mode == "correctness" ? Int[] : seeds[start_index:end_index]
    default_cycles = mode == "pilot" ? Int(config["pilot"]["cycles_per_cluster"]) : Int(config["formal"]["cycles_per_cluster"])
    cycles = parse(Int, opts["cycles"]) > 0 ? parse(Int, opts["cycles"]) : default_cycles
    official_root = joinpath(REPO_ROOT, "third_party", "LatticeAlgorithms.jl")
    result = Dict{String,Any}(
        "schema_version" => "t6.18.3-julia-shard-v1",
        "task_id" => "T6.18.3",
        "mode" => mode,
        "generated_at" => string(now()),
        "config_path" => relpath(config_path, REPO_ROOT),
        "config_sha256" => sha256_file(config_path),
        "official_head" => git_head(official_root),
        "expected_official_head" => config["official_commit"],
        "head_matches" => git_head(official_root) == config["official_commit"],
        "julia_version" => string(VERSION),
        "selected_seeds" => selected_seeds,
        "cycles_per_cluster" => cycles,
        "correctness" => correctness_checks(config),
    )
    rows = Vector{Dict{String,Any}}()
    started = time()
    if mode != "correctness"
        for seed in selected_seeds, family in String.(config["formal"]["families"])
            push!(rows, simulate_cluster(seed, family, cycles, config))
            println(JSON.json(Dict("seed" => seed, "family" => family, "completed" => length(rows), "wall_clock_seconds" => time() - started)))
            flush(stdout)
        end
    end
    result["rows"] = rows
    result["wall_clock_seconds"] = time() - started
    output = normpath(opts["output"])
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, result, 2)
        write(io, "\n")
    end
    println(JSON.json(Dict("task_id" => "T6.18.3", "mode" => mode, "output" => output, "rows" => length(rows), "wall_clock_seconds" => result["wall_clock_seconds"])))
end

main()

using Dates
using JSON
using LinearAlgebra
using Random
using SHA
using Statistics

using LatticeAlgorithms

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SQRT_PI = sqrt(pi)
const DEFAULT_CONFIG = joinpath(REPO_ROOT, "configs", "phase6d", "t6_20_4_causal_headroom.json")

sha256_file(path) = bytes2hex(open(sha256, path))
git_head(path) = strip(read(`git -C $path rev-parse HEAD`, String))

function parse_args(args)
    opts = Dict(
        "config" => DEFAULT_CONFIG,
        "output" => joinpath(REPO_ROOT, "runs", "t6_20_4_causal_headroom_raw.json"),
        "seed-limit" => "0",
        "family-limit" => "0",
        "rounds" => "0",
        "correctness-only" => "false",
    )
    index = 1
    while index <= length(args)
        startswith(args[index], "--") || error("unexpected positional argument $(args[index])")
        key = args[index][3:end]
        index == length(args) && error("missing value for --$(key)")
        opts[key] = args[index + 1]
        index += 2
    end
    return opts
end

function logsumexp(values)
    m = maximum(values)
    return isfinite(m) ? m + log(sum(exp(value - m) for value in values)) : m
end

function bitvector(value::Int, n::Int)
    return [Int((value >> (index - 1)) & 1) for index in 1:n]
end

function vector_key(bits)
    value = 0
    for (index, bit) in enumerate(bits)
        value |= (Int(bit) & 1) << (index - 1)
    end
    return value
end

function generator_vectors(stabilizers, n::Int)
    rows = Vector{Vector{Int}}()
    for key in sort(collect(keys(stabilizers)))
        row = zeros(Int, n)
        row[stabilizers[key]] .= 1
        push!(rows, row)
    end
    return rows
end

function span_vectors(generators, n::Int)
    rows = Vector{Vector{Int}}()
    for mask in 0:(2^length(generators) - 1)
        row = zeros(Int, n)
        for index in eachindex(generators)
            if ((mask >> (index - 1)) & 1) == 1
                row .= xor.(row, generators[index])
            end
        end
        push!(rows, row)
    end
    unique_keys = unique(vector_key(row) for row in rows)
    length(unique_keys) == length(rows) || error("dependent stabilizer generators")
    return rows
end

struct CosetTable
    checks::Vector{Vector{Int}}
    stabilizer_span::Vector{Vector{Int}}
    logical_generator::Vector{Int}
    logical_detector::Vector{Int}
    coset0::Dict{Int,Matrix{Int}}
    coset1::Dict{Int,Matrix{Int}}
    base::Dict{Int,Vector{Int}}
end

function syndrome_key(bits, checks)
    syndrome = [mod(sum(bits[index] for index in check), 2) for check in checks]
    return vector_key(syndrome)
end

function build_coset_table(check_dict, stabilizer_dict, logical_generator_indices, logical_detector_indices, n::Int)
    checks = [check_dict[key] for key in sort(collect(keys(check_dict)))]
    generators = generator_vectors(stabilizer_dict, n)
    stabilizer_span = span_vectors(generators, n)
    logical_generator = zeros(Int, n)
    logical_generator[logical_generator_indices] .= 1
    logical_detector = zeros(Int, n)
    logical_detector[logical_detector_indices] .= 1
    mod(sum(logical_generator .* logical_detector), 2) == 1 || error("logical intersection must be odd")
    base = Dict{Int,Vector{Int}}()
    for value in 0:(2^n - 1)
        bits = bitvector(value, n)
        key = syndrome_key(bits, checks)
        if !haskey(base, key)
            base[key] = bits
        end
    end
    length(base) == 2^length(checks) || error("not all syndromes represented")
    coset0 = Dict{Int,Matrix{Int}}()
    coset1 = Dict{Int,Matrix{Int}}()
    for (key, representative) in base
        rows0 = [xor.(representative, stabilizer) for stabilizer in stabilizer_span]
        rows1 = [xor.(row, logical_generator) for row in rows0]
        coset0[key] = reduce(vcat, [transpose(row) for row in rows0])
        coset1[key] = reduce(vcat, [transpose(row) for row in rows1])
    end
    return CosetTable(checks, stabilizer_span, logical_generator, logical_detector, coset0, coset1, base)
end

function build_tables(d::Int)
    n = d^2
    xstabs = LatticeAlgorithms.surface_code_X_stabilizers(d)
    zstabs = LatticeAlgorithms.surface_code_Z_stabilizers(d)
    xlogical = LatticeAlgorithms.surface_code_X_logicals(d)[1]
    zlogical = LatticeAlgorithms.surface_code_Z_logicals(d)[1]
    qtable = build_coset_table(zstabs, xstabs, xlogical, zlogical, n)
    ptable = build_coset_table(xstabs, zstabs, zlogical, xlogical, n)
    return qtable, ptable, xstabs, zstabs, xlogical, zlogical
end

function parity_loglikelihoods(residuals, sigmas; terms::Int, means=zeros(length(residuals)))
    n = length(residuals)
    logp0 = zeros(Float64, n)
    logp1 = zeros(Float64, n)
    for index in 1:n
        centered = residuals[index] - means[index]
        values0 = [-0.5 * (SQRT_PI * (centered + 2integer) / sigmas[index])^2 for integer in -terms:terms]
        values1 = [-0.5 * (SQRT_PI * (centered + 2integer + 1) / sigmas[index])^2 for integer in -terms:terms]
        logp0[index] = logsumexp(values0) - log(sigmas[index])
        logp1[index] = logsumexp(values1) - log(sigmas[index])
    end
    return logp0, logp1
end

mutable struct CircularMomentEstimator
    moment::Vector{ComplexF64}
    accumulated::Vector{ComplexF64}
    accumulated_rounds::Int
    cadence::Int
    alpha::Float64
    sigma_floor::Float64
    sigma_ceiling::Float64
end

function build_circular_estimator(n::Int, base_sigma::Float64, cfg; cadence::Int, alpha::Float64)
    # For r = x/sqrt(pi) mod 1 and x~N(mu,sigma^2),
    # E exp(i 2pi r) = exp(i 2pi mu/sqrt(pi)) exp(-2pi sigma^2).
    initial = ComplexF64(exp(-2pi * base_sigma^2))
    return CircularMomentEstimator(
        fill(initial, n), zeros(ComplexF64, n), 0, cadence, alpha,
        Float64(cfg["sigma_floor"]), Float64(cfg["sigma_ceiling"]),
    )
end


function circular_parameters(estimator::CircularMomentEstimator)
    magnitudes = clamp.(abs.(estimator.moment), exp(-2pi * estimator.sigma_ceiling^2), exp(-2pi * estimator.sigma_floor^2))
    sigmas = sqrt.(-log.(magnitudes) ./ (2pi))
    means = angle.(estimator.moment) ./ (2pi)
    return sigmas, means
end


function observe_circular!(estimator::CircularMomentEstimator, residuals)
    estimator.accumulated .+= exp.(2pi * im .* residuals)
    estimator.accumulated_rounds += 1
    if estimator.accumulated_rounds == estimator.cadence
        block = estimator.accumulated ./ estimator.accumulated_rounds
        estimator.moment .= (1 - estimator.alpha) .* estimator.moment .+ estimator.alpha .* block
        fill!(estimator.accumulated, 0.0 + 0.0im)
        estimator.accumulated_rounds = 0
        return true
    end
    return false
end

function coset_scores(table::CosetTable, syndrome::Int, logp0, logp1)
    rows0 = table.coset0[syndrome]
    rows1 = table.coset1[syndrome]
    function score(rows)
        values = zeros(Float64, size(rows, 1))
        for row in axes(rows, 1)
            values[row] = sum(rows[row, column] == 0 ? logp0[column] : logp1[column] for column in axes(rows, 2))
        end
        return logsumexp(values)
    end
    return score(rows0), score(rows1)
end

function class_for_error(table::CosetTable, syndrome::Int, error_bits)
    delta = xor.(error_bits, table.base[syndrome])
    return mod(sum(delta .* table.logical_detector), 2)
end

function build_transition(grid, cfg)
    n = length(grid)
    transition = zeros(Float64, n, n)
    for index in 1:n
        local_kernel = exp.(-0.5 .* ((grid .- grid[index]) ./ Float64(cfg["transition_local_sigma"])).^2)
        local_kernel ./= sum(local_kernel)
        transition[index, :] .= Float64(cfg["transition_local_mass"]) .* local_kernel .+
            Float64(cfg["transition_uniform_mass"]) / n
        transition[index, index] += Float64(cfg["transition_identity_mass"])
        transition[index, :] ./= sum(transition[index, :])
    end
    return transition
end

mutable struct CausalFilter
    grid::Vector{Float64}
    transition::Matrix{Float64}
    posterior::Vector{Float64}
    accumulated_loglik::Vector{Float64}
    accumulated_rounds::Int
    cadence::Int
end

function build_filter(cfg; cadence::Int)
    grid = Float64.(cfg["theta_grid"])
    initial = exp.(-0.5 .* (grid ./ Float64(cfg["initial_std"])).^2)
    initial ./= sum(initial)
    return CausalFilter(grid, build_transition(grid, cfg), initial, zeros(length(grid)), 0, cadence)
end

function normalize_logweights(logweights)
    shifted = logweights .- maximum(logweights)
    weights = exp.(shifted)
    weights ./= sum(weights)
    return weights
end

function update_filter!(filter::CausalFilter, observation_loglik)
    filter.accumulated_loglik .+= observation_loglik
    filter.accumulated_rounds += 1
    if filter.accumulated_rounds == filter.cadence
        filtered = normalize_logweights(log.(filter.posterior .+ eps()) .+ filter.accumulated_loglik)
        filter.posterior .= transpose(filter.transition) * filtered
        filter.posterior ./= sum(filter.posterior)
        fill!(filter.accumulated_loglik, 0.0)
        filter.accumulated_rounds = 0
        return true
    end
    return false
end

function spatial_loadings(signs, permutation, law::String)
    n = length(signs)
    ranks = zeros(Float64, n)
    for (rank, coordinate) in enumerate(permutation)
        ranks[Int(coordinate) + 1] = 2 * (rank - 1) / (n - 1) - 1
    end
    if law == "train_log_spline"
        raw = 0.72 .* Float64.(signs) .+ 0.28 .* sin.(pi .* ranks)
    elseif law == "train_affine_modewise"
        raw = 0.60 .* Float64.(signs) .+ 0.40 .* ranks
    else
        error("unsupported train variance law $law")
    end
    raw .-= mean(raw)
    raw ./= sqrt(mean(raw.^2))
    return raw
end

function sigma_vector(base_sigma::Float64, theta::Float64, loadings)
    variance_multiplier = exp.(theta .* loadings)
    variance_multiplier ./= mean(variance_multiplier)
    return base_sigma .* sqrt.(variance_multiplier)
end

scaled_sigma_vector(nominal_sigmas, theta) = nominal_sigmas .* exp(0.5theta)

function observation_loglikelihood(residuals, nominal_sigmas, means, grid; terms::Int)
    values = zeros(Float64, length(grid))
    for (state, theta) in enumerate(grid)
        sigmas = scaled_sigma_vector(nominal_sigmas, theta)
        total = 0.0
        for index in eachindex(residuals)
            centered = residuals[index] - means[index]
            aliases = [-0.5 * (SQRT_PI * (centered + integer) / sigmas[index])^2 for integer in -terms:terms]
            total += logsumexp(aliases) - log(sigmas[index])
        end
        values[state] = total
    end
    return values
end

function predictive_precision(filter::CausalFilter, nominal_sigmas)
    precision = zeros(length(nominal_sigmas))
    for (weight, theta) in zip(filter.posterior, filter.grid)
        precision .+= weight ./ scaled_sigma_vector(nominal_sigmas, theta).^2
    end
    return precision
end

struct PreparedSubspace
    closest::Vector{Int}
    residuals::Vector{Float64}
    syndrome::Vector{Int}
end

function prepare_subspace(x, stabilizers)
    scaled = x ./ SQRT_PI
    closest = Int.(LatticeAlgorithms.closest_integer.(scaled))
    residuals = scaled .- closest
    syndrome = [mod(sum(closest[index] for index in stabilizers[key]), 2) for key in sort(collect(keys(stabilizers)))]
    return PreparedSubspace(closest, residuals, syndrome)
end

official_mwpm(graph, syndrome) = Base.invokelatest(LatticeAlgorithms.mwpm, graph, syndrome)

function minimum_tjoin(graph, syndrome)
    edges = Tuple{Int,Int,Float64}[]
    for left in 1:(size(graph, 1) - 1), right in (left + 1):size(graph, 2)
        graph[left, right] > 0 && push!(edges, (left, right, graph[left, right]))
    end
    length(edges) <= 24 || error("pure-Julia exhaustive T-join is restricted to the d=3 diagnostic")
    best_cost = Inf
    best_edges = Tuple{Int,Int}[]
    for mask in 0:(2^length(edges) - 1)
        parity = zeros(Int, length(syndrome))
        cost = 0.0
        selected = Tuple{Int,Int}[]
        for index in eachindex(edges)
            if ((mask >> (index - 1)) & 1) == 1
                left, right, weight = edges[index]
                parity[left] ⊻= 1
                parity[right] ⊻= 1
                cost += weight
                push!(selected, (left, right))
            end
        end
        if parity == syndrome && cost < best_cost
            best_cost = cost
            best_edges = selected
        end
    end
    isfinite(best_cost) || error("no T-join satisfies syndrome")
    return best_edges
end

function decode_cpd(prepared::PreparedSubspace, stabilizers, weights; backend::Symbol=:pure_julia)
    boundary = length(stabilizers) + 1
    keys_stabilizers = sort(collect(keys(stabilizers)))
    values_stabilizers = [stabilizers[key] for key in keys_stabilizers]
    edge_to_mode = Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    for qubit in eachindex(prepared.closest)
        membership = findall(qubit .∈ values_stabilizers)
        isempty(membership) && continue
        length(membership) <= 2 || error("unsupported topology")
        vertices = length(membership) == 2 ?
            (keys_stabilizers[membership[1]], keys_stabilizers[membership[2]]) :
            (keys_stabilizers[membership[1]], boundary)
        key = minmax(vertices...)
        if !haskey(edge_to_mode, key) || weights[qubit] < edge_to_mode[key][2]
            edge_to_mode[key] = (qubit, weights[qubit])
        end
    end
    graph = zeros(Float64, boundary, boundary)
    for ((left, right), (_, weight)) in edge_to_mode
        graph[left, right] = max(Float64(weight), 1e-12)
        graph[right, left] = max(Float64(weight), 1e-12)
    end
    syndrome_with_boundary = vcat(prepared.syndrome, mod(sum(prepared.syndrome), 2))
    paths = backend == :official ? collect(official_mwpm(graph, syndrome_with_boundary)) :
        minimum_tjoin(graph, syndrome_with_boundary)
    correction = zeros(Int, length(prepared.closest))
    for (left, right) in paths
        correction[edge_to_mode[minmax(Int(left), Int(right))][1]] += 1
    end
    return mod.(correction, 2)
end

function gaussian_cpd_weights(prepared::PreparedSubspace, precision)
    # Squared-distance gap between the nearest and second-nearest integers.
    # `sign(0)==0` would incorrectly produce a zero edge at the exact cell
    # centre, so use the closed form instead of a sign-based construction.
    return (1 .- 2 .* abs.(prepared.residuals)) .* precision
end

function periodic_mwpm_weights(residuals, posterior, grid, nominal_sigmas, means; terms::Int)
    p0 = zeros(Float64, length(residuals))
    p1 = zeros(Float64, length(residuals))
    for (weight, theta) in zip(posterior, grid)
        log0, log1 = parity_loglikelihoods(
            residuals, scaled_sigma_vector(nominal_sigmas, theta); terms=terms, means=means,
        )
        p0 .+= weight .* exp.(log0)
        p1 .+= weight .* exp.(log1)
    end
    # The hard GKP decision is the nearest integer.  Its flip cost depends on
    # the folded residual only, hence the absolute parity labels are sorted.
    return abs.(log.(p0 .+ eps()) .- log.(p1 .+ eps()))
end

function exact_state_scores(table, syndrome, residuals, nominal_sigmas, means, grid; terms::Int)
    scores0 = zeros(Float64, length(grid))
    scores1 = zeros(Float64, length(grid))
    for (state, theta) in enumerate(grid)
        logp0, logp1 = parity_loglikelihoods(
            residuals, scaled_sigma_vector(nominal_sigmas, theta); terms=terms, means=means,
        )
        scores0[state], scores1[state] = coset_scores(table, syndrome, logp0, logp1)
    end
    return scores0, scores1
end

function mixture_action(scores0, scores1, posterior)
    logweights = log.(posterior .+ eps())
    total0 = logsumexp(logweights .+ scores0)
    total1 = logsumexp(logweights .+ scores1)
    action = total1 > total0 ? 1 : 0
    confidence = exp(max(total0, total1) - logsumexp([total0, total1]))
    return action, confidence
end

function plugin_action(table, syndrome, residuals, nominal_sigmas, means, filter; terms::Int)
    theta = sum(filter.posterior .* filter.grid)
    logp0, logp1 = parity_loglikelihoods(
        residuals, scaled_sigma_vector(nominal_sigmas, theta); terms=terms, means=means,
    )
    score0, score1 = coset_scores(table, syndrome, logp0, logp1)
    return score1 > score0 ? 1 : 0
end

function robust_action(scores0, scores1, posterior, trusted_scores0, trusted_scores1, trusted_posterior, risk_cfg)
    evidence = [logsumexp([scores0[index], scores1[index]]) for index in eachindex(posterior)]
    state_posterior = normalize_logweights(log.(posterior .+ eps()) .+ evidence)
    order = sortperm(state_posterior; rev=true)
    selected = Int[]
    mass = 0.0
    for index in order
        push!(selected, index)
        mass += state_posterior[index]
        mass >= Float64(risk_cfg["posterior_mass"]) && break
    end
    support0 = minimum(exp(scores0[index] - logsumexp([scores0[index], scores1[index]])) for index in selected)
    support1 = minimum(exp(scores1[index] - logsumexp([scores0[index], scores1[index]])) for index in selected)
    trusted_logweights = log.(trusted_posterior .+ eps())
    trusted_total0 = logsumexp(trusted_logweights .+ trusted_scores0)
    trusted_total1 = logsumexp(trusted_logweights .+ trusted_scores1)
    trusted_norm = logsumexp([trusted_total0, trusted_total1])
    trusted_support0 = exp(trusted_total0 - trusted_norm)
    trusted_support1 = exp(trusted_total1 - trusted_norm)
    support0 = min(support0, trusted_support0)
    support1 = min(support1, trusted_support1)
    trusted_action = trusted_total1 > trusted_total0 ? 1 : 0
    threshold = Float64(risk_cfg["robust_state_support_threshold"])
    if max(support0, support1) >= threshold
        return support1 > support0 ? 1 : 0
    end
    # Ambiguous model set: fail closed to the immutable trusted static bank.
    return trusted_action
end

function logical_failure(error_bits, correction_bits, detector)
    return mod(sum(xor.(error_bits, correction_bits) .* detector), 2) == 1
end

function ar1_cholesky(n::Int, rho::Float64)
    covariance = [rho^abs(i - j) for i in 1:n, j in 1:n]
    return cholesky(Symmetric(covariance + 1e-12I)).L
end

mutable struct ScenarioState
    theta::Float64
    telegraph_sign::Int
    next_burst::Int
end

function scenario_parameters!(state::ScenarioState, family::String, cycle::Int, rounds::Int, amplitude, rate, base_rho, rng)
    theta = 0.0
    mean_scale = 0.0
    rho = base_rho
    scale = 1.0
    heavy_tail = false
    mismatch = false
    if family == "mean_drift"
        mean_scale = amplitude * sin(2pi * (cycle - 1) / rounds)
    elseif family == "variance_drift"
        theta = -amplitude + 2amplitude * (cycle - 1) / max(rounds - 1, 1)
    elseif family == "correlation_drift"
        rho = base_rho * sin(2pi * (cycle - 1) / rounds)
    elseif family == "periodic_drift"
        theta = amplitude * sin(4pi * (cycle - 1) / rounds)
    elseif family == "ou_drift"
        state.theta = 0.985 * state.theta + amplitude * sqrt(1 - 0.985^2) * randn(rng)
        theta = clamp(state.theta, -1.25amplitude, 1.25amplitude)
    elseif family == "random_walk"
        state.theta = clamp(state.theta + 0.08amplitude * randn(rng), -amplitude, amplitude)
        theta = state.theta
    elseif family == "step_calibration_shift"
        theta = cycle <= div(rounds, 2) ? -amplitude : amplitude
    elseif family == "telegraph_drift"
        if rand(rng) < rate
            state.telegraph_sign *= -1
        end
        theta = state.telegraph_sign * amplitude
    elseif family == "burst_outlier"
        if cycle == state.next_burst
            state.next_burst += max(64, div(rounds, 3))
        end
        scale = cycle in (state.next_burst - 8):(state.next_burst - 1) ? 2.4 : 1.0
    elseif family == "heavy_tail"
        heavy_tail = true
    elseif family == "compound_ood"
        theta = cycle <= div(rounds, 2) ? -amplitude : amplitude
        rho = clamp(1.5base_rho, -0.45, 0.45)
        scale = abs(cycle - div(rounds, 2)) <= 5 ? 2.2 : 1.0
    elseif family == "likelihood_mismatch"
        mismatch = true
    elseif family != "stationary_control"
        error("unknown scenario family $family")
    end
    return theta, mean_scale, rho, scale, heavy_tail, mismatch
end

function draw_noise(rng, sigmas, loadings, rho, mean_scale, scale, heavy_tail, mismatch)
    correlated = ar1_cholesky(length(sigmas), rho) * randn(rng, length(sigmas))
    noise = sigmas .* correlated
    noise .+= 0.30 * mean_scale * mean(sigmas) .* loadings
    noise .*= scale
    if heavy_tail
        chi2 = sum(randn(rng)^2 for _ in 1:5)
        noise .*= sqrt(3 / chi2)
    end
    if mismatch && rand(rng) < 0.12
        coordinate = rand(rng, eachindex(noise))
        noise[coordinate] += (rand(rng, Bool) ? 1 : -1) * randexp(rng) * 1.5sigmas[coordinate]
    end
    return noise
end

function posterior_entropy(posterior)
    return -sum(posterior .* log.(posterior .+ eps()))
end

function simulate_cell(cell, pattern, rounds::Int, config, tables)
    qtable, ptable, xstabs, zstabs, _, _ = tables
    methods = String.(config["methods"])
    errors = Dict(method => 0 for method in methods)
    xerrors = Dict(method => 0 for method in methods)
    zerrors = Dict(method => 0 for method in methods)
    action_disagreements = Dict(method => 0 for method in methods)
    base_sigma = Float64(cell["base_sigma"])
    loadings = spatial_loadings(Int.(pattern["signs"]), Int.(pattern["permutation"]), String(cell["variance_law_id"]))
    posterior_cfg = config["posterior"]
    legacy = build_filter(posterior_cfg; cadence=Int(posterior_cfg["legacy_block_cadence"]))
    fast = build_filter(posterior_cfg; cadence=1)
    static_posterior = copy(fast.posterior)
    legacy_moments = build_circular_estimator(
        length(loadings), base_sigma, posterior_cfg;
        cadence=Int(posterior_cfg["legacy_block_cadence"]),
        alpha=Float64(posterior_cfg["circular_alpha_legacy_block"]),
    )
    fast_moments = build_circular_estimator(
        length(loadings), base_sigma, posterior_cfg;
        cadence=1, alpha=Float64(posterior_cfg["circular_alpha_fast"]),
    )
    static_sigmas = fill(base_sigma, length(loadings))
    static_means = zeros(length(loadings))
    terms = Int(posterior_cfg["alias_terms"])
    rng = MersenneTwister(Int(cell["seed"]) * 1009 + sum(codeunits(String(cell["scenario_family"]))))
    state = ScenarioState(0.0, isodd(Int(cell["seed"])) ? 1 : -1, max(32, div(rounds, 4)))
    trace_hasher = SHA.SHA256_CTX()
    posterior_prior_hash = SHA.SHA256_CTX()
    entropy_sum = 0.0
    robust_interventions = 0
    sigma_estimate_squared_error = 0.0

    for cycle in 1:rounds
        family = String(cell["scenario_family"])
        theta, mean_scale, rho, scale, heavy_tail, mismatch = scenario_parameters!(
            state, family, cycle, rounds, Float64(cell["amplitude"]),
            Float64(cell["transition_rate_per_1000_rounds"]), Float64(cell["covariance_rho"]), rng,
        )
        sigmas = sigma_vector(base_sigma, theta, loadings)
        noise = draw_noise(rng, sigmas, loadings, rho, mean_scale, scale, heavy_tail, mismatch)
        prepared_q = prepare_subspace(noise[1:2:end], zstabs)
        prepared_p = prepare_subspace(noise[2:2:end], xstabs)
        qsyndrome = vector_key(prepared_q.syndrome)
        psyndrome = vector_key(prepared_p.syndrome)
        qerror = mod.(prepared_q.closest, 2)
        perror = mod.(prepared_p.closest, 2)
        qclass = class_for_error(qtable, qsyndrome, qerror)
        pclass = class_for_error(ptable, psyndrome, perror)

        SHA.update!(trace_hasher, reinterpret(UInt8, [Float64(value) for value in noise]))
        SHA.update!(posterior_prior_hash, reinterpret(UInt8, [Float64(value) for value in fast.posterior]))
        SHA.update!(posterior_prior_hash, reinterpret(UInt8, [Float64(real(value)) for value in fast_moments.moment]))
        SHA.update!(posterior_prior_hash, reinterpret(UInt8, [Float64(imag(value)) for value in fast_moments.moment]))

        legacy_sigmas, legacy_means = circular_parameters(legacy_moments)
        fast_sigmas, fast_means = circular_parameters(fast_moments)
        precision_legacy = predictive_precision(legacy, legacy_sigmas)
        precision_fast = predictive_precision(fast, fast_sigmas)
        correction_q_legacy = decode_cpd(prepared_q, zstabs, gaussian_cpd_weights(prepared_q, precision_legacy[1:2:end]))
        correction_p_legacy = decode_cpd(prepared_p, xstabs, gaussian_cpd_weights(prepared_p, precision_legacy[2:2:end]))
        correction_q_fast = decode_cpd(prepared_q, zstabs, gaussian_cpd_weights(prepared_q, precision_fast[1:2:end]))
        correction_p_fast = decode_cpd(prepared_p, xstabs, gaussian_cpd_weights(prepared_p, precision_fast[2:2:end]))

        q_periodic_weights = periodic_mwpm_weights(
            prepared_q.residuals, fast.posterior, fast.grid,
            fast_sigmas[1:2:end], fast_means[1:2:end]; terms=terms,
        )
        p_periodic_weights = periodic_mwpm_weights(
            prepared_p.residuals, fast.posterior, fast.grid,
            fast_sigmas[2:2:end], fast_means[2:2:end]; terms=terms,
        )
        correction_q_periodic = decode_cpd(prepared_q, zstabs, q_periodic_weights)
        correction_p_periodic = decode_cpd(prepared_p, xstabs, p_periodic_weights)

        q_scores0, q_scores1 = exact_state_scores(
            qtable, qsyndrome, prepared_q.residuals, fast_sigmas[1:2:end],
            fast_means[1:2:end], fast.grid; terms=terms,
        )
        p_scores0, p_scores1 = exact_state_scores(
            ptable, psyndrome, prepared_p.residuals, fast_sigmas[2:2:end],
            fast_means[2:2:end], fast.grid; terms=terms,
        )
        q_static_scores0, q_static_scores1 = exact_state_scores(
            qtable, qsyndrome, prepared_q.residuals, static_sigmas[1:2:end],
            static_means[1:2:end], fast.grid; terms=terms,
        )
        p_static_scores0, p_static_scores1 = exact_state_scores(
            ptable, psyndrome, prepared_p.residuals, static_sigmas[2:2:end],
            static_means[2:2:end], fast.grid; terms=terms,
        )
        q_static, _ = mixture_action(q_static_scores0, q_static_scores1, static_posterior)
        p_static, _ = mixture_action(p_static_scores0, p_static_scores1, static_posterior)
        q_plugin = plugin_action(
            qtable, qsyndrome, prepared_q.residuals, fast_sigmas[1:2:end], fast_means[1:2:end], fast; terms=terms,
        )
        p_plugin = plugin_action(
            ptable, psyndrome, prepared_p.residuals, fast_sigmas[2:2:end], fast_means[2:2:end], fast; terms=terms,
        )
        q_pp, q_confidence = mixture_action(q_scores0, q_scores1, fast.posterior)
        p_pp, p_confidence = mixture_action(p_scores0, p_scores1, fast.posterior)
        q_risk = robust_action(
            q_scores0, q_scores1, fast.posterior,
            q_static_scores0, q_static_scores1, static_posterior, config["risk_action"],
        )
        p_risk = robust_action(
            p_scores0, p_scores1, fast.posterior,
            p_static_scores0, p_static_scores1, static_posterior, config["risk_action"],
        )
        robust_interventions += (q_risk != q_pp) || (p_risk != p_pp)

        outcomes = Dict(
            "current_adaptive_weighted_cpd" => (
                logical_failure(qerror, correction_q_legacy, qtable.logical_detector),
                logical_failure(perror, correction_p_legacy, ptable.logical_detector),
            ),
            "static_mixture_exact_mld" => (qclass != q_static, pclass != p_static),
            "estimator_upgrade_weighted_cpd" => (
                logical_failure(qerror, correction_q_fast, qtable.logical_detector),
                logical_failure(perror, correction_p_fast, ptable.logical_detector),
            ),
            "periodic_likelihood_mwpm" => (
                logical_failure(qerror, correction_q_periodic, qtable.logical_detector),
                logical_failure(perror, correction_p_periodic, ptable.logical_detector),
            ),
            "plugin_exact_logical_coset_mld" => (qclass != q_plugin, pclass != p_plugin),
            "posterior_predictive_exact_mld" => (qclass != q_pp, pclass != p_pp),
            "risk_aware_observed_only_action" => (qclass != q_risk, pclass != p_risk),
        )
        reference = outcomes["current_adaptive_weighted_cpd"]
        for method in methods
            xerror, zerror = outcomes[method]
            xerrors[method] += xerror && !zerror
            zerrors[method] += zerror && !xerror
            errors[method] += xerror || zerror
            action_disagreements[method] += outcomes[method] != reference
        end

        residuals = zeros(Float64, length(loadings))
        residuals[1:2:end] .= prepared_q.residuals
        residuals[2:2:end] .= prepared_p.residuals
        legacy_observation = observation_loglikelihood(residuals, legacy_sigmas, legacy_means, legacy.grid; terms=terms)
        fast_observation = observation_loglikelihood(residuals, fast_sigmas, fast_means, fast.grid; terms=terms)
        update_filter!(legacy, legacy_observation)
        update_filter!(fast, fast_observation)
        observe_circular!(legacy_moments, residuals)
        observe_circular!(fast_moments, residuals)
        entropy_sum += posterior_entropy(fast.posterior)
        sigma_estimate_squared_error += mean((fast_sigmas .- sigmas).^2)
    end
    yerrors = Dict(method => errors[method] - xerrors[method] - zerrors[method] for method in methods)
    return Dict(
        "cell_id" => cell["cell_id"],
        "source_cell_sha256" => cell["cell_sha256"],
        "seed" => cell["seed"],
        "family" => cell["scenario_family"],
        "distance" => cell["distance"],
        "base_sigma" => cell["base_sigma"],
        "variance_law_id" => cell["variance_law_id"],
        "spatial_pattern_sha256" => cell["spatial_pattern_sha256"],
        "rounds" => rounds,
        "errors" => errors,
        "x_only_errors" => xerrors,
        "z_only_errors" => zerrors,
        "y_errors" => yerrors,
        "action_disagreements_vs_current" => action_disagreements,
        "robust_interventions" => robust_interventions,
        "posterior_entropy_mean" => entropy_sum / rounds,
        "modewise_sigma_rmse_scoring_only" => sqrt(sigma_estimate_squared_error / rounds),
        "physical_trace_sha256" => bytes2hex(SHA.digest!(trace_hasher)),
        "predecision_posterior_sha256" => bytes2hex(SHA.digest!(posterior_prior_hash)),
    )
end

function explicit_probability_checks(config, tables)
    qtable, ptable, xstabs, zstabs, _, _ = tables
    correctness = config["correctness"]
    samples = Int(correctness["official_bsv_samples"])
    terms = Int(correctness["independent_alias_terms"])
    sigma = 0.54
    rng = MersenneTwister(620049998)
    official_mismatches = 0
    maximum_log_odds_error = 0.0
    normalization_error = 0.0
    alias_mismatches = 0
    alias_samples = Int(correctness["alias_convergence_samples"])
    tjoin_samples = 128
    tjoin_correction_mismatches = 0

    for sample in 1:max(samples, alias_samples)
        table, whichtype = isodd(sample) ? (qtable, "x") : (ptable, "z")
        residuals = rand(rng, 9) .- 0.5
        syndrome = rand(rng, collect(keys(table.base)))
        logp0, logp1 = parity_loglikelihoods(residuals, fill(sigma, 9); terms=terms)
        score0, score1 = coset_scores(table, syndrome, logp0, logp1)
        probability0 = exp(score0 - logsumexp([score0, score1]))
        probability1 = exp(score1 - logsumexp([score0, score1]))
        normalization_error = max(normalization_error, abs(probability0 + probability1 - 1.0))
        if sample <= samples
            eta = SQRT_PI .* (residuals .+ table.base[syndrome])
            logical = SQRT_PI .* table.logical_generator
            official0 = Base.invokelatest(LatticeAlgorithms.get_prob, eta, whichtype, sigma; Nv=Int(correctness["official_nv"]))
            official1 = Base.invokelatest(LatticeAlgorithms.get_prob, eta - logical, whichtype, sigma; Nv=Int(correctness["official_nv"]))
            ours_log10_odds = (score1 - score0) / log(10)
            official_log10_odds = official1 - official0
            maximum_log_odds_error = max(maximum_log_odds_error, abs(ours_log10_odds - official_log10_odds))
            official_mismatches += (score1 > score0) != (official1 > official0)
        end
        if sample <= alias_samples
            short0, short1 = parity_loglikelihoods(residuals, fill(sigma, 9); terms=Int(config["posterior"]["alias_terms"]))
            short_score0, short_score1 = coset_scores(table, syndrome, short0, short1)
            alias_mismatches += (score1 > score0) != (short_score1 > short_score0)
        end
        if sample <= tjoin_samples
            noise = sigma .* randn(rng, 18)
            prepared_q = prepare_subspace(noise[1:2:end], zstabs)
            prepared_p = prepare_subspace(noise[2:2:end], xstabs)
            qweights = gaussian_cpd_weights(prepared_q, ones(9))
            pweights = gaussian_cpd_weights(prepared_p, ones(9))
            pure_q = decode_cpd(prepared_q, zstabs, qweights; backend=:pure_julia)
            pure_p = decode_cpd(prepared_p, xstabs, pweights; backend=:pure_julia)
            official_q = decode_cpd(prepared_q, zstabs, qweights; backend=:official)
            official_p = decode_cpd(prepared_p, xstabs, pweights; backend=:official)
            tjoin_correction_mismatches += (pure_q != official_q) || (pure_p != official_p)
        end
    end
    cardinalities = [
        size(table.coset0[key], 1) for table in [qtable, ptable] for key in keys(table.coset0)
    ]
    return Dict(
        "official_bsv_samples" => samples,
        "official_bsv_action_mismatches" => official_mismatches,
        "maximum_official_log10_odds_error" => maximum_log_odds_error,
        "alias_convergence_samples" => alias_samples,
        "alias_action_mismatches" => alias_mismatches,
        "pure_julia_tjoin_samples" => tjoin_samples,
        "pure_julia_vs_official_correction_mismatches" => tjoin_correction_mismatches,
        "maximum_probability_normalization_error" => normalization_error,
        "coset_cardinality_min" => minimum(cardinalities),
        "coset_cardinality_max" => maximum(cardinalities),
        "passed" => official_mismatches == Int(correctness["required_action_mismatches"]) &&
            tjoin_correction_mismatches == 0 &&
            alias_mismatches == 0 && maximum_log_odds_error <= Float64(correctness["maximum_log_odds_error"]) &&
            normalization_error <= Float64(correctness["maximum_probability_normalization_error"]) &&
            minimum(cardinalities) == 16 && maximum(cardinalities) == 16,
    )
end

function causal_probe(noise_trace, cell, config, tables)
    qtable, ptable, xstabs, zstabs, _, _ = tables
    base_sigma = Float64(cell["base_sigma"])
    filter = build_filter(config["posterior"]; cadence=1)
    moments = build_circular_estimator(
        2Int(config["distance"])^2, base_sigma, config["posterior"];
        cadence=1, alpha=Float64(config["posterior"]["circular_alpha_fast"]),
    )
    terms = Int(config["posterior"]["alias_terms"])
    actions = Tuple{Int,Int}[]
    priors = Vector{Vector{Float64}}()
    for noise in noise_trace
        push!(priors, copy(filter.posterior))
        sigmas, means = circular_parameters(moments)
        prepared_q = prepare_subspace(noise[1:2:end], zstabs)
        prepared_p = prepare_subspace(noise[2:2:end], xstabs)
        q_scores0, q_scores1 = exact_state_scores(
            qtable, vector_key(prepared_q.syndrome), prepared_q.residuals,
            sigmas[1:2:end], means[1:2:end], filter.grid; terms=terms,
        )
        p_scores0, p_scores1 = exact_state_scores(
            ptable, vector_key(prepared_p.syndrome), prepared_p.residuals,
            sigmas[2:2:end], means[2:2:end], filter.grid; terms=terms,
        )
        q_action, _ = mixture_action(q_scores0, q_scores1, filter.posterior)
        p_action, _ = mixture_action(p_scores0, p_scores1, filter.posterior)
        push!(actions, (q_action, p_action))
        residuals = zeros(Float64, length(sigmas))
        residuals[1:2:end] .= prepared_q.residuals
        residuals[2:2:end] .= prepared_p.residuals
        observation = observation_loglikelihood(residuals, sigmas, means, filter.grid; terms=terms)
        update_filter!(filter, observation)
        observe_circular!(moments, residuals)
    end
    return actions, priors
end

function causal_prefix_check(config, cell, tables)
    rounds = 128
    prefix = Int(config["correctness"]["causal_prefix_rounds"])
    rng = MersenneTwister(620049997)
    base_sigma = Float64(cell["base_sigma"])
    original_trace = [base_sigma .* randn(rng, 2Int(config["distance"])^2) for _ in 1:rounds]
    mutated_trace = deepcopy(original_trace)
    for cycle in (prefix + 1):rounds
        mutated_trace[cycle] .= 0.49SQRT_PI .* (-1.0) .^ collect(1:length(mutated_trace[cycle]))
    end
    actions_a, priors_a = causal_probe(original_trace, cell, config, tables)
    actions_b, priors_b = causal_probe(mutated_trace, cell, config, tables)
    prefix_action_mismatches = count(actions_a[index] != actions_b[index] for index in 1:prefix)
    prefix_prior_max_abs_error = maximum(maximum(abs.(priors_a[index] .- priors_b[index])) for index in 1:prefix)
    suffix_action_divergence = any(actions_a[index] != actions_b[index] for index in (prefix + 1):rounds)
    suffix_prior_divergence = any(maximum(abs.(priors_a[index] .- priors_b[index])) > 1e-12 for index in (prefix + 2):rounds)
    return Dict(
        "prefix_rounds" => prefix,
        "prefix_action_mismatches" => prefix_action_mismatches,
        "prefix_prior_max_abs_error" => prefix_prior_max_abs_error,
        "mutated_suffix_action_divergence" => suffix_action_divergence,
        "mutated_suffix_prior_divergence" => suffix_prior_divergence,
        "passed" => prefix_action_mismatches == 0 && prefix_prior_max_abs_error == 0.0 &&
            suffix_action_divergence && suffix_prior_divergence,
    )
end

function main()
    opts = parse_args(ARGS)
    config_path = normpath(opts["config"])
    config = JSON.parsefile(config_path)
    manifest_path = normpath(joinpath(REPO_ROOT, config["source_manifest"]))
    manifest = JSON.parsefile(manifest_path)
    split_id = String(config["source_split"])
    allowed = String.(config["eligibility_boundary"]["allowed_split_ids"])
    split_id in allowed || error("source split is not allowed")
    split = only([row for row in manifest["splits"] if row["split_id"] == split_id])
    patterns = Dict(
        row["pattern_sha256"] => row for row in split["spatial_patterns"] if Int(row["distance"]) == Int(config["distance"])
    )
    cells = [row for row in manifest["execution_cells"] if row["split_id"] == split_id && Int(row["distance"]) == Int(config["distance"])]
    expected_families = String.(manifest["config_snapshot"]["scenario_families"])
    actual_families = sort(unique(String(row["scenario_family"]) for row in cells))
    sort(expected_families) == actual_families || error("train family coverage mismatch")
    seed_limit = parse(Int, opts["seed-limit"])
    family_limit = parse(Int, opts["family-limit"])
    seeds = sort(unique(Int(row["seed"]) for row in cells))
    seed_limit > 0 && (seeds = seeds[1:min(seed_limit, length(seeds))])
    families = expected_families
    family_limit > 0 && (families = families[1:min(family_limit, length(families))])
    cells = [row for row in cells if Int(row["seed"]) in seeds && String(row["scenario_family"]) in families]
    rounds = parse(Int, opts["rounds"]) > 0 ? parse(Int, opts["rounds"]) : Int(config["rounds_per_cell"])
    tables = build_tables(Int(config["distance"]))
    correctness = explicit_probability_checks(config, tables)
    isempty(cells) && error("no development cells selected")
    causality = causal_prefix_check(config, cells[1], tables)
    rows = Vector{Dict{String,Any}}()
    started = time()
    if lowercase(opts["correctness-only"]) != "true"
        for cell in cells
            row = simulate_cell(cell, patterns[cell["spatial_pattern_sha256"]], rounds, config, tables)
            push!(rows, row)
            println(JSON.json(Dict(
                "cell_id" => cell["cell_id"], "completed" => length(rows), "total" => length(cells),
                "wall_clock_seconds" => time() - started,
            )))
            flush(stdout)
        end
    end
    official_root = joinpath(REPO_ROOT, "third_party", "LatticeAlgorithms.jl")
    result = Dict(
        "schema_version" => "t6.20.4-causal-headroom-raw-v1",
        "task_id" => "T6.20.4",
        "generated_at" => string(now()),
        "config_path" => relpath(config_path, REPO_ROOT),
        "config_sha256" => sha256_file(config_path),
        "source_manifest_path" => relpath(manifest_path, REPO_ROOT),
        "source_manifest_sha256" => sha256_file(manifest_path),
        "source_split" => split_id,
        "selected_seed_count" => length(seeds),
        "selected_family_count" => length(families),
        "selected_cell_count" => length(cells),
        "rounds_per_cell" => rounds,
        "official_repository_head" => git_head(official_root),
        "julia_version" => string(VERSION),
        "correctness" => correctness,
        "causality" => causality,
        "rows" => rows,
        "wall_clock_seconds" => time() - started,
    )
    output = normpath(opts["output"])
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, result, 2)
        write(io, "\n")
    end
    println(JSON.json(Dict("output" => output, "rows" => length(rows), "correctness" => correctness["passed"])))
end

main()

## Universal priors defined based on the problem domain and ampltiude & noise std estimates.

function get_lengthscale_priors(problem::AbstractProblem)
    bounds = domain(problem).bounds
    ydim = y_dim(problem)
    return get_lengthscale_priors(bounds, ydim)
end
function get_lengthscale_priors(bounds::AbstractBounds, ydim::Int)
    d = (bounds[2] .- bounds[1])

    min_λs = d ./ 20
    max_λs = d

    μs = (log.(min_λs) .+ log.(max_λs)) ./ 2
    σs = (log.(max_λs) .- log.(min_λs)) ./ 2 # bounds are within 2 stds
    dists = LogNormal.(μs, σs)

    dists = map((d, max_λ) -> truncated(d; upper=max_λ), dists, max_λs)

    return fill(product_distribution(dists), ydim)
end

function get_amplitude_priors(problem::AbstractProblem)
    est_α = est_amplitude(problem)
    return get_amplitude_priors(est_α)
end
function get_amplitude_priors(est_α::AbstractVector{<:Real})
    d = TDist(2)
    d = truncated(d; lower=0.)
    dists = transformed.(Ref(d), Bijectors.Scale.(est_α))
    return dists
end

function get_noise_std_priors(problem::AbstractProblem)
    est_σ = est_noise_std(problem)

    if isnothing(est_σ)
        # we know that the simulator is noiseless
        return fill(Dirac(1e-6), y_dim(problem))
    end

    d = TDist(2)
    d = truncated(d; lower=0.)
    dists = transformed.(Ref(d), Bijectors.Scale.(est_σ))

    return dists
end

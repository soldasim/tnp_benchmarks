
"""
Subtypes of `AbstractProblem` represent benchmark problems for SBI.

Each subtype of `AbstractProblem` *should* implement:
- `simulator(::AbstractProblem) -> ::Function`
- `domain(::AbstractProblem) -> ::Domain`
- `y_max(::AbstractProblem) -> ::Union{Nothing, AbstractVector{<:Real}}`: Optional, defaults to `nothing`.
- `likelihood(::AbstractProblem) -> ::Likelihood`
- `prior_mean(::AbstractProblem) -> ::AbstractVector{<:Real}`
- `x_prior(::AbstractProblem) -> ::MultivariateDistribution`
- `est_amplitude(::AbstractProblem) -> ::AbstractVector{<:Real}`
- `est_noise_std(::AbstractProblem) -> ::AbstractVector{<:Real} or ::Nothing`

Each subtype of `AbstractProblem` *should* implement *at least one* of:
- `true_f(::AbstractProblem) -> ::Union{Nothing, Function}`: Defaults to `nothing`.
- `reference_samples(::AbstractProblem) -> ::Union{Nothing, Matrix{Float64}}`: Defaults to `nothing`.

The reference solution can be obtained by:
- `reference(::AbstractProblem) -> ::Union{Function, Matrix{Float64}}`:
    Returns either the `true_logpost` or the `reference_samples`
    depending on which of the `true_f` and `reference_samples` function have been defined for the problem.
- `true_loglike(::AbstractProblem) -> ::Union{Nothing, Function}`: Returns the true likelihood
    if `true_f` is defined for the given problem.
- `true_logpost(::AbstractProblem) -> ::Union{Nothing, Function}`: Returns the true posterior
    if `true_f` is defined for the given problem.

Each `AbstractProblem` additionally provides default implementations for:
- `x_dim(::AbstractProblem) -> ::Int`
- `y_dim(::AbstractProblem) -> ::Int`
"""
abstract type AbstractProblem end

"""
    simulator(::AbstractProblem) -> ::Function

Return the simulator function (i.e. the expensive simulator) of the given problem.
"""
function simulator end

"""
    domain(::AbstractProblem) -> ::Domain

Return the parameter domain of the given problem.
"""
function domain end

"""
    likelihood(::AbstractProblem) -> ::Likelihood

Return the likelihood function of the given problem.
"""
function likelihood end

"""
    prior_mean(::AbstractProblem) -> ::AbstractVector{<:Real}

Return the prior mean of the simulator outputs.
This is closely related to the `likelihood` definition.
Usually, it is reasonable to set the prior mean for the simulation outputs
such that it maximizes the likelihood. (E.g. set it to `y_obs` in case of the Gaussian likelihood.)
"""
function prior_mean end

"""
    x_prior(::AbstractProblem) -> ::MultivariateDistribution

Return the prior parameter distribution of the given problem.
"""
function x_prior end

"""
    est_amplitude(::AbstractProblem) -> ::AbstractVector{<:Real}

Return the (estimated) amplitude.

The amplitude prior is initialized to support values similar to the estimated amplitude.
"""
function est_amplitude end

"""
    est_noise_std(::AbstractProblem) -> ::AbstractVector{<:Real} or ::Nothing

Return the (estimated) noise standard deviation.

The noise std prior is initialized to support values similar to the estimated noise standard deviation.

Return `nothing` if the problem is noise-less (and we assume to have this knowledge).
"""
function est_noise_std end

"""
    true_f(::AbstractProblem) -> ::Union{Nothing, Function}

Return the true noise-less simulator function or `nothing` if the true function cannot be evaluated
or is too expensive to compute.

Either `true_f` or `reference_samples` should be defined for each problem.
"""
true_f(::AbstractProblem) = nothing

"""
    reference_samples(::AbstractProblem) -> ::Union{Nothing, Matrix{Float64}}

Return reference samples from the true parameter posterior for the given problem
or `nothing` if the reference samples are not provided.

Either `reference_samples` or `true_f` should be defined for each problem.
"""
reference_samples(::AbstractProblem) = nothing


"""
    y_max(::AbstractProblem) -> ::Union{Nothing, AbstractVector{<:Real}}

Return the constraint values for the outputs of the given problem.
"""
y_max(::AbstractProblem) = nothing

"""
    reference(::AbstractProblem) -> ::Union{Function, Matrix{Float64}}

Returns either the `true_posterior` or the `reference_samples`
depending on which of the `true_f` and `reference_samples` function have beed defined for the problem.
"""
function reference(problem::AbstractProblem)
    logpost = true_logpost(problem)
    isnothing(logpost) || return logpost
    ref_samples = reference_samples(problem)
    isnothing(ref_samples) && error("Define `f_true` or `reference_samples` for the problem.")
    return ref_samples
end

"""
    true_loglike(::AbstractProblem) -> ::Union{Nothing, Function}

Return the true log-likelihood function of the given problem
or `nothing` if the problem does not have the `true_f` function defined.

The true log-likelihood can be used to evaluate performance metrics.
"""
function true_loglike(problem::AbstractProblem)
    like = likelihood(problem)
    f = true_f(problem)

    isnothing(f) && return nothing

    function true_like(x)
        y = f(x)
        ll = loglike(like, y, x)
        return ll
    end
end

"""
    true_logpost(::AbstractProblem) -> ::Union{Nothing, Function}

Return the true log-posterior function of the given problem
or `nothing` if the problem does not have the `true_f` function defined.

The true log-posterior can be used to evaluate performance metrics.
"""
function true_logpost(problem::AbstractProblem)
    like = likelihood(problem)
    prior = x_prior(problem)
    f = true_f(problem)

    isnothing(f) && return nothing

    function true_post(x::AbstractVector{<:Real})
        y = f(x)
        ll = loglike(like, y, x)
        lp = logpdf(prior, x)
        return ll + lp
    end
    function true_post(X::AbstractMatrix{<:Real})
        ps = similar(X, size(X, 2))
        Threads.@threads for i in 1:size(X, 2)
            ps[i] = true_post(@view X[:,i])
        end
        return ps
    end
    return true_post
end

x_dim(problem::AbstractProblem) = length(domain(problem).bounds[1])
y_dim(problem::AbstractProblem) = length(est_amplitude(problem))


"""
    BimodalProblem()

The "bimodal" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

In contrast to the problem as defined in the paper, here the vector `[x[1], x[2] + x[1]^2 + 1.]`
is returned as the simulator output instead of the log-likelihood value.
See the `LogSimpleProblem` for the original version of the problem.
"""
struct BimodalProblem <: AbstractProblem end


module BimodalProblemModule

import ..BimodalProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOSIP
using Distributions
using Bijectors

# --- API ---

simulator(::BimodalProblem) = simulation

domain(::BimodalProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::BimodalProblem) = get_likelihood()

prior_mean(::BimodalProblem) = [0., 0.]

x_prior(::BimodalProblem) = get_x_prior()

# est_amplitude(::BimodalProblem) = fill(20., 2)
est_amplitude(::BimodalProblem) = fill(1., 2)

# TODO noise
est_noise_std(::BimodalProblem) = nothing

true_f(::BimodalProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
# get_bounds() = (fill(-6., x_dim()), fill(6., x_dim()))
get_bounds() = (fill(-1., x_dim()), fill(1., x_dim()))


# - - - EXPERIMENT - - - - -

const ρ = 0.5
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

# function f_(x)
#     θ = [x[1], x[2]^2 - 2]
#     return -(1/2) * θ' * inv_S * θ
# end

function simulation(x)
    x1 = 6 * x[1]
    x2 = 6 * x[2]

    θ = [x1, x2^2 - 2]

    θ ./= 20.
    return θ
end

get_likelihood() = MvNormalLikelihood(;
    z_obs = [0., 0.],
    Σ_obs = Σ ./ (20^2),
)

# truncate the prior to the bounds
function get_x_prior()
    return Product(Uniform.(get_bounds()...))
end

end # module BimodalProblemModule

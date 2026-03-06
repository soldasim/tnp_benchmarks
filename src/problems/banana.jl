
"""
    BananaProblem()

The "banana" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

In contrast to the problem as defined in the paper, here the vector `[x[1], x[2] + x[1]^2 + 1.]`
is returned as the simulator output instead of the log-likelihood value.
See the `LogSimpleProblem` for the original version of the problem.
"""
struct BananaProblem <: AbstractProblem end


module BananaProblemModule

import ..BananaProblem

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

simulator(::BananaProblem) = simulation

domain(::BananaProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::BananaProblem) = get_likelihood()

prior_mean(::BananaProblem) = [0., 0.]

x_prior(::BananaProblem) = get_x_prior()

# est_amplitude(::BananaProblem) = fill(20., 2)
est_amplitude(::BananaProblem) = fill(1., 2)

# TODO noise
est_noise_std(::BananaProblem) = nothing

true_f(::BananaProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
# get_bounds() = ([-6., -20.], [6., 2.])
get_bounds() = ([-1., -1.], [1., 1.])


# - - - EXPERIMENT - - - - -

const ρ = 0.9
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

# function f_(x)
#     θ = [x[1], x[2] + x[1]^2 + 1.]
#     return -(1/2) * θ' * inv_S * θ
# end

function simulation(x)
    x1 = 6 * x[1]
    x2 = 11 * x[2] - 9

    θ = [x1, x2 + x1^2 + 1.]

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

end # module BananaProblemModule

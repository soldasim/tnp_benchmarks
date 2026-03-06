"""
    DuffingProblem()

The Duffing oscillator problem for simulation-based inference.

The Duffing oscillator is a nonlinear second-order differential equation:
    ẍ + δẋ + αx + βx³ = γcos(ωt)

where:
- δ: damping coefficient
- α: linear stiffness coefficient  
- β: nonlinear stiffness coefficient
- γ: driving amplitude
- ω: driving frequency

This implementation simulates the forced Duffing oscillator and observes
the steady-state response characteristics to infer the parameters.

The parameters to infer are [δ, α, β] with fixed γ=0.3 and ω=1.0.
The observations are time series samples of the displacement x(t).
"""
struct DuffingProblem <: AbstractProblem end

module DuffingModule

import ..DuffingProblem

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
using DifferentialEquations
using JLD2


# --- API ---

simulator(::DuffingProblem) = model_target

domain(::DuffingProblem) = Domain(;
    bounds = _get_bounds(),
)

likelihood(::DuffingProblem) = NormalLikelihood(; z_obs, std_obs)

prior_mean(::DuffingProblem) = _get_prior_mean()

x_prior(::DuffingProblem) = _get_trunc_x_prior()

est_amplitude(::DuffingProblem) = _get_est_amplitude()

# TODO noise
est_noise_std(::DuffingProblem) = nothing

true_f(::DuffingProblem) = model_target
reference_samples(::DuffingProblem) = load(joinpath(@__DIR__, "duffing_ref.jld2"))["xs"]


# --- UTILS ---

normalize_input(x) = (x .- [0.05, -2.0, 0.05]) ./ ([1.0, 2.0, 2.0] .- [0.05, -2.0, 0.05]) .* 2 .- 1
denormalize_input(x) = (x .+ 1) ./ 2 .* ([1.0, 2.0, 2.0] .- [0.05, -2.0, 0.05]) .+ [0.05, -2.0, 0.05]
normalize_output(y) = y ./ 2.
denormalize_output(y) = y .* 2.

# Fixed parameters
const γ = 0.65    # driving amplitude
const ω = 2π / 1  # driving frequency (one period = 1s)

# Reference parameters: δ, α, β
const x_ref = [0.15, -1.0, 0.5]

# Observation parameters
const t_span = (0.0, 10.0)  # simulation time span
const t_transient = 0.0     # transient time to ignore
const save_freq = 0.05
const n_obs = 1             # number of observation points

# Noise parameters
# const std_obs = fill(0.1, n_obs)  # observation noise
const std_obs = fill(0.05, n_obs)  # no observation noise

"""
    duffing_ode!(du, u, p, t)

Duffing oscillator ODE system.
u[1] = x (position), u[2] = ẋ (velocity)
p = [δ, α, β] (parameters to infer)

# Duffing equation:
#   ẍ + δẋ + αx + βx³ = γcos(ωt)
# Rewritten as a system:
#   du[1] = ẋ
#   du[2] = -δ*ẋ - α*x - β*x^3 + γ*cos(ω*t)
"""
function duffing_ode!(du, u, p, t)
    δ, α, β = p
    x, x_dot = u
    
    du[1] = x_dot
    du[2] = -δ*x_dot - α*x - β*x^3 + γ*cos(ω*t)
end

"""
    model_target(x_)

The Duffing problem simulator together with the mapping to the model target variable.
"""
function model_target(x_)
    x_ = denormalize_input(x_)
    
    sol = duffing_simulation(x_)
    positions, velocities, times = extract_measurements(sol)

    positions = normalize_output(positions)
    return positions
end

"""
    duffing_simulation(x)

Simulate the Duffing oscillator with parameters x = [δ, α, β].
Returns time series observations of the displacement.
"""
function duffing_simulation(x_)
    δ, α, β = x_

    # Initial conditions (start near equilibrium)
    u0 = [0.0, 0.0]
    
    # Set up and solve ODE
    prob = ODEProblem(duffing_ode!, u0, t_span, [δ, α, β])
    sol = solve(prob, Tsit5(), saveat=save_freq)

    return sol
end

"""
    extract_measurements(sol)

Extract position and velocity measurements from the simulation solution `sol`.
"""
function extract_measurements(sol)
    # Extract position data after transient period
    t_indices = findall(t -> t > t_transient, sol.t)
    positions = [sol.u[i][1] for i in t_indices]
    velocities = [sol.u[i][2] for i in t_indices]
    
    # Downsample to desired number of observations
    step = length(t_indices) ÷ n_obs
    @assert step > 0
    obs_indices = step:step:(step*n_obs)
    positions = positions[obs_indices[1:n_obs]]
    velocities = velocities[obs_indices[1:n_obs]]
    times = sol.t[obs_indices[1:n_obs]]

    return positions, velocities, times
end

"""
Generate reference observation data with known parameters.
"""
function _generate_reference_data()
    sol = duffing_simulation(x_ref)
    positions, velocities, times = extract_measurements(sol)

    positions = normalize_output(positions)
    return positions
end

# Generate synthetic observation data (example trajectory)
const z_obs = _generate_reference_data()

"""
Parameter bounds: [δ_min, α_min, β_min], [δ_max, α_max, β_max]
"""
# _get_bounds() = ([0.05, -2.0, 0.05], [1.0, 2.0, 2.0])
_get_bounds() = ([-1., -1., -1.], [1., 1., 1.])

"""
Prior mean based on typical parameter values.
"""
function _get_prior_mean()
    return z_obs
end

"""
Estimated amplitude for each observation dimension.
"""
# _get_est_amplitude() = fill(2.0, n_obs)
_get_est_amplitude() = fill(1.0, n_obs)

"""
Truncated prior distribution for parameters.
"""
function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end

"""
Prior distribution are automatically truncated to the parameter bounds.
Thus, we can use Normal distributions even for positive parameters.
"""
function _get_x_prior()
    dists = [
        Normal(0.1, 0.5), # δ: damping coefficient
        Normal(0.0, 1.0), # α: linear stiffness
        Normal(0.0, 1.0), # β: nonlinear stiffness
    ]
    dists = normalize_input(dists)
    return Product(dists)
end


# ### Parametrization Conversions

# """
# Convert position-velocity parametrization
# to the amplitude-phase parametrization.
# """
# function posvel_to_amppha(pos, vel)
#     amp = sqrt(pos^2 + (vel / ω)^2)
#     pha = atan(- (vel / ω), pos)
#     return amp, pha
# end

# """
# Convert amplitude-phase parametrization
# to the position-velocity parametrization.
# """
# function amppha_to_posvel(amp, pha)
#     pos = amp * cos(pha)
#     vel = (-1) * amp * ω * sin(pha)
#     return pos, vel
# end

end # module DuffingModule

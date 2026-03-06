"""
    DiffusionProblem(; kwargs...)

The advection-diffusion equation problem for simulation-based inference.

The advection-diffusion equation is a partial differential equation:
    ∂u/∂t = D (∂²u/∂x² + ∂²u/∂y²) - A_adv * (v_x(x,y) ∂u/∂x + v_y(x,y)         # Update derivative: diffusion - advection + source
        du[idx] = D * (d2u_dx2 + d2u_dy2) - A_adv * (v_x * du_dx + v_y * du_dy) + S_mult * source/∂y) + S(x,y,t)

where:
- u(x,y,t): concentration field
- D: diffusion coefficient (fixed)
- A_adv: advection coefficient (fixed)
- v_x(x,y), v_y(x,y): wind velocity components (spatially varying)
- S(x,y,t): source term

This implementation solves the 2D advection-diffusion equation with:
- Initial condition: zero concentration everywhere
- Boundary conditions: zero flux at boundaries
- Wind field: circular flow pattern with some randomness
- Source term: S(x,y,t) = A*exp(-((x-x_s)² + (y-y_s)²)/(2σ_source²)) for t_s < t < t_s + Δt, 0 otherwise

The parameters to infer are [x_s, y_s, t_s] where:
- x_s: source x-location
- y_s: source y-location
- t_s: source activation time

The source starts radiating at unknown time t_s at unknown location (x_s, y_s) with fixed amplitude A = 1.0 and continues for a duration Δt,
where t_s ∈ [-10.0, -0.5], x_s ∈ [-5.0, 5.0], y_s ∈ [-5.0, 5.0].

The observations are integrated concentration measurements at 3 locations arranged in a triangle over configurable time intervals during the full simulation time (-10, 0). Reference data uses 10-second intervals (entire simulation), while model outputs use 2-second intervals by default.

## Keywords
- `model_interval::Float64=10.0`: The length of the modeled intervals.
"""
@kwdef struct DiffusionProblem <: AbstractProblem
    model_interval::Float64 = 10.0
end

module DiffusionModule

import ..DiffusionProblem

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


# --- API ---

simulator(p::DiffusionProblem) = _get_model_target(p)

domain(::DiffusionProblem) = Domain(;
    bounds = _get_bounds(),
)

likelihood(p::DiffusionProblem) = _get_likelihood(p)

prior_mean(p::DiffusionProblem) = _get_prior_mean(p)

x_prior(::DiffusionProblem) = _get_trunc_x_prior()

###
# est_amplitude(p::DiffusionProblem) = _get_est_amplitude(p)
est_amplitude(p::DiffusionProblem) = fill(1., n_obs_locs * n_modeled_times(p))

# TODO noise
est_noise_std(::DiffusionProblem) = nothing

true_f(p::DiffusionProblem) = _get_model_target(p)


# --- UTILS ---

# Spatial discretization
const x_grid = collect(range(-5., 5.; length=51))  # reduced for 2D
const y_grid = collect(range(-5., 5.; length=51))  # reduced for 2D
const nx = length(x_grid)               # number of x grid points
const ny = length(y_grid)               # number of y grid points
const Lx = (x_grid[end] - x_grid[begin]) # x domain length
const Ly = (y_grid[end] - y_grid[begin]) # y domain length
const dx = Lx / (nx - 1)                 # x spatial step size
const dy = Ly / (ny - 1)                 # y spatial step size

# Time parameters
const t_span = (-10, 0.0)  # simulation time span

# Measurement intervals
const observation_interval = 10.0   # interval length for reference data (entire simulation)
### MOVED TO THE PROBLEM STRUCTURE ###
# const model_interval = 10.0         # interval length for model outputs

# Simulation fidelity
const sim_step = 0.1

# The `Likelihood` describes how the model output is mapped to observation likelihood values
function _get_likelihood(p::DiffusionProblem)
    ratio = Int(observation_interval / p.model_interval)
    if ratio == 1
        return LogNormalLikelihood(;
            log_z_obs = log.(z_obs),
            CV,
        )
    else
        return LogNormalSumLikelihood(;
            sum_lengths = fill(ratio, n_obs_locs * n_obs_times),
            log_z_obs = log.(z_obs),
            CV,
        )
    end
end

# Fixed parameters
const D = 0.05          # diffusion coefficient (fixed)
const A_adv = 10.0      # advection coefficient (fixed)
const S_mult = 200.0    # source multiplier coefficient (fixed)

# Wind parameters
const v_max = 0.8       # maximum wind speed
const wind_scale = 2.0  # spatial scale of wind variations

# Source parameters
const Δt_source = 0.5     # source duration
const σ_source = 0.5      # source width
const A_source = 100.0    # source amplitude (fixed)

# Pre-compute wind field at grid points
const v_x_field = [v_max * sin(wind_scale * y_grid[j] / 5.0) * cos(wind_scale * x_grid[i] / 5.0) 
                   for j in 1:ny, i in 1:nx]
const v_y_field = [-v_max * cos(wind_scale * y_grid[j] / 5.0) * sin(wind_scale * x_grid[i] / 5.0) 
                   for j in 1:ny, i in 1:nx]

# Observation parameters (3 points in equilateral triangular arrangement)
const obs_points = [(-2.0, -2.0), (2.0, -2.0), (0.0, 1.4)]  # equilateral triangle observation points
const n_obs_locs = length(obs_points)  # total observations for reference data (3 points × 1 interval = 3)
const n_obs_times = Int((t_span[2] - t_span[1]) / observation_interval)  # number of observation intervals
n_modeled_times(p) = Int((t_span[2] - t_span[1]) / p.model_interval)  # number of 2-second intervals (5)

# Assert that observation points are exactly at grid points
for (x_loc, y_loc) in obs_points
    @assert x_loc in x_grid "Observation x-coordinate $x_loc not in grid"
    @assert y_loc in y_grid "Observation y-coordinate $y_loc not in grid"
end

# Noise parameters
const CV = fill(0.1, n_obs_locs * n_obs_times)  # coefficient of variation of the observations

const INIT_CONCENTRATION = 1e-2  # the initial concentration everywhere
const MIN_MEASURED_VALUE_PER_TIME_UNIT = 1e-2 / 10.  # minimum measurable concentration (to avoid -inf likelihoods)

# [x_s, y_s, t_s]: position and activation time of the source (A is now fixed)
const x_ref = [1.5, 1.5, -4.0]

###
normalize_inputs(x) = (x .+ [0., 0., 5.25]) ./ [5., 5., 4.75]
denormalize_inputs(x) = x .* [5., 5., 4.75] .- [0., 0., 5.25]

###
α_est(p) = (A_source * p.model_interval) / (observation_interval / p.model_interval)
const α_est_ = α_est(DiffusionProblem())
normalize_observations(y) = y ./ α_est_
denormalize_observations(y) = y .* α_est_

"""
    diffusion_pde!(du, u, p, t)

2D advection-diffusion PDE discretized in space using finite differences.
u: concentration field at spatial grid points (flattened 2D array)
p = [x_s, y_s, t_s]: parameters (source location and activation time, amplitude is fixed)
"""
function diffusion_pde!(du, u, p, t)
    x_s, y_s, t_s = p
    
    # Initialize all derivatives to zero
    fill!(du, 0.0)
    
    # Check if source is active
    source_active = (t_s <= t < t_s + Δt_source)
    
    # Process all grid points
    for j in 1:ny, i in 1:nx
        idx = (j-1)*nx + i
        
        # Compute diffusion terms with zero-flux boundary conditions
        d2u_dx2 = 0.0
        d2u_dy2 = 0.0
        
        # X-direction second derivative (zero-flux: ∂u/∂x = 0 at boundaries)
        if i == 1  # Left boundary: ∂u/∂x = 0, so u[0] = u[1]
            d2u_dx2 = (u[idx+1] - 2*u[idx] + u[idx]) / dx^2  # u[idx-1] = u[idx] (ghost point)
        elseif i == nx  # Right boundary: ∂u/∂x = 0, so u[nx+1] = u[nx]
            d2u_dx2 = (u[idx] - 2*u[idx] + u[idx-1]) / dx^2  # u[idx+1] = u[idx] (ghost point)
        else  # Interior
            d2u_dx2 = (u[idx+1] - 2*u[idx] + u[idx-1]) / dx^2
        end
        
        # Y-direction second derivative (zero-flux: ∂u/∂y = 0 at boundaries)
        if j == 1  # Bottom boundary: ∂u/∂y = 0, so u[j-1] = u[j]
            d2u_dy2 = (u[idx+nx] - 2*u[idx] + u[idx]) / dy^2  # u[idx-nx] = u[idx] (ghost point)
        elseif j == ny  # Top boundary: ∂u/∂y = 0, so u[j+1] = u[j]
            d2u_dy2 = (u[idx] - 2*u[idx] + u[idx-nx]) / dy^2  # u[idx+nx] = u[idx] (ghost point)
        else  # Interior
            d2u_dy2 = (u[idx+nx] - 2*u[idx] + u[idx-nx]) / dy^2
        end
        
        # Advection terms (upwind scheme for stability)
        du_dx = 0.0
        du_dy = 0.0
        
        v_x = v_x_field[j, i]
        v_y = v_y_field[j, i]
        
        # X-direction first derivative (upwind with proper boundary conditions)
        if i == 1  # Left boundary
            if v_x >= 0  # Inflow: use zero-flux condition (∂u/∂x = 0)
                du_dx = 0.0
            else  # Outflow: use upwind (forward difference)
                du_dx = (u[idx+1] - u[idx]) / dx
            end
        elseif i == nx  # Right boundary
            if v_x >= 0  # Outflow: use upwind (backward difference)
                du_dx = (u[idx] - u[idx-1]) / dx
            else  # Inflow: use zero-flux condition (∂u/∂x = 0)
                du_dx = 0.0
            end
        else  # Interior: standard upwind
            if v_x >= 0
                du_dx = (u[idx] - u[idx-1]) / dx  # Upwind (backward)
            else
                du_dx = (u[idx+1] - u[idx]) / dx  # Upwind (forward)
            end
        end
        
        # Y-direction first derivative (upwind with proper boundary conditions)
        if j == 1  # Bottom boundary
            if v_y >= 0  # Inflow: use zero-flux condition (∂u/∂y = 0)
                du_dy = 0.0
            else  # Outflow: use upwind (forward difference)
                du_dy = (u[idx+nx] - u[idx]) / dy
            end
        elseif j == ny  # Top boundary
            if v_y >= 0  # Outflow: use upwind (backward difference)
                du_dy = (u[idx] - u[idx-nx]) / dy
            else  # Inflow: use zero-flux condition (∂u/∂y = 0)
                du_dy = 0.0
            end
        else  # Interior: standard upwind
            if v_y >= 0
                du_dy = (u[idx] - u[idx-nx]) / dy  # Upwind (backward)
            else
                du_dy = (u[idx+nx] - u[idx]) / dy  # Upwind (forward)
            end
        end
        
        # Source term
        source = 0.0
        if source_active
            r_squared = (x_grid[i] - x_s)^2 + (y_grid[j] - y_s)^2
            source = A_source * exp(-r_squared / (2*σ_source^2))
        end
        
        # Update derivative: diffusion - advection + source
        du[idx] = D * (d2u_dx2 + d2u_dy2) - A_adv * (v_x * du_dx + v_y * du_dy) + source
    end
end

function _get_model_target(p::DiffusionProblem)
    function model_target(x)
        x = denormalize_inputs(x)
        sol = diffusion_simulation(x)
        observations = extract_measurements(sol; interval_length=p.model_interval)
        observations = normalize_observations(observations)
        return log.(observations)
    end
end

"""
    diffusion_simulation(x)

Simulate the 2D diffusion equation with parameters x = [x_s, y_s, t_s].
Returns the full ODE solution.
"""
function diffusion_simulation(x::AbstractVector{<:Real})
    x_s, y_s, t_s = x
    
    # Initial condition: zero concentration at all grid points
    u0 = INIT_CONCENTRATION * ones(Float64, nx * ny)
    
    # Set up and solve PDE
    # start the simulation when source activates
    t_span_ = (max(t_span[1], t_s), t_span[2])
    prob = ODEProblem(diffusion_pde!, u0, t_span_, [x_s, y_s, t_s])
    sol = solve(prob, Tsit5(), saveat=sim_step)
    
    return sol
end

"""
    extract_measurements(sol; interval_length)

Extract concentration measurements from the simulation solution at specified observation points.
Integrates concentration over specified intervals using trapezoidal integration.
"""
function extract_measurements(sol; interval_length)
    n_intervals_local = Int((t_span[2] - t_span[1]) / interval_length)
    n_total = n_intervals_local * length(obs_points)
    y = zeros(Float64, n_total)
    
    # Pre-compute observation point indices
    obs_indices = [(findfirst(==(x), x_grid), findfirst(==(y), y_grid)) for (x, y) in obs_points]
    
    for interval_idx in 1:n_intervals_local
        # Define time bounds
        t_start = t_span[1] + (interval_idx - 1) * interval_length
        t_end = t_start + interval_length

        # concentrations before simulation start are zero and can be skipped from integration
        (t_end <= sol.t[begin]) && continue
        t_start = max(t_start, sol.t[begin])
        
        # Find time indices, handle edge cases
        start_idx = findfirst(t -> t >= t_start, sol.t)
        end_idx = findfirst(t -> t >= t_end, sol.t)
        @assert !isnothing(t_start)
        end_idx = something(end_idx, length(sol.t)+1)
        end_idx -= 1
        
        # Integrate for each observation point
        for (obs_idx, (i, j)) in enumerate(obs_indices)
            grid_idx = (j-1)*nx + i
            integrated = 0.0
            
            # ADD initial trapezoid segment from t_start to first integrated interval
            if start_idx == 1
                @assert t_start == sol.t[start_idx]
            else
                c1, c2 = sol.u[start_idx-1][grid_idx], sol.u[start_idx][grid_idx]
                integrated += 0.5 * (c1 + c2) * (sol.t[start_idx] - t_start)
            end

            for t_idx in start_idx:end_idx
                dt = sol.t[t_idx+1] - sol.t[t_idx]
                c1, c2 = sol.u[t_idx][grid_idx], sol.u[t_idx+1][grid_idx]
                integrated += 0.5 * (c1 + c2) * dt
            end

            # SUBSTRACT final trapezoid segment from t_end to end of last integrated interval
            if end_idx == length(sol.t)
                @assert t_end == sol.t[end_idx]
            else
                c1, c2 = sol.u[end_idx][grid_idx], sol.u[end_idx+1][grid_idx]
                integrated -= 0.5 * (c1 + c2) * (sol.t[end_idx+1] - t_end)
            end

            # store values for each observation point together
            y[(obs_idx-1)*n_intervals_local + interval_idx] = integrated
        end
    end
    
    # never return 0 measurements to avoid numerical issues
    min_val = MIN_MEASURED_VALUE_PER_TIME_UNIT * interval_length
    return max.(min_val, y)
end

"""
Generate reference observation data with known parameters.
Uses interval_length=10.0 to measure over the entire simulation period.
"""
function _generate_reference_data()
    sol = diffusion_simulation(x_ref)
    observations = extract_measurements(sol; interval_length=observation_interval)
    observations = normalize_observations(observations)
    return observations
end

# Generate synthetic observation data
const z_obs = _generate_reference_data()

"""
Parameter bounds: [x_s_min, y_s_min, t_s_min], [x_s_max, y_s_max, t_s_max]
"""
###
# _get_bounds() = ([-5.0, -5.0, -10.0], [5.0, 5.0, -0.5])
_get_bounds() = ([-1., -1., -1.], [1., 1., 1.])

"""
Prior mean based on typical parameter values.
"""
function _get_prior_mean(p::DiffusionProblem)
    ratio = Int(observation_interval / p.model_interval)
    μ = similar(z_obs, length(z_obs) * ratio)
    for i in eachindex(z_obs)
        idx = range((i-1)*ratio + 1, i*ratio)
        μ[idx] .= log(z_obs[i] / ratio)
    end
    return μ
end

"""
Truncated prior distribution for parameters.
"""
function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end

"""
Prior distribution:
- x_s ~ Uniform (source x-location can vary across domain)
- y_s ~ Uniform (source y-location can vary across domain)
- t_s ~ Normal (source activation time in the past)
Note: A (amplitude) is now fixed at 1.0
"""
function _get_x_prior()
    # the priors are truncated to the domain bounds automatically
    dists = [
        Uniform(-5.0, 5.0),   # x_s: source x-location
        Uniform(-5.0, 5.0),   # y_s: source y-location
        Normal(-2.0, 5.0),    # t_s: source activation time (closer to 0 more probable)
    ]
    dists = normalize_inputs(dists)
    return Product(dists)
end

end # module DiffusionModule

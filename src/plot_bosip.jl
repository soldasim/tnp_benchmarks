
using CairoMakie
using Distributions

function plot_results(bosip::BosipProblem;
	grid_size::Int = 50,
	resolution::Int = 900,
	colormap::Symbol = :viridis,
	display::Bool = true,
)
	x_dim = BOSIP.x_dim(bosip)
	y_dim = BOSIP.y_dim(bosip)
	@assert x_dim == 2 "plot_results currently supports 2D inputs"
	@assert y_dim == 1 "plot_results currently supports 1D outputs"

	lb, ub = bosip.problem.domain.bounds
	xs_1 = range(lb[1], ub[1]; length=grid_size)
	xs_2 = range(lb[2], ub[2]; length=grid_size)

	X = Matrix{Float64}(undef, 2, grid_size * grid_size)
	idx = 1
	for x2 in xs_2, x1 in xs_1
		X[:, idx] .= (x1, x2)
		idx += 1
	end

	model_post = BOSS.model_posterior(bosip.problem)
	y_model = BOSS.mean(model_post, X)
	y_true = hcat(bosip.problem.f.(eachcol(X))...)

	Z_true = reshape(y_true[1, :], length(xs_1), length(xs_2))
	Z_model = reshape(y_model, length(xs_1), length(xs_2))

	function true_logpost(x::AbstractVector{<:Real})
		y = bosip.problem.f(x)
		ll = BOSIP.loglike(bosip.likelihood, y, x)
		lp = logpdf(bosip.x_prior, x)
		return ll + lp
	end

	logpost_true = true_logpost.(eachcol(X))
	logpost_est = BOSIP.log_posterior_mean(bosip).(eachcol(X))

	# Normalize and exponentiate log-posteriors to get posterior densities
	function _normalized_exp(log_vals::AbstractVector{<:Real})
		log_vals = log_vals .- maximum(log_vals)
		return exp.(log_vals)
	end

	Z_post_true = reshape(_normalized_exp(logpost_true), length(xs_1), length(xs_2))
	Z_post_est = reshape(_normalized_exp(logpost_est), length(xs_1), length(xs_2))

	# Get the experiment data points
	data_X = bosip.problem.data.X

	fig = Figure(size=(resolution, resolution))

	ax_true = Axis(fig[1, 1]; title="True simulator", xlabel="x1", ylabel="x2")
	heatmap!(ax_true, xs_1, xs_2, Z_true; colormap)
	scatter!(ax_true, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	ax_model = Axis(fig[1, 2]; title="Model prediction", xlabel="x1", ylabel="x2")
	heatmap!(ax_model, xs_1, xs_2, Z_model; colormap)
	scatter!(ax_model, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	ax_post_true = Axis(fig[2, 1]; title="True posterior", xlabel="x1", ylabel="x2")
	heatmap!(ax_post_true, xs_1, xs_2, Z_post_true; colormap)
	scatter!(ax_post_true, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	ax_post_est = Axis(fig[2, 2]; title="Estimated posterior", xlabel="x1", ylabel="x2")
	heatmap!(ax_post_est, xs_1, xs_2, Z_post_est; colormap)
	scatter!(ax_post_est, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	display && CairoMakie.display(fig)
	return fig
end

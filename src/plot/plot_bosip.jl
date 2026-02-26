
# For BI sampled model posteriors, compute the mean and std of a Gaussian mixture.
function BOSS.mean_and_std(models::Vector{<:ModelPosterior}, x::AbstractVector{<:Real})
    vals = BOSS.mean_and_var.(models, Ref(x))
    mus = getindex.(vals, 1)
    vars = getindex.(vals, 2)

    m = mean(mus)
    v = mean(vars) + var(mus)
    s = sqrt.(v)
    return m, s
end
function BOSS.mean_and_std(models::Vector{<:ModelPosterior}, X::AbstractMatrix{<:Real})
    vals = BOSS.mean_and_std.(Ref(models), eachcol(X))
    mus = getindex.(vals, 1)
    stds = getindex.(vals, 2)
    return hcat(mus...), hcat(stds...)
end

function plot_results(bosip::BosipProblem;
	grid_size::Int = 50,
	resolution::Int = 900,
	colormap::Symbol = :viridis,
	display::Bool = true,
	show_contour::Bool = false,
)
	x_dim = BOSIP.x_dim(bosip)
	y_dim = BOSIP.y_dim(bosip)
	@assert x_dim == 2 "plot_results currently supports 2D inputs"

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
	
    print("Computing model predictions on grid ...")
    y_model, y_model_std = BOSS.mean_and_std(model_post, X)
	println(" done")

    print("Computing true simulator values on grid ...")
    y_true = hcat(bosip.problem.f.(eachcol(X))...)
    println(" done")

	# Reshape each y dimension
	Z_true = [reshape(y_true[i, :], length(xs_1), length(xs_2)) for i in 1:y_dim]
	Z_model = [reshape(y_model[i, :], length(xs_1), length(xs_2)) for i in 1:y_dim]
	Z_model_std = [reshape(y_model_std[i, :], length(xs_1), length(xs_2)) for i in 1:y_dim]

	function true_logpost(x::AbstractVector{<:Real})
		y = bosip.problem.f(x)
		ll = BOSIP.loglike(bosip.likelihood, y, x)
		lp = logpdf(bosip.x_prior, x)
		return ll + lp
	end

    print("Computing true posterior on grid ...")
	logpost_true = true_logpost.(eachcol(X))
    println(" done")

    print("Computing estimated posterior on grid ...")
	logpost_est = BOSIP.log_posterior_mean(bosip)(X)
	logpost_var = BOSIP.log_posterior_variance(bosip)(X)
    println(" done")

	# Normalize and exponentiate log-posteriors to get posterior densities
	function _normalized_exp(log_vals::AbstractVector{<:Real})
		log_vals = log_vals .- maximum(log_vals)
		return exp.(log_vals)
	end

	Z_post_true = reshape(_normalized_exp(logpost_true), length(xs_1), length(xs_2))
	Z_post_est = reshape(_normalized_exp(logpost_est), length(xs_1), length(xs_2))
	Z_post_std = reshape(sqrt.(_normalized_exp(logpost_var)), length(xs_1), length(xs_2))

	# Get the experiment data points
	data_X = bosip.problem.data.X

	# Get the z_obs value for contour plotting
	z_obs = nothing
	if show_contour
		z_obs = bosip.likelihood.z_obs
	end

	# Create figure with y_dim + 1 rows (one row per y dimension + posterior row)
	num_rows = y_dim + 1
	fig = Figure(size=(resolution * 7 ÷ 4, resolution * num_rows ÷ 2))

	# Plot each y dimension as a row
	for i in 1:y_dim
		row = i
		y_label = y_dim > 1 ? "y$i" : "y"
		
		ax_true = Axis(fig[row, 1]; title="True simulator ($y_label)", xlabel="x1", ylabel="x2")
		hm_true = heatmap!(ax_true, xs_1, xs_2, Z_true[i]; colormap)
		scatter!(ax_true, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)
		if show_contour
			contour!(ax_true, xs_1, xs_2, Z_model[i]; levels=[z_obs[i]], color=:red, linewidth=2)
		end
		Colorbar(fig[row, 2], hm_true)

		ax_model = Axis(fig[row, 3]; title="Model prediction ($y_label)", xlabel="x1", ylabel="x2")
		hm_model = heatmap!(ax_model, xs_1, xs_2, Z_model[i]; colormap)
		scatter!(ax_model, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)
		if show_contour
			contour!(ax_model, xs_1, xs_2, Z_model[i]; levels=[z_obs[i]], color=:red, linewidth=2)
		end
		Colorbar(fig[row, 4], hm_model)

		ax_model_std = Axis(fig[row, 5]; title="Model std ($y_label)", xlabel="x1", ylabel="x2")
		hm_model_std = heatmap!(ax_model_std, xs_1, xs_2, Z_model_std[i]; colormap)
		scatter!(ax_model_std, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)
		if show_contour
			contour!(ax_model_std, xs_1, xs_2, Z_model[i]; levels=[z_obs[i]], color=:red, linewidth=2)
		end
		Colorbar(fig[row, 6], hm_model_std)
	end

	# Plot posterior row (last row)
	post_row = y_dim + 1
	
	ax_post_true = Axis(fig[post_row, 1]; title="True posterior", xlabel="x1", ylabel="x2")
	heatmap!(ax_post_true, xs_1, xs_2, Z_post_true; colormap)
	scatter!(ax_post_true, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	ax_post_est = Axis(fig[post_row, 3]; title="Estimated posterior", xlabel="x1", ylabel="x2")
	heatmap!(ax_post_est, xs_1, xs_2, Z_post_est; colormap)
	scatter!(ax_post_est, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	ax_post_std = Axis(fig[post_row, 5]; title="Posterior std", xlabel="x1", ylabel="x2")
	heatmap!(ax_post_std, xs_1, xs_2, Z_post_std; colormap)
	scatter!(ax_post_std, data_X[1, :], data_X[2, :]; color=:white, markersize=10, strokecolor=:black, strokewidth=2)

	display && CairoMakie.display(fig)
	return fig
end

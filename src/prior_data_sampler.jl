using PythonCall

sample_prior(prior::Distribution) = rand(prior)
sample_prior(prior::Tuple{<:Real, <:Real}) = rand(Uniform(prior...))
sample_prior(prior::Function) = prior()
sample_prior(prior::Real) = prior

function default_data_sampler(;
	batch_size = 32,
	num_total_points_range = (64, 256),
	x_range = (-1.0, 1.0),
	kernel = GaussianKernel(),
	length_scale_prior = (0.1, 1.0),
	amplitude_prior = (0.1, 1.0),
	noise_std_prior = 1e-6,
	x_dim = 2,
	y_dim = 1,
)
	np = pyimport("numpy")
	
	function sample_batch()
		x_min, x_max = x_range

		num_total_points = rand(num_total_points_range[1]:num_total_points_range[2])
		num_context = rand(1:num_total_points-1)
		indices = randperm(num_total_points)
		context_idx = indices[1:num_context]
		target_idx = indices[num_context+1:end]

		# Initialize with Python-friendly dimensions: (batch, num_points, dim)
		x = Array{Float64}(undef, batch_size, num_total_points, x_dim)
		y = Array{Float64}(undef, batch_size, num_total_points, y_dim)

		for b in 1:batch_size
			length_scale = [sample_prior(length_scale_prior) for _ in 1:x_dim]
			amplitude = sample_prior(amplitude_prior)
            noise_std = sample_prior(noise_std_prior)

			# Sample x points uniformly
			x[b, :, :] = rand(Uniform(x_min, x_max), num_total_points, x_dim)
			
			# Compute kernel matrix
			Xb = view(x, b, :, :)
			kernel_scaled = KernelFunctions.with_lengthscale(kernel, length_scale)
			K = amplitude^2 .* KernelFunctions.kernelmatrix(kernel_scaled, KernelFunctions.RowVecs(Xb))
			K = K + (noise_std^2) * I

			# Sample y values using GP
			samples = rand(MvNormal(zeros(num_total_points), Symmetric(K)), y_dim)
			y[b, :, :] = samples
		end

		# Extract context and target, convert directly to numpy
		context_x = np.asarray(x[:, context_idx, :])
		context_y = np.asarray(y[:, context_idx, :])
		target_x = np.asarray(x[:, target_idx, :])
		target_y = np.asarray(y[:, target_idx, :])

		return context_x, context_y, target_x, target_y
	end

	return sample_batch
end

function lhc_data_sampler(;
	batch_size = 32,
	context_num_points_range = (8, 128),
	target_num_points = 128,
	x_range = (-1.0, 1.0),
	kernel = GaussianKernel(),
	length_scale_prior = (0.1, 1.0),
	amplitude_prior = (0.1, 1.0),
	noise_std_prior = 1e-6,
	x_dim = 2,
	y_dim = 1,
    max_num_clusters = x_dim + 1,
    outlier_ratio_range = (0.05, 0.2), # outliers out of all context points
    relative_cluster_std_range = (0.02, 0.1),
)
	np = pyimport("numpy")

	function sample_lhc_points(num_points::Int)
		x_min, x_max = x_range
		width = (x_max - x_min) / num_points
		points = Array{Float64}(undef, num_points, x_dim)

		for d in 1:x_dim
			vals = x_min .+ ((0:num_points-1) .+ rand(num_points)) .* width
			points[:, d] = vals[randperm(num_points)]
		end

		return points
	end
	
	function sample_batch()
		x_min, x_max = x_range

		num_context = rand(context_num_points_range[1]:context_num_points_range[2])
		num_total_points = num_context + target_num_points

		# Initialize with Python-friendly dimensions: (batch, num_points, dim)
		x = Array{Float64}(undef, batch_size, num_total_points, x_dim)
		y = Array{Float64}(undef, batch_size, num_total_points, y_dim)

		for b in 1:batch_size
			length_scale = [sample_prior(length_scale_prior) for _ in 1:x_dim]
			amplitude = sample_prior(amplitude_prior)
            noise_std = sample_prior(noise_std_prior)

			# Sample x points with clustering
			# Randomly choose number of clusters (1-4)
			num_clusters = rand(1:max_num_clusters)
			
			# Sample cluster std first
            cluster_std = rand(Uniform(relative_cluster_std_range...), num_clusters) .* (x_max - x_min)
            
            # Sample cluster centers uniformly, ensuring at least 2*std away from edges
            cluster_centers = Array{Float64}(undef, num_clusters, x_dim)
            for c in 1:num_clusters
                for d in 1:x_dim
                    min_center = x_min + 2 * cluster_std[c]
                    max_center = x_max - 2 * cluster_std[c]
                    cluster_centers[c, d] = rand(Uniform(min_center, max_center))
                end
            end

			# Decide how many points are outliers (context only)
			outlier_fraction = rand(Uniform(outlier_ratio_range...))
			num_outliers = round(Int, outlier_fraction * num_context)
			num_clustered = num_context - num_outliers
			
			# Sample outlier points uniformly
			if num_outliers > 0
				x[b, 1:num_outliers, :] = rand(Uniform(x_min, x_max), num_outliers, x_dim)
			end
			
			# Sample clustered points from Gaussians around cluster centers
			for i in 1:num_clustered
				# Randomly assign to a cluster
				cluster_idx = rand(1:num_clusters)
				for d in 1:x_dim
					x[b, num_outliers + i, d] = clamp(
						cluster_centers[cluster_idx, d] + randn() * cluster_std[cluster_idx],
						x_min, x_max
					)
				end
			end

			# Sample target points using a Latin hypercube grid
			x[b, (num_context + 1):num_total_points, :] = sample_lhc_points(target_num_points)
			
			# Compute kernel matrix
			Xb = view(x, b, :, :)
			kernel_scaled = KernelFunctions.with_lengthscale(kernel, length_scale)
			K = amplitude^2 .* KernelFunctions.kernelmatrix(kernel_scaled, KernelFunctions.RowVecs(Xb))
			K = K + (noise_std^2) * I

			# Sample y values using GP
			samples = rand(MvNormal(zeros(num_total_points), Symmetric(K)), y_dim)
			y[b, :, :] = samples
		end

		# Extract context and target, convert directly to numpy
		context_x = np.asarray(x[:, 1:num_context, :])
		context_y = np.asarray(y[:, 1:num_context, :])
		target_x = np.asarray(x[:, (num_context + 1):num_total_points, :])
		target_y = np.asarray(y[:, (num_context + 1):num_total_points, :])

		return context_x, context_y, target_x, target_y
	end

	return sample_batch
end

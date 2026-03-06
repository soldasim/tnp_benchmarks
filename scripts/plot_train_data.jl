using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip

using Random
using CairoMakie
using PythonCall
using PyTNP

# Random.seed!(555)

include(srcdir("include.jl"))

# TODO
# sampler = default_data_sampler()
sampler = lhc_data_sampler()
# sampler = lhc_context_data_sampler()
# sampler = context_data_sampler()

# gp_sampler = pyimport("gp_sampler")
# sampler = gp_sampler.make_gp_sampler(;
#     batch_size = 32,
#     num_total_points_range = (64, 256),
#     x_range = (-1.0, 1.0),
#     kernel_length_scale_prior = (0.1, 2.0),
#     kernel_std_prior = (0.1, 1.0),
#     noise_std = 1e-8,
#     x_dim = 2,
#     y_dim = 1,
# )

xc, yc, xt, yt = sampler()

# Convert numpy arrays to Julia arrays
# xc, xt shape: (batch_size, num_points, x_dim)
xc_ = pyconvert(Array, xc)
xt_ = pyconvert(Array, xt)
yc_ = pyconvert(Array, yc)
yt_ = pyconvert(Array, yt)

# Create a 3x3 grid of plots with larger fonts
fig = Figure(size=(1400, 1400), fontsize=16)

for i in 1:3
    for j in 1:3
        batch_idx = (i - 1) * 3 + j
        
        # Extract batch
        xc_batch = xc_[batch_idx, :, :]
        xt_batch = xt_[batch_idx, :, :]
        yc_batch = yc_[batch_idx, :, :]
        yt_batch = yt_[batch_idx, :, :]

        yc_mean = mean(yc_batch, dims=1)
        yc_std = std(yc_batch, dims=1)
        yt_mean = mean(yt_batch, dims=1)
        yt_std = std(yt_batch, dims=1)
        @show yc_mean, yc_std
        @show yt_mean, yt_std
        
        # Create axis in grid position
        ax = Axis(fig[i, j], xlabel="x₁", ylabel="x₂", title="Batch $batch_idx",
                  xlabelsize=18, ylabelsize=18, titlesize=20)
        
        # Plot context and target points
        scatter!(ax, xc_batch[:, 1], xc_batch[:, 2], label="context", color=:blue, markersize=10, alpha=0.5)
        scatter!(ax, xt_batch[:, 1], xt_batch[:, 2], label="target", color=:red, markersize=10, alpha=0.5)
        
        # Only show legend on first plot
        if i == 1 && j == 1
            axislegend(ax, position=:lt, labelsize=16)
        end
    end
end

# display(fig)
save(plotsdir("train_samples.png"), fig)

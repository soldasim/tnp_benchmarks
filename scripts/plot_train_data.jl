using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip

using Random
using CairoMakie
# Random.seed!(555)

include(srcdir("include.jl"))

# TODO
# sampler = prior_data_sampler()
# sampler = prior_context_data_sampler()
sampler = lhc_data_sampler()

xc, yc, xt, yt = sampler()

# Convert numpy arrays to Julia arrays
# xc, xt shape: (batch_size, num_points, x_dim)
xc_all = pyconvert(Array, xc)
xt_all = pyconvert(Array, xt)

# Create a 3x3 grid of plots with larger fonts
fig = Figure(size=(1400, 1400), fontsize=16)

for i in 1:3
    for j in 1:3
        batch_idx = (i - 1) * 3 + j
        
        # Extract batch
        xc_batch = xc_all[batch_idx, :, :]
        xt_batch = xt_all[batch_idx, :, :]
        
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

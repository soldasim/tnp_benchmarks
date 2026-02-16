using DrWatson
@quickactivate "tnp_benchmarks"
include(srcdir("tag.jl"))

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip
using PythonCall
using PyTNP

# INPUTS
const DIM_MODEL = parse(Int, ARGS[1]) # e.g. 64
const DIM_FEEDFORWARD = 2 * DIM_MODEL
println("Starting TNP training with dim_model = $DIM_MODEL, dim_feedforward = $DIM_FEEDFORWARD")

params = Dict(
    # Universe Settings
    :x_dim => 2,
    :y_dim => 1,

    # Model Settings
    # :dim_model => 128,
    :dim_model => DIM_MODEL,
    :embedder_depth => 4,
    :predictor_depth => 2,
    :num_heads => 8,
    :encoder_depth => 6,
    # :dim_feedforward => 128,
    :dim_feedforward => DIM_FEEDFORWARD,
    :dropout => 0.0,
    :device => "mps",

    # Prior Data Settings
    :batch_size => 32,
    :num_total_points_range => (64, 256),
    :x_range => (-1.0, 1.0),
    :kernel_length_scale_prior => (0.1, 2.0),
    :kernel_amplitude_prior => (0.1, 1.0),
    :noise_std => 1e-8,

    # Training Settings
    :num_iterations => 10_000,
)

# Initialize TNP
model = init_model(;
    x_dim = params[:x_dim],
    y_dim = params[:y_dim],
    dim_model = params[:dim_model],
    embedder_depth = params[:embedder_depth],
    predictor_depth = params[:predictor_depth],
    num_heads = params[:num_heads],
    encoder_depth = params[:encoder_depth],
    dim_feedforward = params[:dim_feedforward],
    dropout = params[:dropout],
    device = params[:device],
)

# Initialize Data Sampler
gp_sampler = pyimport("gp_sampler")
sample_fn = gp_sampler.make_gp_sampler(;
    batch_size = params[:batch_size],
    num_total_points_range = params[:num_total_points_range],
    x_range = params[:x_range],
    kernel_length_scale_prior = params[:kernel_length_scale_prior],
    kernel_std_prior = params[:kernel_amplitude_prior],
    noise_std = params[:noise_std],
    x_dim = params[:x_dim],
    y_dim = params[:y_dim],
)

# Train TNP
_, losses, avg_losses = train_model!(model, sample_fn;
    num_iterations = params[:num_iterations],
    print_freq = 100,
    model.device,
)

# Save parameters and results
run_name = savename(params)
params_file = run_name * "_params.jld2"
model_file = run_name * "_model.pt"
params_path = joinpath(datadir("params"), params_file)
model_path = joinpath(datadir("models"), model_file)

@tag_with_deps! params
params[:model_path] = model_path

wsave(params_path, params)
save_model(model, model_path)

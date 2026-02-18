using DrWatson
@quickactivate "tnp_benchmarks"
include(srcdir("tag.jl"))

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip
using PythonCall
using PyTNP


## Inputs
const DIM_MODEL = parse(Int, ARGS[1]) # e.g. 64
const DIM_FEEDFORWARD = 2 * DIM_MODEL
const NUM_ITERATIONS = parse(Int, ARGS[2]) # e.g. 100_000

println("Training TNP with dim_model = $DIM_MODEL, num_iterations = $NUM_ITERATIONS")

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
    :device => "cuda",

    # Prior Data Settings
    :batch_size => 32,
    :num_total_points_range => (64, 256),
    :x_range => (-1.0, 1.0),
    :kernel_length_scale_prior => (0.1, 2.0),
    :kernel_amplitude_prior => (0.1, 1.0),
    :noise_std => 1e-8,

    # Training Settings
    :num_iterations => NUM_ITERATIONS,
)


## Initialize TNP
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


## Initialize Data Sampler
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


## Train TNP
@info "Training TNP ..."

lrs, losses, avg_losses = train_model!(model, sample_fn;
    num_iterations = params[:num_iterations],
    print_freq = 100,
    model.device,
    save_path = nothing,
)


## Save Results
@info "Saving results ..."

run_name = savename(params)
model_path = joinpath(datadir("train"), "model_" *run_name * ".pt")
data_path = joinpath(datadir("train"), "data_" * run_name * ".jld2")

data = Dict(
    :model_params => params,
    :lr => lrs,
    :loss_history => losses,
    :avg_loss_history => avg_losses,
    :model_path => model_path,
    :data_path => data_path,
)
@tag_with_deps! data

wsave(data_path, data)
save_model(model, model_path)

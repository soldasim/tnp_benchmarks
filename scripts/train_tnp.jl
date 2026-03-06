using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip
using PythonCall
using PyTNP

include(srcdir("include.jl"))

# TODO
subdir = "test"
# subdir = "beta"


## Inputs
const NUM_ITERATIONS = parse(Int, ARGS[1]) # e.g. 10_000
const BASE_MODEL = ARGS[2] # :default, :structured
const SAMPLE_FN = ARGS[3] # :default_data_sampler, :lhc_data_sampler

const ENCODER_DEPTH = parse(Int, ARGS[4]) # e.g. 6
const DIM_MODEL = parse(Int, ARGS[5]) # e.g. 128
const DIM_FEEDFORWARD = 2 * DIM_MODEL

const X_DIM = parse(Int, ARGS[6]) # e.g. 2

const NOTE = length(ARGS) >= 7 ? ARGS[7] : "test"

println("Training TNP with dim_model = $DIM_MODEL, num_iterations = $NUM_ITERATIONS")

params = Dict(
    :note => NOTE,

    # Universe Settings
    :x_dim => X_DIM,
    :y_dim => 1,

    # Model Implementation
    :base_model => Symbol(BASE_MODEL),

    # Model Settings
    # :dim_model => 128,
    :dim_model => DIM_MODEL,
    :embedder_depth => 4,
    :predictor_depth => 2,
    :num_heads => 8,
    :encoder_depth => ENCODER_DEPTH,
    # :dim_feedforward => 128,
    :dim_feedforward => DIM_FEEDFORWARD,
    :dropout => 0.0,
    :device => "cuda",

    # Prior Data Settings
    :sample_fn => Symbol(SAMPLE_FN),

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
    base_model = params[:base_model],
)


## Initialize Data Sampler
sample_fn = getfield(Main, params[:sample_fn])(; x_dim = params[:x_dim], y_dim = params[:y_dim])


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
@info "Run name: $run_name"

model_path = joinpath(datadir("train"), subdir, "model_" *run_name * ".pt")
data_path = joinpath(datadir("train"), subdir, "data_" * run_name * ".jld2")

data = Dict(
    :model_params => params,
    :lr => lrs,
    :loss_history => losses,
    :avg_loss_history => avg_losses,
    :model_path => model_path,
    :data_path => data_path,
)
@tag_with_deps! data storepatch=true

wsave(data_path, data)
save_model(model, model_path)

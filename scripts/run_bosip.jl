using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip
using PyTNP
using BOSIP, BOSS
using OptimizationPRIMA
using JLD2

using Random
Random.seed!(555)

include(srcdir("include.jl"))


## Select Toy Problem & Run Name
problem = ABProblem()
subdir = "default"


## Load TNP Model

const DIM_MODEL = parse(Int, ARGS[1]) # e.g. 64
const DIM_FEEDFORWARD = 2 * DIM_MODEL
const NUM_ITERATIONS = parse(Int, ARGS[2]) # e.g. 100_000

println("Running BOSIP with pre-trained TNP model with dim_model = $DIM_MODEL, num_iterations = $NUM_ITERATIONS")

model_params = Dict(
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

mode = StandardTNP()
# mode = KNNTNP(10)

tnp = load_model(model_params; mode)
model_params[:mode] = mode

function tnp_predict(x::AbstractVector{<:Real}, data::ExperimentData)
    μs, σs = PyTNP.predict(tnp, data.X, data.Y, hcat(x))
    return μs[:,1], σs[:,1]
end
function tnp_predict(X::AbstractMatrix{<:Real}, data::ExperimentData)
    μs, σs = PyTNP.predict(tnp, data.X, data.Y, X)
    return μs, σs
end
model = BlackboxModel(tnp_predict)


## Use GP Model

# model_params = Dict(
#     :mean => prior_mean(problem),
#     :kernel => BOSS.GaussianKernel(),
#     # :kernel => BOSS.Matern32Kernel(),
#     # :kernel => BOSS.Matern52Kernel(),
#     :lengthscale_priors => get_lengthscale_priors(problem),
#     :amplitude_priors => get_amplitude_priors(problem),
#     :noise_std_priors => get_noise_std_priors(problem),
#     # :noise_std_priors => [Uniform(0.1, 1.)],
# )

# model = GaussianProcess(;
#     mean = model_params[:mean],
#     kernel = model_params[:kernel],
#     lengthscale_priors = model_params[:lengthscale_priors],
#     amplitude_priors = model_params[:amplitude_priors],
#     noise_std_priors = model_params[:noise_std_priors],
# )


## Define BOSIP Problem

bosip_params = Dict(
    :problem => problem,
    :init_data => x_dim(problem) + 1,
    :max_data => 100, # TODO
)

acquisition = LogMaxVar()

f = simulator(problem)
X = rand(x_prior(problem), bosip_params[:init_data])
Y = hcat(f.(eachcol(X))...)
data = ExperimentData(X, Y)

bosip = BosipProblem(data;
    f,
    domain = domain(problem),
    acquisition,
    model,
    likelihood = likelihood(problem),
    x_prior = x_prior(problem),
)

## BOSIP Metric

xs = rand(bosip.x_prior, 20 * 10^x_dim(problem))
ws = exp.( (0.) .- logpdf.(Ref(bosip.x_prior), eachcol(xs)) )

struct NoSampler <: DistributionSampler end # dummy

metric = TVMetric(;
    grid = xs,
    ws = ws,
    true_logpost = true_logpost(problem),
)
metric_cb = MetricCallback(;
    reference = true_logpost(problem),
    logpost_estimator = log_posterior_mean,
    sampler = NoSampler(),
    sample_count = 2 * 10^x_dim(problem),
    metric,
)


## Run BOSIP
@info "Running BOSIP ..."

parallel = false # cannot evaluate the python model in parallel
model_fitter = OptimizationMAP(;
    algorithm = NEWUOA(),
    multistart = 24,
    parallel,
    rhoend = 1e-4,
)
acq_maximizer = OptimizationAM(;
    algorithm = BOBYQA(),
    multistart = 24,
    parallel,
    rhoend = 1e-4,
)

term_cond = DataLimit(bosip_params[:max_data])

bosip!(bosip; model_fitter, acq_maximizer, term_cond,
    options = BosipOptions(;
        callback = metric_cb,
    ),
)


## Save Results
@info "Saving results ..."

run_name = savename(model_params)
bosip_path = joinpath(datadir("bosip"), subdir, "bosip_" * run_name * ".jld2")
data_path = joinpath(datadir("bosip"), subdir, "data_" * run_name * ".jld2")

data = Dict(
    :model_params => model_params,
    :bosip_params => bosip_params,
    :score_history => metric_cb.score_history,
    :bosip_path => bosip_path,
    :data_path => data_path,
)
@tag_with_deps! data

wsave(data_path, data)
mkpath(dirname(bosip_path))
@save bosip_path bosip=bosip


## Plots
@info "Plotting ..."

plot_path = joinpath(plotsdir("bosip"), subdir, "bosip_" * run_name * ".png")

fig = plot_results(bosip)
mkpath(dirname(plot_path))
save(plot_path, fig)

using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip
using PyTNP
using BOSIP, BOSS
using OptimizationPRIMA
using JLD2
using Turing
using ADTypes

using Random
# TODO
Random.seed!(555)
# Random.seed!(1234)

include(srcdir("include.jl"))


# TODO
subdir = "test"
# subdir = "beta"


## Load TNP Model

# TODO
const NUM_ITERATIONS = parse(Int, ARGS[1]) # e.g. 10_000
const BASE_MODEL = ARGS[2] # :default, :structured
const SAMPLE_FN = ARGS[3] # :default_data_sampler, :lhc_data_sampler

const ENCODER_DEPTH = parse(Int, ARGS[4]) # e.g. 6
const DIM_MODEL = parse(Int, ARGS[5]) # e.g. 128
const DIM_FEEDFORWARD = 2 * DIM_MODEL

const X_DIM = parse(Int, ARGS[6]) # e.g. 2

const NOTE = ARGS[7] # e.g. "TEST", "log-softmax1", "no-softmax1"

const PROBLEM = ARGS[8] # e.g. ABProblem, SIRProblem, BananaProblem, BimodalProblem

## Select Toy Problem & Run Name
problem = getfield(Main, Symbol(PROBLEM))()

println("Running BOSIP with pre-trained TNP model with dim_model = $DIM_MODEL, num_iterations = $NUM_ITERATIONS")

model_params = Dict(
    # TODO
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

tnp = load_model(model_params;
    mode = DefaultMode(),
    # mode = KNNMode(10)
    base_model = Symbol(BASE_MODEL),
    dir = joinpath(datadir("train"), subdir),
)

function tnp_predict(x::AbstractVector{<:Real}, data::ExperimentData)
    y_dim = size(data.Y, 1)
    μ = Vector{Float64}(undef, y_dim)
    σ = Vector{Float64}(undef, y_dim)
    
    for d in 1:y_dim
        μs, σs = PyTNP.predict(tnp, data.X, data.Y[d:d, :], hcat(x))
        μ[d] = μs[1]
        σ[d] = σs[1]
    end
    
    return μ, σ
end
function tnp_predict(X::AbstractMatrix{<:Real}, data::ExperimentData)
    y_dim = size(data.Y, 1)
    n_points = size(X, 2)
    μ = Matrix{Float64}(undef, y_dim, n_points)
    σ = Matrix{Float64}(undef, y_dim, n_points)
    
    for d in 1:y_dim
        μs, σs = PyTNP.predict(tnp, data.X, data.Y[d:d, :], X)
        μ[d, :] = μs
        σ[d, :] = σs
    end
    
    return μ, σ
end
model = BlackboxModel(tnp_predict)


## Define BOSIP Problem

bosip_params = Dict(
    :problem => problem,
    :init_data => x_dim(problem) + 1,
    :max_data => 50, # TODO
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


## TV Metric

xs = rand(bosip.x_prior, 20 * 10^x_dim(problem))
ws = exp.( (0.) .- logpdf.(Ref(bosip.x_prior), eachcol(xs)) )

metric = TVMetric(;
    grid = xs,
    ws = ws,
    true_logpost = true_logpost(problem),
)

struct NoSampler <: DistributionSampler end # dummy
sampler = NoSampler()

metric_cb = MetricCallback(;
    reference = true_logpost(problem),
    logpost_estimator = log_posterior_mean,
    sampler,
    sample_count = 2 * 10^x_dim(problem),
    metric,
)


## Run BOSIP
@info "Running BOSIP ..."

model_fitter = OptimizationMAP(;
    algorithm = NEWUOA(),
    multistart = 24,
    parallel = false,
    rhoend = 1e-4,
)

acq_maximizer = OptimizationAM(;
    algorithm = BOBYQA(),
    multistart = 24,
    parallel = false,
    rhoend = 1e-4,
)

term_cond = DataLimit(bosip_params[:max_data])

bosip!(bosip; model_fitter, acq_maximizer, term_cond,
    options = BosipOptions(;
        callback = metric_cb, # TODO
    ),
)


## Save Results
subdir = joinpath(subdir, PROBLEM)

@info "Saving results ..."
run_name = savename(model_params)
@info "Run name: $run_name"

bosip_path = joinpath(datadir("bosip"), subdir, "bosip_" * run_name * ".jld2")
data_path = joinpath(datadir("bosip"), subdir, "data_" * run_name * ".jld2")

data = Dict(
    :model_params => model_params,
    :bosip_params => bosip_params,
    :score_history => metric_cb.score_history,
    :bosip_path => bosip_path,
    :data_path => data_path,
)
@tag_with_deps! data storepatch=true

wsave(data_path, data)
mkpath(dirname(bosip_path))
@save bosip_path bosip=bosip


## Plots
@info "Plotting ..."

plot_path = joinpath(plotsdir("bosip"), subdir, "bosip_" * run_name * ".png")

fig = plot_results(bosip)
mkpath(dirname(plot_path))
save(plot_path, fig)

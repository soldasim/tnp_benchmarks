using DrWatson
@quickactivate "tnp_benchmarks"

ENV["JULIA_PYTHONCALL_EXE"] = "/Users/soldasim/Documents/julia-pkg/PyTNP.jl/venv/bin/python"
using PyTNP
using BOSIP, BOSS
using OptimizationPRIMA

include(srcdir("include.jl"))


## Load Model

model_size = 64
tnp = load_model(model_size)

function tnp_predict(x::AbstractVector{<:Real}, data::ExperimentData)
    μs, σs = PyTNP.predict(tnp, data.X, data.Y, hcat(x))
    return μs[:,1], σs[:,1]
end
function tnp_predict(X::AbstractMatrix{<:Real}, data::ExperimentData)
    μs, σs = PyTNP.predict(tnp, data.X, data.Y, X)
    return μs, σs
end
model = BlackboxModel(tnp_predict)


## Define BOSIP Problem

problem = ABProblem()
acquisition = LogMaxVar()

init_data = x_dim(problem) + 1
# init_data = 50
term_cond = DataLimit(50)
# term_cond = IterLimit(50)

f = simulator(problem)
X = rand(x_prior(problem), init_data)
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

bosip!(bosip; model_fitter, acq_maximizer, term_cond,
    options = BosipOptions(;
        callback = metric_cb,
    ),
)


## Plot Results

fig = plot_results(bosip)

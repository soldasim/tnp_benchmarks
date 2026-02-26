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
Random.seed!(555)

include(srcdir("include.jl"))


## Select Toy Problem & Run Name
problem = ABProblem()

# TODO
# subdir = "test"
subdir = "alpha"

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
# model_params[:kernel_name] = string(typeof(model_params[:kernel]).name.name)

model_params = Dict(
    :note => "bayesian-inference",
    :mean => nothing,
    :kernel => BOSS.GaussianKernel(),
    :lengthscale_priors => fill(Product(fill(Uniform(0.1, 1.0), 2)), y_dim(problem)),
    :amplitude_priors => fill(Uniform(0.1, 1.0), y_dim(problem)),
    :noise_std_priors => fill(Dirac(1e-6), y_dim(problem)),
)
model_params[:kernel_name] = string(typeof(model_params[:kernel]).name.name)

model = GaussianProcess(;
    mean = model_params[:mean],
    kernel = model_params[:kernel],
    lengthscale_priors = model_params[:lengthscale_priors],
    amplitude_priors = model_params[:amplitude_priors],
    noise_std_priors = model_params[:noise_std_priors],
)


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

# TODO
# model_fitter = OptimizationMAP(;
#     algorithm = NEWUOA(),
#     multistart = 24,
#     parallel = false,
#     rhoend = 1e-4,
# )
model_fitter = TuringBI(;
    sampler = NUTS(),
    warmup = 400,
    samples_in_chain = 20,
    chain_count = 8,
    leap_size = 5,
    parallel = true,
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
        callback = metric_cb,
    ),
)


## Save Results
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

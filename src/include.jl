ENV["JULIA_PYTHONCALL_EXE"] = read(`which python`, String) |> strip

using Pkg
using DrWatson
using JLD2
using CairoMakie
using Distributions
using Bijectors
using BOSS, BOSIP
using Random
using LinearAlgebra
using KernelFunctions

include("tag.jl")
include("load_model.jl")
include("utils/include.jl")
include("abstract_problem.jl")
include("param_priors.jl")
include("problems/include.jl")
include("plot/include.jl")
include("prior_data_sampler.jl")

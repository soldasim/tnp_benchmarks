using Pkg
using DrWatson
using JLD2
using CairoMakie
using Distributions
using Bijectors
using BOSS, BOSIP

include("tag.jl")
include("utils/include.jl")
include("abstract_problem.jl")
include("param_priors.jl")
include("problems/include.jl")
include("plot/include.jl")

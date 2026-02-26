
function load_model(params::Dict;
    info = true,
    dir=datadir("train"),
    prefix="model_",
    mode = DefaultMode(),   # DefaultMode(), KNNMode(k)
    base_model = :default,  # :default, :structured
)
    run_name = savename(params)
    
    model_path = joinpath(dir, prefix * run_name * ".pt")
    
    !isfile(model_path) && error("Model with the given parameters not found at $(model_path)")
    
    tnp = PyTNP.load_model(model_path; mode, base_model)
    info && println("Loaded model from $(model_path)")
    return tnp
end

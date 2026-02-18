
function load_model(params::Dict; info = true, mode = StandardTNP())
    run_name = savename(params)
    
    models_dir = datadir("train")
    model_path = joinpath(models_dir, "model_" * run_name * ".pt")
    
    !isfile(model_path) && error("Model with the given parameters not found at $(model_path)")
    
    tnp = PyTNP.load_model(model_path; mode)
    info && println("Loaded model from $(model_path)")
    return tnp
end

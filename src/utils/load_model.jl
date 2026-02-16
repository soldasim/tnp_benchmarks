
function load_model(model_size::Int)
	models_dir = datadir("models")
	token = "dim_model=$(model_size)"
	matches = filter(name -> occursin(token, name) && endswith(name, "_model.pt"), readdir(models_dir))

	isempty(matches) && error("No model file found with $(token) in $(models_dir)")
	length(matches) > 1 && error("Multiple model files found with $(token) in $(models_dir): $(matches)")

	model_path = joinpath(models_dir, only(matches))
	return PyTNP.load_model(model_path)
end

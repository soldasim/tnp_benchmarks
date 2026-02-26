using CairoMakie
using DrWatson
using JLD2
using Colors

function plot_score(subdir::String; num_iterations::Union{Int, Nothing} = nothing, savepath = plotsdir("bosip", subdir, "scores.png"), display::Bool = true)
	data_dir = datadir("bosip", subdir)
	isdir(data_dir) || error("Data directory not found: $(data_dir)")

	# Find all data files
	data_files = filter(name -> startswith(name, "data_") && endswith(name, ".jld2"), readdir(data_dir))
	isempty(data_files) && error("No data files found in $(data_dir)")

	entries = NamedTuple[]
	seen_labels = Set()
	
	for file in data_files
		# Extract dim_model, num_iterations, kernel_name, and note from filename
		dim_model_match = match(r"_dim_model=([^_]+)_", file)
		num_iter_match = match(r"_num_iterations=([^_]+)_", file)
		kernel_match = match(r"_kernel_name=([^_\.]+)(?:_|\.jld2)", file)
		note_match = match(r"_note=([^_\.]+)(?:_|\.jld2)", file)
		
		is_gp = kernel_match !== nothing
		num_iter = nothing
		dim_model = nothing
		
		if !is_gp
			if num_iter_match === nothing
				error("Missing num_iterations in filename: $(file). Expected pattern _num_iterations=..._")
			end
			num_iter_str = num_iter_match.captures[1]
			num_iter = try
				parse(Int, num_iter_str)
			catch
				error("Invalid num_iterations value in filename: $(file). Expected integer, got $(num_iter_str)")
			end
			
			if dim_model_match === nothing
				continue  # Skip non-GP files without dim_model
			end
			dim_model_str = dim_model_match.captures[1]
			dim_model = try
				parse(Int, dim_model_str)
			catch
				error("Invalid dim_model value in filename: $(file). Expected integer, got $(dim_model_str)")
			end
		end
		
		# Create label
		if is_gp
			kernel_name = kernel_match.captures[1]
			label = "kernel_name=$(kernel_name)"
		else
			label = "dim_model=$(dim_model), num_iterations=$(num_iter)"
		end
		
		if note_match !== nothing
			note = note_match.captures[1]
			label = label * ", note=$(note)"
		end
		
		if label in seen_labels
			error("Duplicate label $(label) found in $(file). Please ensure all plotted labels are unique.")
		end
		push!(seen_labels, label)
		
		# Load the score history
		path = joinpath(data_dir, file)
        score_history = load(path, "score_history")
		
		push!(entries, (label = label, is_gp = is_gp, dim_model = dim_model, num_iter = num_iter, score = collect(score_history), file = file))
	end
	
	isempty(entries) && error("No entries loaded for subdir=$subdir" * (num_iterations !== nothing ? " with num_iterations=$num_iterations" : ""))
	
	sort!(entries; by = e -> (e.is_gp, e.dim_model === nothing ? Inf : e.dim_model))

	# Generate a large colorset using distinguishable colors
	num_colors = length(entries)
	colors = distinguishable_colors(num_colors, [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)

	# Dynamically set figure height based on number of entries
	legend_height = max(50, length(entries) * 25)
	fig_height = 600 + legend_height
	fig = Figure(size = (800, fig_height))
	ax = Axis(
		fig[1, 1];
		xlabel = "Iteration",
		ylabel = "Score",
		title = "Score history - $subdir" * (num_iterations !== nothing ? " (num_iterations=$num_iterations)" : ""),
		xscale = log10,
		yscale = identity,
	)

	for (idx, entry) in enumerate(entries)
		xs = 1:length(entry.score)
		if entry.is_gp
			linestyle = :dash
		elseif entry.num_iter == 10000
			linestyle = :solid
		elseif entry.num_iter == 100000
			linestyle = :dot
		else
			error("Unsupported num_iterations value: $(entry.num_iter). Expected 10000 or 100000.")
		end
		lines!(ax, xs, entry.score; label = entry.label, linestyle = linestyle, color = colors[idx])
	end

	# Create legend below the plot
	Legend(fig[2, 1], ax)
	rowsize!(fig.layout, 2, legend_height)
	display && CairoMakie.display(fig)
	isnothing(savepath) || (mkpath(dirname(savepath)); save(savepath, fig))
	return fig
end

function plot_final_score(subdir::String; num_iterations::Union{Int, Nothing} = nothing, savepath = plotsdir("bosip", subdir, "final_scores.png"), display::Bool = true)
	data_dir = datadir("bosip", subdir)
	isdir(data_dir) || error("Data directory not found: $(data_dir)")

	# Find all data files
	data_files = filter(name -> startswith(name, "data_") && endswith(name, ".jld2"), readdir(data_dir))
	isempty(data_files) && error("No data files found in $(data_dir)")

	entries = NamedTuple[]
	seen_labels = Set()
	
	for file in data_files
		# Extract dim_model, num_iterations, kernel_name, and note from filename
		dim_model_match = match(r"_dim_model=([^_]+)_", file)
		num_iter_match = match(r"_num_iterations=([^_]+)_", file)
		kernel_match = match(r"_kernel_name=([^_\.]+)(?:_|\.jld2)", file)
		note_match = match(r"_note=([^_\.]+)(?:_|\.jld2)", file)
		
		is_gp = kernel_match !== nothing
		num_iter = nothing
		dim_model = nothing
		
		if !is_gp
			if num_iter_match === nothing
				error("Missing num_iterations in filename: $(file). Expected pattern _num_iterations=..._")
			end
			num_iter_str = num_iter_match.captures[1]
			num_iter = try
				parse(Int, num_iter_str)
			catch
				error("Invalid num_iterations value in filename: $(file). Expected integer, got $(num_iter_str)")
			end
			
			if dim_model_match === nothing
				continue  # Skip non-GP files without dim_model
			end
			dim_model_str = dim_model_match.captures[1]
			dim_model = try
				parse(Int, dim_model_str)
			catch
				error("Invalid dim_model value in filename: $(file). Expected integer, got $(dim_model_str)")
			end
		end
		
		# Load the score history
		path = joinpath(data_dir, file)
		score_history = load(path, "score_history")
		
		final_score = last(score_history)
		
		# Create label
		if is_gp
			kernel_name = kernel_match.captures[1]
			label = "kernel_name=$(kernel_name)"
		else
			label = "dim_model=$(dim_model), num_iterations=$(num_iter)"
		end
		
		if note_match !== nothing
			note = note_match.captures[1]
			label = label * ", note=$(note)"
		end
		
		if label in seen_labels
			error("Duplicate label $(label) found in $(file). Please ensure all plotted labels are unique.")
		end
		push!(seen_labels, label)
		
		push!(entries, (label = label, is_gp = is_gp, dim_model = dim_model, num_iter = num_iter, final_score = final_score, file = file))
	end
	
	isempty(entries) && error("No entries loaded for subdir=$subdir" * (num_iterations !== nothing ? " with num_iterations=$num_iterations" : ""))
	
	sort!(entries; by = e -> (e.is_gp, e.dim_model === nothing ? Inf : e.dim_model))

	# Dynamically set figure height based on number of entries
	legend_height = max(50, length(entries) * 25)
	fig_height = 600 + legend_height
	fig = Figure(size = (800, fig_height))
	ax = Axis(
		fig[1, 1];
		xlabel = "Model size (dim_model)",
		ylabel = "Final score",
		title = "Final score - $subdir" * (num_iterations !== nothing ? " (num_iterations=$num_iterations)" : ""),
		xscale = log2,
		yscale = identity,
	)

	if !isempty(entries)
		# Generate a large colorset using distinguishable colors
		num_colors = length(entries)
		colors = distinguishable_colors(num_colors, [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)
		
		bosip_entries = filter(e -> !e.is_gp, entries)
		gp_entries = filter(e -> e.is_gp, entries)
		
		# Group BOSIP entries by num_iterations
		bosip_10k = filter(e -> e.num_iter == 10000, bosip_entries)
		bosip_100k = filter(e -> e.num_iter == 100000, bosip_entries)
		
		# Check for unsupported num_iterations
		for entry in bosip_entries
			if !(entry.num_iter in [10000, 100000])
				error("Unsupported num_iterations value: $(entry.num_iter). Expected 10000 or 100000.")
			end
		end
		
		if !isempty(bosip_10k)
			sizes_10k = [e.dim_model for e in bosip_10k]
			finals_10k = [e.final_score for e in bosip_10k]
			# Assign colors to 10k entries
			colors_10k = [colors[findfirst(x -> x.label == e.label && x.num_iter == 10000, entries)] for e in bosip_10k]
			scatter!(ax, sizes_10k, finals_10k; label = "BOSIP (10k)", marker = :circle, color = colors_10k)
		end
		
		if !isempty(bosip_100k)
			sizes_100k = [e.dim_model for e in bosip_100k]
			finals_100k = [e.final_score for e in bosip_100k]
			# Assign colors to 100k entries
			colors_100k = [colors[findfirst(x -> x.label == e.label && x.num_iter == 100000, entries)] for e in bosip_100k]
			scatter!(ax, sizes_100k, finals_100k; label = "BOSIP (100k)", marker = :rectangle, color = colors_100k)
		end
		
		# Assign colors to GP entries
		for entry in gp_entries
			color_idx = findfirst(x -> x.label == entry.label, entries)
			hlines!(ax, entry.final_score; label = entry.label, linestyle = :dash, color = colors[color_idx])
		end
	end

	# Create legend below the plot
	Legend(fig[2, 1], ax)
	rowsize!(fig.layout, 2, legend_height)
	display && CairoMakie.display(fig)
	isnothing(savepath) || (mkpath(dirname(savepath)); save(savepath, fig))
	return fig
end

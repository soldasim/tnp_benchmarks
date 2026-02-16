using DrWatson
using Pkg

"""
    @tag_with_deps! dict

Same as `DrWatson.@tag!` but also adds the git commit info of direct dependencies.

See `tag_with_deps` for more information on kwargs.
"""
macro tag_with_deps!(d,args...)
    args = Any[args...]
    # Keywords added after a ; are moved to the front of the expression
    # that is passed to the macro. So instead of getting the dict in d
    # an Expr is passed.
    if d isa Expr && d.head == :parameters
        length(args) > 0 || return :(throw(MethodError(@tag_with_deps!,$(esc(d)),$(esc.(args)...))))
        extra_kw_def = d.args
        d = popfirst!(args)
        append!(args,extra_kw_def)
    end
    s = QuoteNode(__source__)
    return :(tag_with_deps!($(esc(d)),$(esc.(DrWatson.convert_to_kw.(args))...),source=$s))
end

"""
    tag_with_deps!(dict; kwargs...)

Behaves like `DrWatson.tag!` but also adds info about (selected) dependencies.

## Keywords

See the documentation of `DrWatson.tag!` for the common kwargs.
    
The following kwargs control which dependencies are included in the tag.
The options are additive, meaning that if multiple options are set to `true`,
all deps that satisfy any of the options are included.

- `deps_by_path::Bool = true`: If `true`, all deps added via a local path are included (even non-direct deps).
- `deps_by_repo::Bool = true`: If `true`, all deps added via a git repo url are included (even non-direct deps).
- `direct_deps::Bool = false`: If `true`, all direct deps are included.
"""
function tag_with_deps!(dict::AbstractDict{K,T};
    gitpath = projectdir(),
    deps_by_path::Bool = true,
    deps_by_repo::Bool = true,
    direct_deps::Bool = false,
    kwargs...    
) where {K,T}
    tag!(dict; gitpath, kwargs...)

    deps_info = Dict{String, Any}()
    for dep in Pkg.dependencies() |> values
        _dep_is_included(dep; deps_by_path, deps_by_repo, direct_deps) || continue
        info = get_dep_info(dep; kwargs...)
        deps_info[dep.name] = info
    end
    dict[:deps] = deps_info

    return dict
end

function _dep_is_included(dep::Pkg.API.PackageInfo; deps_by_path, deps_by_repo, direct_deps)
    (deps_by_path && dep.is_tracking_path) && return true
    (deps_by_repo && dep.is_tracking_repo) && return true
    (direct_deps && dep.is_direct_dep) && return true
    return false
end

function get_dep_info(dep::Pkg.API.PackageInfo; kwargs...)
    info = Dict{Symbol,Any}(
        :name => dep.name,
        :version => dep.version,
    )

    if dep.is_tracking_path
        tag!(info; gitpath=dep.source, kwargs...)
    elseif dep.is_tracking_repo
        info[:gitcommit] = dep.tree_hash
    elseif dep.is_tracking_registry
        info[:gitcommit] = dep.tree_hash
    end

    return info
end

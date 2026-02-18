
function Distributions.truncated(d::Product; lower, upper)
    @assert length(d) == length(lower) == length(upper)
    return Product([truncated(d.v[i]; lower=lower[i], upper=upper[i]) for i in 1:length(d)])
end

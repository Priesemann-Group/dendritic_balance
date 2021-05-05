using LinearAlgebra

function sigmoid(x::Float64)::Float64
    return 1.0 - 1.0 / (1.0 + exp(x))
end

function logit(x::Float64)::Float64
    return log(1.0 / (1 - x) - 1.0)
end

function nonlinearity(x::Float64, shift::Float64, scale::Float64)::Float64
    return sigmoid(scale * (x - shift))
end

function inverse_nonlinearity(x::Float64, shift::Float64, scale::Float64)::Float64
    x = max(0.001, min(0.999, x))
    return logit(x) / scale + shift
end

function moving_average(xs::Array{Float64,1}, n::Int)::Array{Float64,1}
    l = size(xs,1)
    res = zeros(size(xs)...)
    res = res[1:l-n,:]
    for i in 1:l-n
        res[i, :] = sum(xs[i:i+n,:], dims=1) / n
    end
    return res
end

function normalize_to_01(x::Array{Float64,1})::Array{Float64,1}
    mx = maximum(x)
    mi = minimum(x)
    return (x .- mi) ./ (mx - mi)
end

function log_multivariate_gaussian(x::Array{Float64,1}, mean::Array{Float64,1}, var::Float64)::Float64
    n = length(mean)
    norm = -n * ( log(sqrt(2*pi)) - map(log,var.^(0.5)) )
    M = x - mean
    exponent = - 0.5 * M' * (var^(-1) .* M)
    return norm + exponent
end

function multivariate_gaussian(x::Array{Float64,1}, mean::Array{Float64,1}, var::Float64)::Float64
    n = length(mean)
    norm = 1.0/(sqrt(2*pi) * sqrt(var))^n
    M = x - mean
    exponent = exp( - 0.5 * M' * (var^(-1) .* M) )
    return norm * exponent
end

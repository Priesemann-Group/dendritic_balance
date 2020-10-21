
mutable struct Neuron
    kernelLength :: Int
    kernel :: Array{Float64,1}
    recentSpikes :: Array{Int,1}
    # internal clock
    t :: Int
    # probability for last "action"
    prob :: Float64
end

function create_neuron(s::Dict)::Neuron
    kernel = s["kernel"]::Array{Float64,1}
    kernelLength = s["kernelLength"]::Int
    recentSpikes = []
    prob = 0.5
    return Neuron(kernelLength, kernel, recentSpikes, 1, prob)
end

function step_neuron(n::Neuron, input::Float64)::Bool
    n.t += 1
    if length(n.recentSpikes) > 0 && (n.t - n.recentSpikes[1]) >= n.kernelLength
        popfirst!(n.recentSpikes)
    end

    p = p_spike(input)
    r = rand()
    if r < p
        push!(n.recentSpikes, n.t - 1)
        n.prob = p
        return true
    else
        n.prob = 1.0 - p
        return false
    end
end

function get_output(n::Neuron)::Float64
    out = 0.0
    @inbounds for t in n.recentSpikes
        out += n.kernel[n.t - t]
    end
    return out
end

function p_spike(input::Float64)::Float64
    return sigmoid(input)
end

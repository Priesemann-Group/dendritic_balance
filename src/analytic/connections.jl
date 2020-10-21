using SparseArrays

function calc_z_input(net::Net, s::Dict{String,Any})::Array{Float64,1}

    inputs = calc_bias_inputs(net, s)
    inputs += calc_feed_forward_inputs(net, s)
    inputs += calc_recurrent_inputs(net, s)

    return inputs
end

function calc_bias_inputs(net::Net, s::Dict{String,Any})::Array{Float64,1}
    reparametrizeBias = s["reparametrizeBias"]::Bool
    o2 = net.sigma_2

    if reparametrizeBias
        return o2 * net.z_biases
    else
        return net.z_biases
    end
end

function calc_feed_forward_inputs(net::Net, s::Dict{String,Any})::Array{Float64,1}
    o2 = net.sigma_2
    F = net.xz_weights
    x = net.x_outputs

    x_out = o2 .* x

    return F * x_out
end

function calc_recurrent_inputs(net::Net, s::Dict{String,Any})::Array{Float64,1}
    learnedInhibition = s["learnedInhibition"]::Bool

    if !learnedInhibition
        inputs = calc_exact_inhibition(net, s)
    else
        inputs = calc_learned_inhibition(net, s)
    end
    return inputs
end

function calc_exact_inhibition(net::Net, s::Dict{String,Any})::Array{Float64,1}
    z = net.z_outputs
    W = net.memory["inhibition_weights"]::Array{Float64,2}
    instant_self_inhibition = net.memory["instant_self_inhibition"]::Array{Float64,1}

    inh = W * z
    inh += instant_self_inhibition

    return inh
end

function calc_learned_inhibition(net::Net, s::Dict{String,Any})::Array{Float64,1}
    z = net.z_outputs
    o2 = net.sigma_2
    W = net.zz_weights

    inh = W * z
    # this is just a heuristic to keep the network in check -
    # in the end it will be cancelled by the homeostasis
    inh += 0.25 * diag(W) 

    return o2 * inh
end


function get_inhibition_weights(D::Array{Float64,2}, 
    sigma_2::Float64)::Array{Float64,2}

    return - sigma_2 * D' * D
end


function get_inhibition_weights(D::Array{Float64,2}, F::Array{Float64,2},
    sigma_2::Float64)::Array{Float64,2}

    return - sigma_2 * F * D
end

function calc_instant_self_inhibition(W::Array{Float64,2}, s::Dict{String,Any})::Array{Float64,1}
    return 0.25 * diag(W)
end


function create_xz_connections(s::Dict{String,Any})::Array{Float64,2}
    n_x = s["n_x"]::Int
    n_z = s["n_z"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    xz_weights = randn(n_z, n_x) * sqrt(wVar) .+ wMean
    xz_weights = map(x -> max(0, x), xz_weights)

    return xz_weights
end

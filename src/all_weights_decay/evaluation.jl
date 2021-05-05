

function calc_var(net::Net)::Float64
    x = net.x_outputs
    rec = net.reconstruction

    M = x - rec
    return M.^2
end

function calc_var(net::Net, x_input::Array{Float64,1})::Float64
    x = x_input
    rec = net.reconstruction

    M = x - rec
    return M.^2
end

function log_likelihood(net::Net)::Float64
    sig = net.sigma
    x = net.x_outputs
    rec = net.reconstruction

    dyn = 0.0#sum([log(n.prob) for n in net.z_neurons])
    prior = 0.0#sum([log(p_spike(n.bias)) for n in net.z_neurons])
    post = log_multivariate_gaussian(x, rec, sig^2)
    return post + prior - dyn
end

function log_likelihood(net::Net, x_input::Array{Float64,1}, var::Array{Float64,1})::Float64
    x = x_input
    rec = net.reconstruction

    dyn = 0.0#sum([log(n.prob) for n in net.z_neurons])
    prior = 0.0#sum([log(p_spike(n.bias)) for n in net.z_neurons])
    post = log_multivariate_gaussian(x, rec, var)
    return post + prior - dyn
end

function log_decoder_likelihood(net::Net)::Float64
    x = net.x_outputs
    sig = net.sigma
    z = net.z_outputs
    D = net.log.decoder.D
    var = net.log.decoder.var
    rec = net.reconstruction

    dyn = 0.0#sum([log(n.prob) for n in net.z_neurons])
    prior = 0.0#sum([log(p_spike(n.bias)) for n in net.z_neurons])
    post = log_multivariate_gaussian(x, rec, var)
    return post + prior - dyn
end

function log_decoder_free_energy(net::Net)::Float64
    x = net.x_outputs
    sig = net.sigma
    z = net.z_outputs
    z_s = net.z_spikes
    W = net.zz_weights
    var = net.log.decoder.var
    bias = net.log.decoder.biases
    rec = net.reconstruction
    #prior_in = W' * z

    p = log_decoder_free_energy_helper

    dyn = sum([log(n.prob) for n in net.z_neurons])
    prior = sum([log(p(bias[i], Float64(z_s[i]))) for i in 1:net.n_z])
    post = log_multivariate_gaussian(x, rec, var)
    return post + prior - dyn
end

function log_decoder_free_energy_helper(x::Float64,s::Float64)::Float64
    return s * p_spike(x) + (1.0 - s) * (1.0 - p_spike(x))
end

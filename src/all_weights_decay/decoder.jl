
# struct is in log.jl

function create_decoder(s::Dict{String,Any})
    rho = s["rho"]::Float64
    dt = s["dt"]::Float64
    biases = ones(s["n_z"]) .* logit(rho * dt)
    dec = Decoder(zeros(s["n_x"],s["n_z"]),
                   s["initialSigma"], biases)
    return dec
end

function update_decoder(net::Net, s::Dict{String,Any})
    eta = s["learningRateDecoder"]::Float64 * s["dt"]::Float64 
    D = net.log.decoder.D
    tau = s["kernelTau"]::Float64

    t = net.log.t
    tl = net.memory["last_spike_time"]::Int64
    x0 = net.memory["last_x"]::Array{Float64,1}
    z0 = net.memory["last_z"]::Array{Float64,1}

    et = sum([exp((tl - ts) / tau) for ts in tl:t])#tau * (1 - exp((tl - t) / tau))
    et2 = sum([exp(2 * (tl - ts) / tau) for ts in tl:t])#0.5 * tau * (1 - exp(2*(tl - t) / tau))
    z0e = z0 .* et
    z0e2 = z0 .* et2

    rec0 = D * z0
    dD = x0 * z0e' - rec0 * z0e2' 
    net.log.decoder.D += eta * dD
end

function reconstruct_input(net::Net, s::Dict{String,Any})::Array{Float64,1}
    z = net.z_outputs
    D = net.log.decoder.D

    return D * z
end

""" Calculates decoder loss in respect to the current network input"""
function calc_decoder_loss(net::Net)::Float64
    D = net.log.decoder.D
    x = net.x_outputs
    z = net.z_outputs
    rec = net.reconstruction

    err = x - D * z
    return 0.5 / net.n_x * err' * err
end

""" Calculates decoder loss in respect to input x"""
function calc_decoder_loss(net::Net, x::Array{Float64,1})::Float64
    D = net.log.decoder.D
    z = net.z_outputs
    rec = net.reconstruction

    err = x - D * z
    return 0.5 / net.n_x * err' * err
end

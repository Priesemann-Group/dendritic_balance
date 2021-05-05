
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
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateDecoder"]::Float64 * s["dt"]::Float64 
    D = net.log.decoder.D
    z = net.z_outputs
    x = net.x_outputs
    
    e = x - D * z
    dD = e * z'
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

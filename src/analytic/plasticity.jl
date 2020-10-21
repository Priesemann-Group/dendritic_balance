
function update_net(net::Net, s::Dict{String, Any})
    learnedInhibition = s["learnedInhibition"]::Bool
    learnedSigma = s["learnedSigma"]::Bool
    fixedFinalSigma = s["fixedFinalSigma"]::Bool
    homeostaticBiases = s["homeostaticBiases"]::Bool

    if learnedInhibition
        update_zz_inhibitory_weights(net, s)
    end
    if homeostaticBiases
        update_z_biases_homeostatic(net, s)
    else
        update_z_biases(net, s)
    end
    if learnedSigma
        if fixedFinalSigma
            update_sigma_covariant_fixed(net, s)
        else
            update_sigma_covariant(net, s)
        end
    end
end

function update_net_on_spike(net::Net, s::Dict{String, Any})
    localLearning_xz = s["localLearning_xz"]::Bool
    learnedInhibition = s["learnedInhibition"]::Bool

    if localLearning_xz
        update_xz_weights_local(net, s)
    else
        update_xz_weights(net, s)
    end

    if !learnedInhibition
        F = net.xz_weights
        D = net.log.decoder.D
        o2 = net.sigma_2

        net.memory["inhibition_weights"] .= get_inhibition_weights(D, F, o2)
        net.memory["instant_self_inhibition"] .= 
            calc_instant_self_inhibition(net.memory["inhibition_weights"]::Array{Float64,2}, s)
    end
end

function update_xz_weights(net::Net, s::Dict{String,Any})
    eta = s["learningRateFeedForward"]::Float64 * s["dt"]::Float64 
    F = net.xz_weights
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

    rec0 = F' * z0
    dF = x0 * z0e' - rec0 * z0e2' 
    net.xz_weights += eta * dF'
end

function update_xz_weights_local(net::Net, s::Dict{String,Any})
    eta = s["learningRateFeedForward"]::Float64 * s["dt"]::Float64 
    n_x = net.n_x
    n_z = net.n_z

    F = net.xz_weights
    x = net.x_outputs
    rec = net.reconstruction
    
    tau = s["kernelTau"]::Float64

    t = net.log.t
    tl = net.memory["last_spike_time"]::Int64
    x0 = net.memory["last_x"]::Array{Float64,1}
    z0 = net.memory["last_z"]::Array{Float64,1}

    et = sum([exp((tl - ts) / tau) for ts in tl:t])#tau * (1 - exp((tl - t) / tau))
    et2 = sum([exp(2 * (tl - ts) / tau) for ts in tl:t])#0.5 * tau * (1 - exp(2*(tl - t) / tau))
    z0e = z0 .* et
    z0e2 = z0 .* z0 .* et2

    @inbounds for j in 1:n_z
        for i in 1:n_x
            dF = x0[i] * z0e[j] - F[j,i] * z0e2[j]
            F[j,i] += eta * dF
        end
    end
end

function update_zz_inhibitory_weights(net::Net, s::Dict{String,Any})
    W = net.zz_weights
    z = net.z_outputs
    u = net.z_input
    o = net.sigma
    o2 = net.sigma_2
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateInhibitoryRecurrent"]::Float64 * s["dt"]::Float64

    bias = calc_bias_inputs(net, s) + o2 * 0.25 * diag(W)
    inp = o^2 * (u - bias)
    dW = - inp * z'
    net.zz_weights += eta * dW
end

function update_z_biases(net::Net, s::Dict{String,Any})
    r = net.z_rates
    b = net.z_biases
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateBias"]::Float64 * s["dt"]::Float64

    @inbounds for j in 1:net.n_z
        db = r[j] - p_spike(b[j])
        b[j] += eta * db
    end
    net.z_rates .= 0.0
end

function update_z_biases_homeostatic(net::Net, s::Dict{String,Any})
    r = net.z_rates
    b = net.z_biases
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateHomeostaticBias"]::Float64 * s["dt"]::Float64
    rho = s["rho"]::Float64
    dt = s["dt"]::Float64

    goalrate = dt * rho
    @inbounds for j in 1:net.n_z
        db = goalrate - r[j]
        b[j] += eta * db
    end
    net.z_rates .= 0.0
end

function update_sigma_covariant(net::Net, s::Dict{String,Any})
    sig = net.sigma
    x = net.x_outputs
    n_x = net.n_x
    alpha = 1.0 / s["sigmaLearningOffset"]::Float64
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateSigma"]::Float64 * s["dt"]::Float64

    rec = net.reconstruction
    M = x - rec
    dsig = 0.0
    @inbounds for i in 1:n_x
        dsig += (M[i]^2 - alpha * sig^2) * sig^(-1)
    end

    net.sigma += eta * dsig / n_x
    net.sigma_2 = net.sigma^(-2)
end

function update_sigma_covariant_fixed(net::Net, s::Dict{String,Any})
    sig = net.sigma
    x = net.x_outputs
    n_x = net.n_x
    alpha = s["fixedFinalSigmaValue"]::Float64 ^ 2
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateSigma"]::Float64 * s["dt"]::Float64

    dsig = 0.0
    @inbounds for i in 1:n_x
        dsig += (alpha - sig^2) * sig^(-1)
    end

    net.sigma += eta * dsig / n_x
    net.sigma_2 = net.sigma^(-2)
end


function update_net(net::Net, s::Dict{String, Any})
    hebbianLearning_xz = s["hebbianLearning_xz"]::Bool
    unwhitened_input = s["unwhitenedInput"]::Bool
    learnedInhibition = s["learnedInhibition"]::Bool
    learnedSigma = s["learnedSigma"]::Bool
    homeostaticBiases = s["homeostaticBiases"]::Bool

    if hebbianLearning_xz
        if unwhitened_input
            update_xz_weights_hebbian_unwhitened(net, s)
        else
            update_xz_weights_hebbian(net, s)
        end
    else
        update_xz_weights(net, s)
    end
    if learnedInhibition
        update_zz_inhibitory_weights(net, s)
    end

    if homeostaticBiases
        update_z_biases_homeostatic(net, s)
    else
        update_z_biases(net, s)
    end
    if learnedSigma
        update_sigma_covariant(net, s)
    end
    if !learnedInhibition
        F = net.xz_weights
        D = net.log.decoder.D
        o2 = net.sigma_2

        net.memory["inhibition_weights"] .= get_inhibition_weights(F, o2)
        net.memory["instant_self_inhibition"] .=
            calc_instant_self_inhibition(net.memory["inhibition_weights"]::Array{Float64,2}, s)
    end
end

function update_xz_weights(net::Net, s::Dict{String,Any})
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateFeedForward"]::Float64 * s["dt"]::Float64
    F = net.xz_weights
    z = net.z_outputs
    x = net.x_outputs

    e = x - F' * z
    dF = e * z'
    net.xz_weights += eta * dF'
end

function update_xz_weights_hebbian(net::Net, s::Dict{String,Any})
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateFeedForward"]::Float64 * s["dt"]::Float64
    F = net.xz_weights
    x = net.x_outputs
    z = net.z_outputs

    for i in 1:net.n_x
        for j in 1:net.n_z
            dF = z[j] * (x[i] - F[j,i] * z[j])
            F[j,i] += eta * dF
        end
    end
end

function update_xz_weights_hebbian_unwhitened(net::Net, s::Dict{String,Any})
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateFeedForward"]::Float64 * s["dt"]::Float64
    F = net.xz_weights
    x = net.x_outputs
    z = net.z_outputs

    act = F * x
    # alpha rescales weights to typical size
    alpha = 10.0
    pots = 1 .- act .* z ./ alpha

    for i in 1:net.n_x
        for j in 1:net.n_z
            dF = z[j] * (x[i] * pots[j])
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
    eta = s["learningRateInhibitoryRecurrent"]::Float64 * s["dt"]::Float64

    bias = calc_bias_inputs(net, s) + o2 * 0.25 * diag(W)
    inp = o^2 * (u - bias)
    dW = - inp * z'
    net.zz_weights += eta * dW
end

function update_zz_inhibitory_weights_brendel(net::Net, s::Dict{String,Any})
    W = net.zz_weights
    z = net.z_outputs
    z_s = net.z_spikes
    u = net.z_input
    o = net.sigma
    o2 = net.sigma_2
    batchmult = s["updateInterval"]::Int
    eta = s["learningRateInhibitoryRecurrent"]::Float64
    mu = 0.1
    alpha = 1.0

    bias = calc_bias_inputs(net, s) + o2 * 0.5 * diag(W)
    inp = o^2 * (u - bias)
    dW = - alpha * (inp + mu * z) * z_s' - (ones(net.n_z) * z_s') .* (W + mu * diagm(ones(net.n_z)))
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

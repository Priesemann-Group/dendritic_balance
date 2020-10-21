
function update_net(net::Net, s::Dict{String, Any})
    learnedInhibition = s["learnedInhibition"]::Bool
    learnedSigma = s["learnedSigma"]::Bool
    fixedFinalSigma = s["fixedFinalSigma"]::Bool
    homeostaticBiases = s["homeostaticBiases"]::Bool

    update_xz_weights(net, s)
    update_zz_inhibitory_weights(net, s)
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

function update_xz_weights(net::Net, s::Dict{String,Any})
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateFeedForward"]::Float64 * s["dt"]::Float64 
    F = net.xz_weights
    z = net.z_outputs
    x = net.x_outputs
    u = net.dendritic_potentials

    for j in 1:net.n_z
        for i in 1:net.n_x
            dF = z[j] * (x[i] + (u[i,j] - F[j,i] * x[i]) / max(0.002, abs(F[j,i])) * sign(F[j,i]))
            F[j,i] += eta * dF
        end
    end
end

function update_zz_inhibitory_weights(net::Net, s::Dict{String,Any})
    W = net.zz_weights
    z = net.z_outputs
    u = net.dendritic_potentials
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateInhibitoryRecurrent"]::Float64 * s["dt"]::Float64 

    @inbounds for j in 1:net.n_z
        for i in 1:net.n_x
            for k in 1:net.n_z
                dW = - z[k] * u[i,j]
                W[j,i,k] += eta * dW
            end
        end
    end
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

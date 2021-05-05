
function update_net(net::Net, s::Dict{String, Any})
    learnedInhibition = s["learnedInhibition"]::Bool
    learnedSigma = s["learnedSigma"]::Bool
    fixedFinalSigma = s["fixedFinalSigma"]::Bool
    homeostaticBiases = s["homeostaticBiases"]::Bool


    x = net.x_outputs
    z = net.z_outputs
    F = net.xz_weights
    W_pruned = net.zz_weights_pruned

    dendritic_potentials = zeros(net.n_x, net.n_z)
    @inbounds for j in 1:net.n_z
        dendritic_potentials[:,j] = W_pruned[j,:,:] * z
        dendritic_potentials[:,j] += F[j,:] .* x
    end


    update_xz_weights(net, dendritic_potentials, s)
    # still learning regular weights for dynamics
    update_zz_inhibitory_weights(net, s)
    update_zz_inhibitory_pruned_weights(net, dendritic_potentials, s)
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

function update_xz_weights(net::Net, u::Array{Float64,2}, s::Dict{String,Any})
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateFeedForward"]::Float64 * s["dt"]::Float64 
    eta_bar = batchmult * s["learningRateFeedForwardBar"]::Float64 * s["dt"]::Float64 
    F = net.xz_weights
    F_bar = net.memory["xz_weights_bar"]::Array{Float64,2}
    z = net.z_outputs
   
    for j in 1:net.n_z
        for i in 1:net.n_x
            dF_bar = z[j] * u[i,j] 
            F_bar[j,i] += eta_bar * dF_bar
            dF = F_bar[j,i] / F[j,i] - F[j,i]
            F[j,i] += eta * dF
        end
    end
    net.memory["xz_weights_bar"] .= F_bar
end

function update_zz_inhibitory_weights(net::Net, s::Dict{String,Any})
    u = net.potentials
    z = net.z_outputs
    W = net.zz_weights
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateInhibitoryRecurrent"]::Float64 * s["dt"]::Float64 
    
    @inbounds for j in 1:net.n_z
        for k in 1:net.n_z
            dW = - z[k] * u[j]
            W[j,k] += eta * dW
        end
    end
end

function update_zz_inhibitory_pruned_weights(net::Net, u::Array{Float64,2}, s::Dict{String,Any})
    z = net.z_outputs
    W_pruned = net.zz_weights_pruned
    batchmult = s["updateInterval"]::Int
    alpha = s["learningRateCorrelation"]::Float64 * s["dt"]::Float64 
    beta = s["learningFactorPruning"]::Float64
    eta = batchmult * s["learningRateInhibitoryRecurrent"]::Float64 * s["dt"]::Float64 

    @inbounds for j in 1:net.n_z
        for i in 1:net.n_x
            for k in 1:net.n_z
                dW = - z[k] * u[i,j]
                W_pruned[j,i,k] += eta * dW
            end
        end
    end

    # prune weights
    interval = s["tempLogInterval"]::Int
    t = net.log.t
    k = div(t - 1, interval)

    pruning_fraction = s["pruningFraction"]::Float64
    pruning_coefficients = net.memory["pruning_coefficients"]::Array{Float64,3}
    self_connection_mask = net.memory["self_connection_mask"]::Array{Bool,3}

    # update correlations
    z_cov = net.memory["z_cov"]::Array{Float64,2}
    z_cov .= (1 - alpha) * z_cov + alpha * (z * z')

    pruning_coefficients .= map(abs, permutedims(permutedims(W_pruned, [1,3,2]) .* z_cov, [1,3,2]))

    # find size of p_c at p_f percentile of p_c
    pc_vec = vec(pruning_coefficients)
    pc_prune_order = sortperm(pc_vec)
    ind_crit = Int(floor(1 + pruning_fraction * net.n_z ^ 2 * net.n_x))
    pc_crit = pc_vec[pc_prune_order[ind_crit]]
    # adjust weights with smaller p_c towards zero, keep self connections
    pruning_mask = (pruning_coefficients .<= pc_crit) .& self_connection_mask
    dW = - pruning_mask .* W_pruned
    W_pruned .+= beta * eta * dW 
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

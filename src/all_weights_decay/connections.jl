using SparseArrays

function calc_z_input(net::Net, s::Dict{String,Any})::Array{Float64,1}

    n_z = net.n_z
    x = net.x_outputs
    z = net.z_outputs
    o2 = net.sigma_2
    F = net.xz_weights
    W = net.zz_weights

    inputs = zeros(n_z)

    @inbounds for j in 1:n_z
        net.dendritic_potentials[:,j] = W[j,:,:] * z
        net.dendritic_potentials[:,j] += F[j,:] .* x
        inputs[j] = sum(net.dendritic_potentials[:,j])
        inputs[j] += 0.25 * sum(W[j,:,j]) + net.z_biases[j]
    end

    return o2 * inputs
end

function get_inhibition_weights(D::Array{Float64,2}, 
    sigma_2::Float64)::Array{Float64,2}

    return - sigma_2 * D' * D
end


function get_inhibition_weights(D::Array{Float64,2}, F::Array{Float64,2},
    sigma_2::Float64)::Array{Float64,2}

    return - sigma_2 * F * D
end


function get_all_inhibition_weights(F::Array{Float64,2},
    sigma_2::Float64)::Array{Float64,3}

    n_x = size(F,2)
    n_z = size(F,1)    

    zz_weights = zeros(n_z, n_x, n_z)
    for i in 1:n_x
        for j in 1:n_z
            for k in 1:n_z
                zz_weights[j,i,k] = - sigma_2 * F[j,i] * F[k,i]
            end
        end
    end
    
    return zz_weights
end



function create_xz_connections(s::Dict{String,Any})::Array{Float64,2}
    n_x = s["n_x"]::Int
    n_z = s["n_z"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    xz_weights = randn(n_z, n_x) * sqrt(wVar) .+ wMean
    xz_weights = map(abs, xz_weights)

    return xz_weights
end

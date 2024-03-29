using SparseArrays

function calc_z_input(net::Net, s::Dict{String,Any})::Array{Float64,1}

    n_z = net.n_z
    x = net.x_outputs
    z = net.z_outputs
    o2 = net.sigma_2
    F = net.xz_weights
    W = net.zz_weights

    inputs = zeros(n_z)

    net.potentials = W * z
    net.potentials += F * x
    inputs = net.potentials + 0.25 * diag(W) + net.z_biases

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


function create_xz_connections(s::Dict{String,Any})::Array{Float64,2}
    n_x = s["n_x"]::Int
    n_z = s["n_z"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    xz_weights = randn(n_z, n_x) * sqrt(wVar) .+ wMean
    xz_weights = map(abs, xz_weights)

    return xz_weights
end


""" A Snapshot saves the dynamics and state
of the network at a given moment in time."""
mutable struct Snapshot
    t :: Int

    z_spikes :: Array{Bool}
    x_outputs :: Array{Float64,2}
    z_outputs :: Array{Float64,2}

    xz_weights :: Array{Float64, 2}
    zz_weights :: Array{Float64, 2}
    z_biases :: Array{Float64,1}

    sigma_2 :: Float64

    weights_necessity :: Array{Float64, 3}
    weights_update_magnitude :: Array{Float64, 2}
    decoder_weights :: Array{Float64, 2}

    test_inputs :: Array{Float64, 2}
    reconstructions :: Array{Float64, 2}
    reconstruction_means :: Array{Float64, 2}
    reconstruction_vars :: Array{Float64, 2}
end

function create_snapshot(nSteps::Int, interval::Int, t::Int, inputs::Array{Float64,2}, 
    net::Net, s::Dict{String,Any})::Snapshot
    
    z_spikes = falses(div(nSteps,interval), s["n_z"])
    x_outputs = zeros(div(nSteps,interval), s["n_x"])
    z_outputs = zeros(div(nSteps,interval), s["n_z"])
    z_biases = net.z_biases
    l = s["presentationLength"]::Int
    splitPosNeg = s["splitPosNegInput"]::Bool
    nImg = splitPosNeg ? div(s["n_x"],2) : s["n_x"]
    weights_necessity = zeros(net.n_x,net.n_z,net.n_z) # W_ijk
    weights_update_magnitude = zeros(net.n_x,net.n_z) # F_ij
    decoder_weights = net.log.decoder.D
    reconstructions = zeros(div(nSteps,interval), s["n_x"])
    reconstruction_means = zeros(div(nSteps,l), s["n_x"])
    reconstruction_vars = zeros(div(nSteps,l), s["n_x"])
    return Snapshot(t, z_spikes, x_outputs, z_outputs,
                    copy(net.xz_weights), copy(net.zz_weights),
                    z_biases, copy(net.sigma_2), weights_necessity, weights_update_magnitude,
                    copy(decoder_weights), inputs, reconstructions,
                    reconstruction_means, reconstruction_vars)
end

""" Creates the snapshot and record the dynamics on a training data-set."""
function take_snapshot(net::Net, log::Log, inputs::Array{Float64,2}, s::Dict{String,Any},
    nTestElements::Int=64)

    l = s["presentationLength"]::Int
    fadeLength = s["fadeLength"]::Float64
    interval = s["snapshotLogInterval"]::Int
    testWeightsNecessity = s["testWeightsNecessity"]::Bool

    nSteps = size(inputs, 1) * l
    nFirstSteps = min(nSteps, nTestElements*l)
    snapshot = create_snapshot(nFirstSteps, interval, log.t, inputs, net, s)

    z_spikes = falses(net.n_z)
    rec_counter = 0
    for i in 1:nFirstSteps
        x = fade_images(inputs,i,s)
        step_net(net, x, s, update=false)

        rec = net.reconstruction
        if ((i - 1) % l > l * fadeLength)
            snapshot.reconstruction_means[div(i - 1, l) + 1, :] += rec / (l * (1.0 - fadeLength))
        end
        snapshot.reconstruction_vars[div(i - 1, l) + 1, :] +=
            (net.x_outputs - rec).^2 / (l - 1)

        z_spikes .|= net.z_spikes

        if i % interval == 0
            ind = div(i, interval)
            snapshot.z_spikes[ind,:] = z_spikes
            z_spikes = falses(net.n_z)
            snapshot.x_outputs[ind,:] = net.x_outputs
            snapshot.z_outputs[ind,:] = net.z_outputs
            snapshot.reconstructions[ind, :] = rec
        end
    end

    push!(log.snapshots, snapshot)
end

function get_xz_weights_gradient(net::Net, s::Dict{String,Any})::Array{Float64,2}
    z = net.z_outputs
    rec = net.reconstruction
    x = net.x_outputs

    dF = z * (x - rec)'
    return dF
end


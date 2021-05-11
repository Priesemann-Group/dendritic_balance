using ProgressMeter
using LinearAlgebra: diagind
using BSON

mutable struct Net
    n_x :: Int32
    x_outputs :: Array{Float64,1}

    n_z :: Int32
    z_neurons :: Array{Neuron,1}
    z_outputs :: Array{Float64,1}
    z_spikes :: Array{Bool,1}
    z_rates :: Array{Float64,1}
    z_input :: Array{Float64,1}

    # F has dimensions (#z-nrns, #x-nrns)
    xz_weights :: Array{Float64, 2}

    # W has dimensions (#z-nrns, #z-nrns)
    zz_weights :: Array{Float64, 2}

    z_biases :: Array{Float64,1}

    sigma :: Float64
    sigma_2 :: Float64 # sigma^(-2) !

    # reconstruction of the current input
    reconstruction :: Array{Float64,1}

    memory :: Dict{String, Any}

    log :: Log
end

function create_net(s::Dict{String,Any}, log::Log)::Net
    n_x = s["n_x"]::Int
    n_z = s["n_z"]::Int
    rho = s["rho"]::Float64
    dt = s["dt"]::Float64
    initialSigma = s["initialSigma"]::Float64
    learnedInhibition = s["learnedInhibition"]::Bool
    reparametrizeBias = s["reparametrizeBias"]::Bool

    memory = Dict()

    x_outputs = zeros(n_x)

    z_neurons = [create_neuron(s) for i in 1:n_z]
    z_outputs = zeros(n_z)
    z_input = zeros(n_z)
    z_spikes  = falses(n_z)
    z_rates = zeros(n_z)

    sigma = initialSigma
    sigma_2 = sigma ^ (-2)

    xz_weights = create_xz_connections(s)
    
    if learnedInhibition
        zz_weights = get_inhibition_weights(xz_weights, sigma_2) ./ sigma_2
    else
        zz_weights = get_inhibition_weights(xz_weights, sigma_2) 
        memory["inhibition_weights"] = zeros(n_z, n_z)
        memory["instant_self_inhibition"] = zeros(n_z)
    end

    z_biases = logit(rho * dt) * ones(n_z)
    # When learning inhibition reparametrise bias
    if reparametrizeBias
        z_biases *= initialSigma^2
    end

    reconstruction = zeros(n_x)
    
    net = Net(n_x, x_outputs,
              n_z, z_neurons, z_outputs, z_spikes, z_rates, z_input,
              xz_weights, zz_weights, z_biases, sigma, sigma_2,
              reconstruction, memory, log)
    return net
end

""" Runs the network on a dataset and a testset.

 - x_inputs: (nPatterns, n)-Matrix with input strengths.
 - test_input: (:, n)-Matrix with test-inputs.
 - test_times: Specifies at what times the snapshots are taken.
"""
function run_net(net::Net, x_inputs::Array{Float64,2}, test_input::Array{Float64,2}, 
    test_times::Array{Int,1}, save_loc::String, s::Dict{String,Any})

    l = s["presentationLength"]::Int
    interval = s["tempLogInterval"]::Int
    updateInterval = s["updateInterval"]::Int
    changeDict = s["paramChangeDict"]::Dict{String,Dict{Int64,Any}}
    showProgressBar = s["showProgressBar"]::Bool

    nSteps = size(x_inputs,1) * l
    runningLog = setup_temp_log(net, nSteps, interval)

    dtBar = showProgressBar ? 0.1 : Inf
    @showprogress dtBar for t in 1:nSteps
        net.log.t = t

        # get input to net by fading between images
        x = @inbounds fade_images(x_inputs,t,s)

        # update network
        batchUpdate = t % updateInterval == 0 # use "batch" update to save time
        step_net(net, x, s, update=true, batchUpdate=batchUpdate)

        # log everything
        log_everything(x, test_input, runningLog, test_times, save_loc, net, s)

        # online-update of parameters
        for (param, timeDict) in changeDict
            if haskey(timeDict, t)
                s[param] = timeDict[t]
            end
        end
    end
end

""" Takes the network and steps one timestep forward.

 - update: If true, the weights will be updated.
"""
function step_net(net::Net, x_input::Array{Float64,1}, s::Dict{String,Any};
    update::Bool = false, batchUpdate::Bool = false)

    net.x_outputs = x_input

    # fire z neurons
    net.z_input = calc_z_input(net, s)
    net.z_spikes = map(step_neuron, net.z_neurons, net.z_input)

    one_spike = s["oneSpikePerTimestep"]::Bool
    if one_spike
        reduce_spikes_to_one_spike(net, s)
    end

    # update parameters
    if update
        if batchUpdate
            update_net(net, s)
            update_decoder(net, s)
        end
    end  

    # update z outputs
    net.z_outputs = Array{Float64,1}(map(get_output, net.z_neurons))

    # reconstruct the current "prediction" of the input
    net.reconstruction = reconstruct_input(net, s)
end

function reduce_spikes_to_one_spike(net::Net, s::Dict{String,Any})
    if sum(net.z_spikes) > 1
        _, ind_spiker = findmax([net.z_spikes[j] * net.z_neurons[j].prob for j in 1:net.n_z])
        for j in 1:net.n_z
            if net.z_spikes[j] && j != ind_spiker
                pop!(net.z_neurons[j].recentSpikes)
                net.z_neurons[j].prob = 1.0 - net.z_neurons[j].prob
                net.z_spikes[j] = false
            end
        end
    end
end

function write_status(save_loc::String, net::Net, log::Log)
    try
        save_net(net, save_loc)
    catch e
        print("Error saving net:\n")
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        println(msg)
    end
    try
        save_log(log, save_loc)
    catch e
        print("Error saving log:\n")
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        println(msg)
    end
end

function log_everything(x::Array{Float64,1}, test_input::Array{Float64,2}, runningLog::Dict{String, Any}, 
    test_times::Array{Int,1}, save_loc::String, net::Net, s::Dict{String, Any})
    
    interval = s["tempLogInterval"]::Int
    sampleInterval = s["tempLogSampleInterval"]::Int
    numSamples = div(interval, sampleInterval)
    updateInterval = s["updateInterval"]::Int
    changeDict = s["paramChangeDict"]::Dict{String,Dict{Int64,Any}}

    # rates have to be updated for homeostasis
    net.z_rates += net.z_spikes / updateInterval
    runningLog["mean_firing_rate"] += mean(net.z_spikes) / interval

    # update & save the log at certain time intervals
    t = net.log.t

    if t % sampleInterval == 0
        update_running_log(net, runningLog, x, interval, numSamples, s)
    end
    if t % interval == 0 
        log_temp_log(net, runningLog, test_input, interval, s)
    end
    if t in test_times
        take_snapshot(net, net.log, test_input, s)
        write_status(save_loc, net, net.log)
    end
end

function save_net(net::Net, name::String)
    rm("$name/net.bson", force=true)
    bson("$name/net.bson", net = net)
end

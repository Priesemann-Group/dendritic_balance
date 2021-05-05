using Statistics: mean, var
using Dates
""" The temp log can be used to save additional quantities.
They will be plotted automatically in the end. The 'runningLog'
is used to average over time until the next logging point is reached.

To log something:
1. create desired variable in runningLog & temp log in *setup_temp_log*
2. update runningLog every timestep in *update_running_log*
3. save results from averaging to the temp log in *log_temp_log*"""

function setup_temp_log(net::Net, nSteps::Int, interval::Int)::Dict{String, Any}
    runningLog = Dict{String, Any}()

    # time of log
    net.log.temp["t"] = zeros(div(nSteps, interval))

    # variance learned by the net
    net.log.temp["var"] = zeros(div(nSteps, interval))
    runningLog["var"] = 0.0

    # mean firing rate
    net.log.temp["mean_firing_rate"] = zeros(div(nSteps, interval))
    runningLog["mean_firing_rate"] = 0.0

    # mean z bias
    net.log.temp["mean_z_bias"] = zeros(div(nSteps, interval))
    runningLog["mean_z_bias"] = 0.0

    # mean xz weights
    net.log.temp["mean_xz_weights"] = zeros(div(nSteps, interval))
    runningLog["mean_xz_weights"] = 0.0

    # var xz weights
    net.log.temp["var_xz_weights"] = zeros(div(nSteps, interval))
    runningLog["var_xz_weights"] = 0.0

    # membrane
    net.log.temp["z_inputs"] = zeros(div(nSteps, interval), net.n_z)
    runningLog["z_inputs"] = zeros(net.n_z)

    # decoder var
    net.log.temp["decoder_var"] = zeros(div(nSteps, interval))
    runningLog["decoder_var"] = 0.0

    # test measures
    net.log.temp["test_decoder_loss"] = zeros(div(nSteps, interval))
    net.log.temp["test_decoder_likelihood"] = zeros(div(nSteps, interval), 2)

    return runningLog
end

function update_running_log(net::Net, runningLog::Dict{String,Any},
    x_input::Array{Float64,1}, interval::Int, numSamples::Int, s::Dict{String,Any})

    t = net.log.t
    k = div(t - 1, interval) + 1

    strengthOn = s["stimulusStrengthOn"]::Float64
    strengthOff = s["stimulusStrengthOff"]::Float64
    input_01 = (x_input .- strengthOff) ./ (strengthOn - strengthOff)

    runningLog["var"] += net.sigma ^ 2 / numSamples
    runningLog["mean_z_bias"] += mean(net.z_biases) / numSamples
    runningLog["mean_xz_weights"] += mean(net.xz_weights) / numSamples
    runningLog["var_xz_weights"] += var(net.xz_weights) / numSamples
    runningLog["z_inputs"] += net.z_input ./ numSamples
    runningLog["decoder_var"] += net.log.decoder.var ./ numSamples
end

function log_temp_log(net::Net, runningLog::Dict{String,Any},
    inputs::Array{Float64,2}, interval::Int, s::Dict{String,Any})

    t = net.log.t
    k = div(t - 1, interval) + 1
    dt = net.log.settings["dt"]
    temp = net.log.temp

    test_performance(net, inputs, interval, s)

    temp["t"][k] = t

    temp["var"][k] = runningLog["var"]
    runningLog["var"] = 0.0

    temp["mean_firing_rate"][k] = runningLog["mean_firing_rate"] / (0.001 * dt)
    runningLog["mean_firing_rate"] = 0.0

    temp["mean_z_bias"][k] = runningLog["mean_z_bias"]
    runningLog["mean_z_bias"] = 0.0

    temp["mean_xz_weights"][k] = runningLog["mean_xz_weights"]
    runningLog["mean_xz_weights"] = 0.0

    temp["var_xz_weights"][k] = runningLog["var_xz_weights"]
    runningLog["var_xz_weights"] = 0.0

    temp["z_inputs"][k, :] = runningLog["z_inputs"]
    runningLog["z_inputs"] = zeros(net.n_z)

    temp["decoder_var"][k] = runningLog["decoder_var"]
    runningLog["decoder_var"] = 0.0
end

""" Tests the performance on a training data-set."""
function test_performance(net::Net, inputs::Array{Float64,2}, interval::Int, s::Dict{String,Any})
    l = s["presentationLength"]::Int

    nSteps = size(inputs, 1) * l
    k = div(net.log.t - 1, interval) + 1
    temp = net.log.temp

    for i in 1:nSteps
        x = fade_images(inputs,i,s)
        step_net(net, x, s, update=false)

        temp["test_decoder_loss"][k] += calc_decoder_loss(net, x) / nSteps
        temp["test_decoder_likelihood"][k,1] += log_decoder_likelihood(net) / nSteps
        temp["test_decoder_likelihood"][k,2] += log_decoder_free_energy(net) / nSteps
    end
end

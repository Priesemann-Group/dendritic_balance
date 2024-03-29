include("../src/all_weights_simultaneous/settings.jl")
include("../src/all_weights_simultaneous/main.jl")

using Statistics: var
using DelimitedFiles
using Random

function main_bars(plotflag, s, corr, noise, id)
    print("Setting up...\n")
    nPatterns = 1000000
    nTestPatterns = 500

    set!(s, "showProgressBar", true)

    dt = 1.0
    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", Int(5.0/dt))
    set!(s, "updateInterval", 1)
    set!(s, "snapshotLogInterval", Int(3.0/dt))

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 50000 * l)

    set!(s, "n_x", 4^2) # number x neurons
    set!(s, "n_z", 8) # number z neurons

    set!(s, "learningRateFeedForward", 5.0e-5)
    set!(s, "learningRateDecoder", 5.0e-5)
    set!(s, "learningRateInhibitoryRecurrent", 10.0e-5)
    set!(s, "learningRateHomeostaticBias", 1.0e-2)
    set!(s, "learningRateSigma", 7.0e-8)


    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigmaValue", sqrt(0.1))

    set!(s, "hebbianLearning_xz", false)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedInhibition", false)
    set!(s, "homeostaticBiases", true)
    set!(s, "rho", 0.015)

    set!(s, "inputNoise", noise)

    inputs, _ = create_bar_inputs(nPatterns, s, corr, 0.0)
    test_inputs, labels = create_bar_inputs(nTestPatterns, s, corr, 0.0)
    test_times = [1, div(nSteps,2), nSteps-1]

    return main("S3_bars_all_weights_simultaneous", plotflag, s, nSteps, inputs, test_inputs, test_times)
end

plotflag = true
id = 1
corr = 0.7
noise = 0.0
main_bars(plotflag, copy(standardSettings), corr, noise, id)

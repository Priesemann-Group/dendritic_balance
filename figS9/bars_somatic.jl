include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

using Statistics: var
using DelimitedFiles
using Random

function main_bars_local(plotflag, s, corr)
    print("Setting up...\n")
    nPatterns = 400000
    nTestPatterns = 500

    set!(s, "showProgressBar", true)

    dt = 1.0
    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", Int(5.0/dt))
    set!(s, "updateInterval", 1)
    set!(s, "snapshotLogInterval", Int(3.0/dt))

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 10000 * l)

    set!(s, "n_x", 4^2) # number x neurons
    set!(s, "n_z", 8) # number z neurons

    speed = 0.9

    set!(s, "learningRateFeedForward", speed * 1.0e-4)
    set!(s, "learningRateDecoder", speed * 1.0e-4)
    set!(s, "learningRateInhibitoryRecurrent", speed * 5.0e-4)
    set!(s, "learningRateHomeostaticBias", speed * 4.0e-2)
    set!(s, "learningRateSigma", speed * 7.0e-7)

    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigmaValue", sqrt(0.1))

    set!(s, "hebbianLearning_xz", true)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedInhibition", true)
    set!(s, "homeostaticBiases", true)
    set!(s, "rho", 0.015)

    inputs, _ = create_bar_inputs(nPatterns, s, corr, 0.0)
    test_inputs, labels = create_bar_inputs(nTestPatterns, s, corr, 0.0)
    test_times = [1, div(nSteps,2), nSteps-1]

    return main("S9_bars_somatic_pruning_reference", plotflag, s, nSteps, inputs, test_inputs, test_times, name="p=$(corr)")
end

plotflag = true
corr = 0.6
net = main_bars_local(plotflag, copy(standardSettings), corr)

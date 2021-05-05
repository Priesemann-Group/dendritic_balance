include("../src/all_weights_slow_pruning/settings.jl")
include("../src/all_weights_slow_pruning/main.jl")

using Statistics: var
using DelimitedFiles
using Random

function main_bars(plotflag, s, corr, pruning_fraction)
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

    set!(s, "learningRateFeedForward", speed * 1.0e-6)
    set!(s, "learningRateDecoder", speed * 5.0e-4)
    set!(s, "learningRateInhibitoryRecurrent", speed * 5.0e-4)
    set!(s, "learningRateHomeostaticBias", speed * 4.0e-2)
    set!(s, "learningRateSigma", speed * 7.0e-7)
    set!(s, "learningRateFeedForwardBar", speed * 5.0e-4)

    s["paramChangeDict"]["pruningFraction"] =
        Dict(1        => 0.0,
             4000 * l => pruning_fraction)
    set!(s, "learningRateCorrelation", speed * 1.0e-4)
    set!(s, "learningFactorPruning", 3.0)

    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigma", true)
    set!(s, "fixedFinalSigmaValue", sqrt(0.1))

    set!(s, "localLearning_xz", false)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedInhibition", false)
    set!(s, "homeostaticBiases", true)
    set!(s, "rho", 0.015)
    
    set!(s, "weightMean", 0.001)

    Random.seed!(1010)
    inputs, _ = create_bar_inputs(nPatterns, s, corr, 0.0)
    test_inputs, labels = create_bar_inputs(nTestPatterns, s, corr, 0.0)
    test_times = collect(1:div(nSteps,3):nSteps-1)

    return main("S9_bars_all_weights_slow_pruning", plotflag, s, nSteps, inputs, test_inputs, test_times, name="pf=$(pruning_fraction)")
end

plotflag = true
corr = 0.6
pruning_fractions = [0.0, 0.25, 0.5, 0.75, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
for pruning_fraction in pruning_fractions
    net = main_bars(plotflag, copy(standardSettings), corr, pruning_fraction)
end

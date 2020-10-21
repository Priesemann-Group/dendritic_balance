include("../src/all_weights_delay/settings.jl")
include("../src/all_weights_delay/main.jl")

function main_mnist(plotflag, s)
    print("Setting up...\n")
    nPatterns = 120000
    nTestPatterns = 1000
    nNumbers = 3

    set!(s, "comment", "")
    set!(s, "showProgressBar", true)

    set!(s, "dt", 0.1)
    set!(s, "tempLogSampleInterval", 5)
    set!(s, "updateInterval", 5)
    set!(s, "snapshotLogInterval", 5)

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 2000 * l)

    set!(s, "n_x", 16^2 )# number x neurons
    set!(s, "n_z", 9) # number z neurons

    set!(s, "learningRateFeedForward", 4e-7)
    set!(s, "learningRateInhibitoryRecurrent", 4e-5)    
    set!(s, "learningRateHomeostaticBias", 0.5e-2)
    set!(s, "learningRateDecoder", 0.5e-5)
    set!(s, "learningRateFeedForwardBar", 4e-5)

    set!(s, "initialSigma", sqrt(0.1))

    set!(s, "localLearning_xz", false)
    set!(s, "learnedInhibition", false)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedSigma", false)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", 0.02)
    
    set!(s, "weightVariance", 0.0)
    set!(s, "weightMean", 0.001)

    inputs = create_mnist_inputs(nPatterns, s, nNumbers=nNumbers, scale=16/28)
    test_inputs = create_mnist_test(nTestPatterns, s, nNumbers=nNumbers, scale=16/28)
    test_times = collect(0:div(nSteps, 3):nSteps)

    return main("mnist_all_weights_delay", plotflag, s, nSteps, inputs, test_inputs, test_times)
end
    
plotflag = true
main_mnist(plotflag, copy(standardSettings))

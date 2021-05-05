include("../src/all_weights_decay/settings.jl")
include("../src/all_weights_decay/main.jl")

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

    set!(s, "learningRateInhibitoryRecurrent", 6e-5)    
    set!(s, "learningRateHomeostaticBias", 3e-3)
    set!(s, "learningRateDecoder", 3e-6)

    set!(s, "weightMean", 0.001)
    
    nInitSteps = 5000 * l
    s["paramChangeDict"]["learningRateFeedForward"] =
        Dict(1          => 0.0,    
             nInitSteps => 2.0e-6)

    set!(s, "initialSigma", sqrt(0.1))

    set!(s, "localLearning_xz", false)
    set!(s, "learnedInhibition", false)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedSigma", false)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", 0.005)

    lambda = 0.005
    set!(s, "lambdaConstraint", lambda)
    set!(s, "comment", "lambda=$(lambda)")
    
    inputs = create_mnist_inputs(nPatterns, s, nNumbers=nNumbers)
    test_inputs = create_mnist_test(nTestPatterns, s, nNumbers=nNumbers)
    test_times = collect(0:div(nSteps, 5):nSteps)

    return main("S2_mnist_all_weights_decay", plotflag, s, nSteps, inputs, test_inputs, test_times)
end
    
plotflag = true
main_mnist(plotflag, copy(standardSettings))

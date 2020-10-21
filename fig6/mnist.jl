include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

function main_mnist(plotflag, dt, s)
    print("Setting up...\n")
    nInitPatterns = 40000
    nPatterns = 120000
    nTestPatterns = 1000
    nNumbers = 3

    set!(s, "comment", "dt=$(dt)")
    set!(s, "showProgressBar", false)

    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", 5)
    set!(s, "updateInterval", 1)
    set!(s, "snapshotLogInterval", 5)

    l = s["presentationLength"]
    nSteps = nPatterns * l
    nInitSteps = nInitPatterns * l
    set!(s, "tempLogInterval", 2000 * l)

    set!(s, "n_x", 16^2 )# number x neurons
    set!(s, "n_z", 9) # number z neurons
     
    startingweights = map(x -> exp(max(0,0.3*x-0.2)) - 1, randn(9,16^2))

    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigma", true)
    set!(s, "fixedFinalSigmaValue", sqrt(0.3))
    set!(s, "learningRateSigma", 1e-6)

    set!(s, "localLearning_xz", false)
    set!(s, "learnedInhibition", true)
    set!(s, "reparametrizeBias", true)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", 0.015)

    inputs = create_mnist_inputs(nInitPatterns + nPatterns, s, nNumbers=nNumbers)
    test_inputs = create_mnist_test(nTestPatterns, s, nNumbers=nNumbers)
    test_times = [nInitSteps;collect(0:div(nSteps, 5):nSteps).+nInitSteps]
    
    s["paramChangeDict"]["learningRateDecoder"] =
        Dict(1          => 0.5e-5,
             200000 * l => 0.5e-5)
    s["paramChangeDict"]["learningRateFeedForward"] =
        Dict(1          => 0.0,    
             nInitSteps => 0.5e-5)
    s["paramChangeDict"]["learningRateHomeostaticBias"] =
        Dict(1          => 0.5e-2,
             nInitSteps => 0.5e-2)
    s["paramChangeDict"]["learningRateInhibitoryRecurrent"] =
        Dict(1          => 1.0e-5,
             nInitSteps => 1.0e-5)

    return main("mnist_random_start", plotflag, s, nSteps, inputs, test_inputs, test_times, 
                startingweights = startingweights)
end
    
plotflag = true

dt = 3.0

main_mnist(plotflag, dt, copy(standardSettings))

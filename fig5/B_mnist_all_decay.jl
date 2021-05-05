include("../src/all_weights_decay/settings.jl")
include("../src/all_weights_decay/main.jl")

function main_mnist(plotflag, dt, s)
    print("Setting up...\n")
    nInitPatterns = 60000
    nInhibitionPatterns = 30000
    nFFPatterns = 120000
    nTestPatterns = 1000
    nNumbers = 3

    set!(s, "comment", "dt=$(dt)")
    set!(s, "showProgressBar", true)

    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", 5)
    set!(s, "updateInterval", 5)
    set!(s, "snapshotLogInterval", 5)

    l = s["presentationLength"]
    nInitSteps = nInitPatterns * l
    nInhibitionSteps = (nInitPatterns + nInhibitionPatterns) * l
    nSteps = (nInitPatterns + nInhibitionPatterns + nFFPatterns) * l
    set!(s, "tempLogInterval", 2000 * l)

    set!(s, "n_x", 16^2 )# number x neurons
    set!(s, "n_z", 9) # number z neurons
    
    startingweights = map(x -> exp(max(0,0.3*x-0.2)) - 0.99, randn(9,16^2))
    startinginhibition = - map(x -> 0.0, randn(9,16^2,9))
    for j in 1:9
        for i in 1:16^2
            startinginhibition[j,i,j] = - startingweights[j,i]^2
        end
    end

    set!(s, "learnedSigma", false)
    set!(s, "initialSigma", sqrt(0.1))

    set!(s, "learnedInhibition", true)
    set!(s, "reparametrizeBias", true)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", 0.015)

    lambda = 0.03
    set!(s, "lambdaConstraint", lambda)
    set!(s, "comment", "lambda=$(lambda)")
    
    inputs = create_mnist_inputs(nInitPatterns + nInhibitionPatterns + nFFPatterns, s, nNumbers=nNumbers)
    test_inputs = create_mnist_test(nTestPatterns, s, nNumbers=nNumbers)
    test_times = [nInitSteps,nInhibitionSteps,nSteps]
    
    s["paramChangeDict"]["learningRateDecoder"] =
        Dict(1          => 0.5e-5)
    s["paramChangeDict"]["learningRateFeedForward"] =
        Dict(1          => 0.0,    
             nInitSteps => 0.0,    
             nInhibitionSteps => 1.5e-5)
    s["paramChangeDict"]["learningRateHomeostaticBias"] =
        Dict(1          => 0.7e-2,
             nInitSteps => 0.7e-2,    
             nInhibitionSteps => 0.7e-2)
    s["paramChangeDict"]["learningRateInhibitoryRecurrent"] =
        Dict(1          => 0.0,
             nInitSteps => 2.0e-5,    
             nInhibitionSteps => 3.0e-5)

    return main("Fig5B_mnist_all_weights_decay", plotflag, s, nSteps, inputs, test_inputs, test_times, 
                startingweights = startingweights, startinginhibition = startinginhibition)
end
    
plotflag = true

dt = 3.0

main_mnist(plotflag, dt, copy(standardSettings))

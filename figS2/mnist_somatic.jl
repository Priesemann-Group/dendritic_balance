include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

function main_mnist_local(plotflag, s)
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

    set!(s, "learningRateFeedForward", 0.5e-5)
    set!(s, "learningRateInhibitoryRecurrent", 1e-5)    
    set!(s, "learningRateHomeostaticBias", 0.5e-2)
    set!(s, "learningRateDecoder", 0.5e-5)
    

    set!(s, "initialSigma", sqrt(0.1))

    set!(s, "hebbianLearning_xz", true)
    set!(s, "learnedInhibition", true)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedSigma", false)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", 0.02)

    inputs = create_mnist_inputs(nPatterns, s, nNumbers=nNumbers)
    test_inputs = create_mnist_test(nTestPatterns, s, nNumbers=nNumbers)
    test_times = collect(0:div(nSteps, 5):nSteps)

    return main("S2_mnist_somatic", plotflag, s, nSteps, inputs, test_inputs, test_times)
end

plotflag = true
main_mnist_local(plotflag, copy(standardSettings))

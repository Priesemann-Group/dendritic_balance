include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

using Statistics: var
using DelimitedFiles
using Random

function main_bars(plotflag, s, corr, noise, id)
    print("Setting up...\n")
    nPatterns = 1000000
    nTestPatterns = 500
    
    set!(s, "showProgressBar", false)
    
    dt = 1.0
    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", Int(5.0/dt))
    set!(s, "updateInterval", 1) 
    set!(s, "snapshotLogInterval", Int(3.0/dt))

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 50000 * l)

    set!(s, "n_x", 8^2) # number x neurons
    set!(s, "n_z", 16) # number z neurons

    set!(s, "learningRateFeedForward", 5.0e-5)
    set!(s, "learningRateDecoder", 5.0e-5)
    set!(s, "learningRateInhibitoryRecurrent", 10.0e-5)
    set!(s, "learningRateHomeostaticBias", 1.0e-2)
    set!(s, "learningRateSigma", 7.0e-8)
    

    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigma", true)
    set!(s, "fixedFinalSigmaValue", sqrt(0.1))

    set!(s, "localLearning_xz", false)
    set!(s, "reparametrizeBias", true)
    set!(s, "learnedInhibition", false)
    set!(s, "homeostaticBiases", true)
    set!(s, "rho", 0.015)

    set!(s, "inputNoise", noise)

    inputs, _ = create_bar_inputs(nPatterns, s, corr, 0.0)
    test_inputs, labels = create_bar_inputs(nTestPatterns, s, corr, 0.0)
    test_times = [1, div(nSteps,2), nSteps-1]

    return main("Fig4_bars", plotflag, s, nSteps, inputs, test_inputs, test_times, name="$(id)_$(corr)")
end

plotflag = true
id = parse(Int,ARGS[1])
corr = 0.05 * floor((id-1) / (50))
noise = 0.0
net = main_bars(plotflag, copy(standardSettings), corr, noise, id)
loss = net.log.temp["test_decoder_loss"][end]
mkpath("../../bars/logs/")
file = open("../../bars/logs/results$id.txt", "w")
writedlm(file, [corr, noise, loss]')
close(file)

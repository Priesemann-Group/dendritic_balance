include("../src/analytic_speech/settings.jl")
include("../src/analytic_speech/main.jl")

function main_speech(plotflag, dt, oneSpikePerTimestep, s)
    print("Setting up...\n")
    nPatterns = 1000000
    nTestPatterns = 2000
    
    set!(s, "comment", "dt=$(dt)_onespike=$(oneSpikePerTimestep)")
    set!(s, "showProgressBar", true)

    set!(s, "splitPosNegInput", true)
    
    set!(s, "dt", dt)
    set!(s, "presentationLength", Int(floor(5.0 / s["dt"]))) # 5 ms means sampling rate of 200Hz audio
    set!(s, "fadeLength", 1.0) # completely interpolate signal
    set!(s, "tempLogSampleInterval", 5)
    set!(s, "updateInterval", 1)
    set!(s, "snapshotLogInterval", 5)
    set!(s, "snapshotNumberTestImages", 1000)

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", div(nPatterns, 30) * l)

    set!(s, "n_x", 25) # number x neurons
    set!(s, "n_z", 100) # number z neurons

    speed = 0.7

    set!(s, "learningRateFeedForward", speed*7e-5)
    set!(s, "learningRateInhibitoryRecurrent", speed*4e-4)
    set!(s, "learningRateHomeostaticBias", speed*2e-2)
    set!(s, "learningRateDecoder", speed*7e-5)

    set!(s, "initialSigma", sqrt(0.2))

    set!(s, "hebbianLearning_xz", false)
    set!(s, "learnedInhibition", true)
    set!(s, "homeostaticBiases", true)
    
    set!(s, "weightVariance", 0.01)
    set!(s, "weightMean", 0.05)

    set!(s, "rho", 0.005)

    set!(s, "oneSpikePerTimestep", oneSpikePerTimestep)
    
    set!(s, "unwhitenedInput", true)
    Random.seed!(10)
    inputs = get_speech_data(nPatterns, s, whiten=false)
    test_inputs = get_speech_data(nTestPatterns, s, whiten=false)
    test_times = collect(1:div(nSteps-1, 5):nSteps)

    return main("S11_speech", plotflag, s, nSteps, inputs, test_inputs, test_times)
end
    
plotflag = true

dt = 0.5 # [ms]

main_speech(plotflag, dt, true, copy(standardSettings))
main_speech(plotflag, dt, false, copy(standardSettings))

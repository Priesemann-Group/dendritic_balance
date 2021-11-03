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

    set!(s, "n_x", 50) # number x neurons
    set!(s, "n_z", 100) # number z neurons

    speed = 0.7

    set!(s, "learningRateFeedForward", speed*3e-4)
    
    set!(s, "learningRateHomeostaticBias", speed*2e-2)
    set!(s, "learningRateDecoder", speed*3e-4)

    set!(s, "initialSigma", sqrt(0.2))

    set!(s, "hebbianLearning_xz", false)
    set!(s, "learnedInhibition", false)
    set!(s, "homeostaticBiases", true)
    
    set!(s, "weightVariance", 0.01)
    set!(s, "weightMean", 0.05)

    set!(s, "rho", 0.005)

    set!(s, "oneSpikePerTimestep", oneSpikePerTimestep)
    
    Random.seed!(10)
    inputs, W = get_speech_data(nPatterns, s, whiten=true)
    test_inputs, _ = get_speech_data(nTestPatterns, s, whiten=true, whitening_matrix=W)
    test_times = collect(1:div(nSteps-1, 5):nSteps)
    
    s["whitening_matrix"] = W

    return main("Fig5_speech", plotflag, s, nSteps, inputs, test_inputs, test_times)
end
    
plotflag = true

dt = 0.05 # [ms]

main_speech(plotflag, dt, true, copy(standardSettings))
main_speech(plotflag, dt, false, copy(standardSettings))

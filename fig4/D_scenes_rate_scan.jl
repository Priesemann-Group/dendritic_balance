include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

using Profile
import Random

function main_scenes(plotflag, s, m, rate, seed=1234)
    print("Setting up...\n")
    nPatterns = 1500000
    nTestPatterns = 500
    imageDim = 8
    scale = 0.5

    dt = 0.1
    set!(s, "comment", "poprate=$(m*rate)_m=$(m)_rate=$(rate)")

    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", Int(5.0/dt))
    set!(s, "updateInterval", Int(round(0.2/dt)))
    set!(s, "snapshotLogInterval", Int(3.0/dt))

    # mandatory for scenes
    set!(s, "splitPosNegInput", true)

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 15000 * l)

    set!(s, "n_x", 2*imageDim^2) # number x neurons
    set!(s, "n_z", m) # number z neurons
    set!(s, "weightVariance", 0.0)

    set!(s, "learnedSigma", true)
    set!(s, "initialSigma", 1.0)
    set!(s, "fixedFinalSigma", true)
    set!(s, "fixedFinalSigmaValue", sqrt(0.13))

    set!(s, "localLearning_xz", false)
    set!(s, "reparametrizeBias", false)
    set!(s, "learnedInhibition", false)
    set!(s, "homeostaticBiases", true)

    set!(s, "rho", rate)

    set!(s, "learningRateSigma", 7e-8)

    s["paramChangeDict"]["learningRateDecoder"] =
        Dict(1          => 4e-5,
             200000 * l => 3e-5)
    s["paramChangeDict"]["learningRateFeedForward"] =
        Dict(1          => 4e-5,    
             200000 * l => 3e-5)
    s["paramChangeDict"]["learningRateHomeostaticBias"] =
        Dict(1          => 6e-3,
             200000 * l => 4e-3)
    s["paramChangeDict"]["learningRateInhibitoryRecurrent"] =
        Dict(1          => 10e-5,
             200000 * l => 7e-5)

    Random.seed!(seed)
    inputs = get_natural_scenes(nPatterns, s, imageDim, scale=scale)
    Random.seed!(seed)
    test_inputs = get_natural_scenes(nTestPatterns, s, imageDim, scale=scale)

    test_times = collect(0:div(nSteps, 10):nSteps)

    return main("Fig4_scenes_rate_scan", plotflag, s, nSteps, inputs, test_inputs, test_times)
end

plotflag = true
id = parse(Int,ARGS[1])
poprates = [1.0]
rates = collect(0.005:0.0025:0.025)

cs = CartesianIndices((1:length(poprates), 1:length(rates)))
i, j = Tuple(cs[id])

m = Int(round(poprates[i] / rates[j]))
main_scenes(plotflag, copy(standardSettings), m, rates[j])


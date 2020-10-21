include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

using Profile
import Random

function main_scenes(plotflag, s, timestep, noise, seed=1234)
    print("Setting up...\n")
    nPatterns = 1500000
    nTestPatterns = 500
    imageDim = 8
    scale = 0.5

    m = 200
    rate = 1.0/200
    dt = timestep
    set!(s, "comment", "dt=$(dt)_noise=$(noise)")

    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", max(1, Int(round(5.0/dt))))
    set!(s, "updateInterval", 1)
    set!(s, "snapshotLogInterval", max(1,Int(round(3.0/dt))))

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
    set!(s, "fixedFinalSigmaValue", sqrt(noise))

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

    return main("scenes_timestep_scan", plotflag, s, nSteps, inputs, test_inputs, test_times)
end

plotflag = true
id = parse(Int,ARGS[1])
timesteps = [collect(0.1:0.025:0.3); [0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0]]
noises = [0.13,0.15,0.2,0.3]

cs = CartesianIndices((1:length(timesteps), 1:length(noises)))
i, j = Tuple(cs[id])

main_scenes(plotflag, copy(standardSettings), timesteps[i], noises[j])


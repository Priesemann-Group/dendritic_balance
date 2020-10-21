
standardSettings = Dict{String,Any}()

_dt = 1.0 # [ms]/[step]
_tau = 10.0 # [ms]
_presentationLength = 100 # [ms]

standardSettings["showProgressBar"] = false

standardSettings["comment"] = ""

### Generating the dataset
# 100 ms presentation length in steps
standardSettings["presentationLength"] = Int(_presentationLength / _dt) # [step]
standardSettings["stimulusStrengthOn"] = 1.0
standardSettings["stimulusStrengthOff"] = 0.0
# represent positive and negative part of input
# in two neural populations (for noise and scenes)
standardSettings["splitPosNegInput"] = false
# changes nonlinearity for linear-Nonlinear Model
# used in noise and scenes
standardSettings["preprocessingNonlinearityShift"] = 0.8
standardSettings["preprocessingNonlinearityScale"] = 3.2
# 'fades' between image-presentations, denotes fraction of fade to constant
standardSettings["fadeLength"] = 0.3
standardSettings["inputNoise"] = 0.0


### Network architecture
# number x neurons
standardSettings["n_x"] = NaN
# number z neurons
standardSettings["n_z"] = NaN

### Neuronal dynamics
# simulation time step - if this is changed later in the simulation,
# use 'set_dt'
standardSettings["dt"] = _dt
# time constant of PSP decay, 10ms
standardSettings["kernelTau"] = _tau / _dt # [step]
standardSettings["kernelLength"] = Int(5 * Int(ceil(standardSettings["kernelTau"])))
standardSettings["kernel"] = [exp(-t/standardSettings["kernelTau"])
                              for t in 0:standardSettings["kernelLength"]-1]
# target rate
standardSettings["rho"] = 0.02 # 1/[ms]

### For generating the random initial params
standardSettings["weightVariance"] = 0.0
standardSettings["weightMean"] = 0.0
# The initial sigma of the gaussian
standardSettings["initialSigma"] = 1.0


### Learning settings
standardSettings["learningRateFeedForward"] = 0.00006
standardSettings["learningRateRecurrent"] = 0.00006
standardSettings["learningRateInhibitoryRecurrent"] = 0.001
standardSettings["learningRateBias"] = 0.00001
standardSettings["learningRateHomeostaticBias"] = 0.001
standardSettings["learningRateSigma"] = 5*1e-6
standardSettings["learningRateRates"] = 5*1e-6
standardSettings["learningRateDecoder"] = 5*1e-4
# Sometimes the model sigma is better to be higher than
# according to the model. The factor is there to prevent
# under-estimation of sigma which can lead to bad model
# performance. This is the factor to learn α⋅σ² instead of σ².
standardSettings["sigmaLearningOffset"] = 2.0
standardSettings["learnedSigma"] = true
# sigma approaches α
standardSettings["fixedFinalSigma"] = false
# If "fixedFinalSigma" sigma will exponentially decay to this
standardSettings["fixedFinalSigmaValue"] = 0.1
# apply local learning rule to xz-weights
standardSettings["localLearning_xz"] = false
# Reparametrize bias so that it scales with the precision
standardSettings["reparametrizeBias"] = false
# learn homeostatic biases instead of model biases
standardSettings["homeostaticBiases"] = false
# learn inhibitory zz-weights instead of using perfect inhibition
standardSettings["learnedInhibition"] = false
# update only every nth timestep
standardSettings["updateInterval"] = 1

### Online-updates of parameters
# First key is name of parameter, second is time of change
# So to change "learningRateSigma" at t=100 to 0.1:
# s["paramChangeDict"]["learningRateSigma"] = Dict(100 => 0.1)
standardSettings["paramChangeDict"] = Dict{String, Dict{Int64, Any}}()


### Logging settings
# Length of averaging window for templog
standardSettings["tempLogInterval"] = 1000
# Sample performance every nth timestep
standardSettings["tempLogSampleInterval"] = 1
# Save only every nth timestep in the snapshot to save space
standardSettings["snapshotLogInterval"] = 1
standardSettings["testWeightsNecessity"] = false


""" Save way of changing settings, so typos are noticed."""
function set!(s::Dict{String, Any}, key::String, value::Any)
    @assert (key in keys(s)) "Can't set setting. Key $key not valid."
    if (key == "dt")
        set_dt(s, value)
    else
        s[key] = value
    end
end

function set_dt(s::Dict{String, Any}, dt::Any)
    s["dt"] = dt
    s["presentationLength"] = Int(_presentationLength / dt)
    s["kernelTau"] = _tau / dt # [step]
    s["kernelLength"] = Int(5 * s["kernelTau"])
    s["kernel"] = [exp(-t/s["kernelTau"])
                    for t in 0:s["kernelLength"]-1]
end

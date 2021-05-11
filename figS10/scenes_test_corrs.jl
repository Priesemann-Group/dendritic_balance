include("../src/main.jl")

using BSON
using Random
using Statistics


function shuffle_square!(W::Array{Float64,2})
    n = size(W,1)
    for i in 1:1000*length(W)
        j = rand(1:n)
        k = rand(filter!(e->e≠j,collect(1:n)))
        l = rand(1:n)
        m = rand(filter!(e->e≠l,collect(1:n)))
        temp = W[j,k]
        W[j,k] = W[l,m]
        W[l,m] = temp
    end
end

function shuffle!(W::Array{Float64,2})
    n1 = size(W,1)
    n2 = size(W,2)
    for i in 1:100*length(W)
        j = rand(1:n1)
        k = rand(1:n2)
        l = rand(1:n1)
        m = rand(1:n2)
        temp = W[j,k]
        W[j,k] = W[l,m]
        W[l,m] = temp
    end
end


function adapt_biases(net)
    nPatterns = 20000
    imageDim = 8
    scale = 0.5
    seed = 1234

    s = net.log.settings
    set!(s, "tempLogInterval", typemax(Int))
    set!(s, "learningRateFeedForward", 0.0)
    set!(s, "learningRateHomeostaticBias", 5e-4)
    set!(s, "learningRateDecoder", 5e-4)
    set!(s, "learningRateInhibitoryRecurrent", 0.0)
    set!(s, "showProgressBar", true)
    
    Random.seed!(seed)
    inputs = get_natural_scenes(nPatterns, s, imageDim, scale=scale)

    run_net(net, inputs, Array{Float64,2}(undef, 0, 2), Array{Int}([]), "", s)
end

function test_net()
    @assert (length(ARGS) > 0) "Please provide the filename (.bson) of the net you want to run."

    file = ARGS[1]
    runfolder = join(split(file,"/")[1:end-1],"/")

    dict = BSON.load("$(file)")
    net = dict[:net]


    println("Setting up...")
    nPatterns = 1
    nTestPatterns = 500
    scale = 0.5
    plotflag = true

    s = net.log.settings
    s["initialSigma"] = s["fixedFinalSigmaValue"]
    l = s["presentationLength"]
    nSteps = nPatterns * l
    imageDim = Int(sqrt(div(net.n_x, 2)))
    interval = net.log.settings["tempLogInterval"]

    seed = 1234
    Random.seed!(seed)
    test_input = get_natural_scenes(nTestPatterns, net.log.settings, imageDim, scale=scale)

    println("Creating snapshot for $(file), wrong everything")
    l = create_log(nSteps, s)
    net2 = create_net(s, l)
    net2.xz_weights .= copy(net.xz_weights)
    net2.zz_weights .= copy(net.zz_weights)
    shuffle_square!(net2.zz_weights)
    net2.log.settings["learnedInhibition"] = true
    adapt_biases(net2)
    net2.log.t = -1
    take_snapshot(net2, net2.log, test_input, net2.log.settings)
    filter!(sn->sn.t≠-1,net.log.snapshots) 
    push!(net.log.snapshots, net2.log.snapshots[end])

    write_status(runfolder, net, net.log)
    println(length(net.log.snapshots))
end

test_net()

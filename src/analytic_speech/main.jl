include("settings.jl")
include("utils.jl")
include("../input_generation/imageprocessing.jl")
include("../input_generation/inputs.jl")
include("log.jl")
include("neuron.jl")
include("net.jl")
include("decoder.jl")
include("connections.jl")
include("plasticity.jl")
include("templog.jl")
include("snapshot.jl")
include("plot_functions.jl")
include("evaluation.jl")
include("decoder.jl")
import Random

""" Runs network on dataset and creates logs and plots.

 - project: Name of the project (determines the name of the save folder)
 - plotflag: If true, at the end an overview of the network performance is plotted
 - s: Settings used for the run
 - nSteps: Number of simulated steps
 - inputs: (nPatterns, n_x)-Matrix with input strengths.
 - test_input: (:, n_x)-Matrix with test-inputs.
 - test_times: Specifies at what times the snapshots are taken.

 - name: Name of the run (for saving), is date by default
 - basefolder: Location to save the results
 - startingweights: Matrix (n_x,n_z) of weights used at the start
"""
function main(project::String, plotflag::Bool, s::Dict{String,Any}, nSteps::Int,
    inputs::Array{Float64,2}, test_inputs::Array{Float64,2}, test_times::Array{Int,1};
    name::String = string(Dates.now()), seed::Int = 87294, basefolder = "../../", 
    startingweights = nothing)::Net

    Random.seed!(seed)

    log = create_log(nSteps, s)
    net = create_net(s, log)

    if !isnothing(startingweights)
        net.xz_weights = startingweights
        if log.settings["learnedInhibition"]
            net.zz_weights = get_inhibition_weights(Matrix(net.xz_weights'), 1.0)
        end
    end
    
    if s["comment"] != ""
        name *= "_" * s["comment"]
    end
    save_loc = basefolder * project * "/logs/" * name
    mkpath(save_loc)

    print("Running net $(name)...\n")
    run_net(net, inputs, test_inputs, test_times, save_loc, s)

    print("Saving log...\n")
    write_status(save_loc, net, log)

    if (plotflag)
        try
            print("Plotting...\n")
            folder = save_loc * "/plots"
            rm(folder, recursive=true, force=true)
            mkpath(folder)
            plot_net(net, folder)
        catch e
            print("Error while plotting:\n")
            bt = catch_backtrace()
            msg = sprint(showerror, e, bt)
            println(msg)
        end
    end

    print("Done.\n")
    return net
end

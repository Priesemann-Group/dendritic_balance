using ImageMagick
include("../src/log.jl")
include("../src/neuron.jl")
include("../src/net.jl")
include("../src/snapshot.jl")
include("../src/utils.jl")
include("plot_functions.jl")
include("../input_generation/imageprocessing.jl")
include("../src/connections.jl")

using BSON

@assert (length(ARGS) > 0) "Please provide the name of the folder you want to plot."

file = ARGS[1]
runfolder = join(split(file,"/")[1:end-1],"/")

print("Plotting " * split(runfolder,"/")[end] * "\n")
dict = BSON.load(file)
plotfolder = "$(runfolder)/plots/"
rm(plotfolder, recursive=true, force=true)
mkdir(plotfolder)
plot_net(dict[:net], plotfolder)

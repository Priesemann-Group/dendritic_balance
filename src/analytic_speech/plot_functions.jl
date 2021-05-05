using Statistics: mean
import JSON
import PyPlot
using Plots
pyplot()

""" Plots net into 'name'. Folder 'name' has to exist! """
function plot_net(net::Net, name::String)
    folder = name * "/"

    open(folder * "settings.json","w") do f
        json_string = JSON.json(net.log.settings, 4)
        write(f, json_string)
    end

    make_plots(net, folder)
end

""" Folder 'name' has to exist! """
function plot_patterns(patterns::Array{Float64, 2}, name::String, s::Dict{String,Any})
    folder = name * "/patterns/"

    splitPosNeg = s["splitPosNegInput"]::Bool

    mkdir(folder)
    nPatterns = min(64,size(patterns)[1])
    n = size(patterns)[2]
    nImg = splitPosNeg ? div(n, 2) : n
    sidelength = Int(sqrt(nImg))
    nCols = Int(ceil(sqrt(nPatterns)))

    img = zeros(nCols*sidelength,nCols*sidelength)
    for i in 1:nPatterns
        pattern = patterns[i,:]
        if splitPosNeg
            pattern = (pattern[1:nImg] - pattern[nImg+1:end])
        end
        col = ((i-1)%nCols)
        row = div(i-1,nCols)
        img[col*sidelength+1:(col+1)*sidelength, row*sidelength+1:(row+1)*sidelength] =
            reshape(pattern,sidelength,sidelength)
    end
    img = separate_subimages(img, sidelength, 1)
    PyPlot.clf()
    PyPlot.imshow(img',cmap="Greys")
    ax = PyPlot.gca()
    ax.axis("off")
    PyPlot.colorbar()
    PyPlot.savefig(folder * "patterns.svg")
    PyPlot.close_figs()
end

function make_plots(net::Net, folder::String)
    log = net.log
    dt = net.log.settings["dt"]

    # plot temporally saved variables
    for key in keys(log.temp)
        if (key != "t")
            plot_temp(log.temp[key], dt .* log.temp["t"], folder * "temp_" * key)
        end
    end

    # plot snapshots
    for i in 1:length(log.snapshots)
        print("Plotting snapshot $i\n")
        plot_snapshot(net, log.snapshots[i], folder, log.settings)
    end
end

function plot_snapshot(net::Net, snapshot::Snapshot, folder::String, s::Dict{String,Any})
    t = snapshot.t
    ap = ""
    try
        mkdir(folder * "snapshot-$t")
    catch e
        try 
            mkdir(folder * "snapshot2-$t")
            ap = "2"
        catch e
        end
    end
    plot_output(snapshot.z_outputs[:,1:min(4,s["n_z"])], folder * "snapshot$ap-$t/output")
    plot_spikes(snapshot.z_spikes, folder * "snapshot$ap-$t/z_spikes", s)
    plot_xz_weights(snapshot.xz_weights, folder * "snapshot$ap-$t/xz_weights", s)
    plot_zz_weights(snapshot.zz_weights, folder * "snapshot$ap-$t/zz_weights")
    plot_biases(snapshot.z_biases, folder * "snapshot$ap-$t/z_biases")
    plot_reconstructions(snapshot.reconstruction_means, folder * "snapshot$ap-$t/", s)
    plot_reconstructions(snapshot.reconstruction_vars, folder * "snapshot$ap-$t/", s, "_var")
    plot_reconstruction_comparison(snapshot.reconstructions, snapshot.x_outputs, folder * "snapshot$ap-$t/", s)
    plot_decoder_weights(snapshot.decoder_weights, folder * "snapshot$ap-$t/decoder_weights", s)
    w = get_inhibition_weights(net.log.decoder.D, snapshot.xz_weights, snapshot.sigma_2)
    plot_zz_weights(w, folder * "snapshot$ap-$t/zz_weights_perfect_inhibition")
end

function plot_temp(temp, ts, name::String)
    plot(ts, temp, linewidth=1.5, grid=false, legend=false)
    savefig(name * ".svg")
end

function plot_reconstructions(reconstructions::Array{Float64}, name::String,
    s::Dict{String,Any}, app::String="")

    splitPosNeg = s["splitPosNegInput"]::Bool

    nRecs =  min(64,size(reconstructions,1))
    n = size(reconstructions,2)
    nImg = splitPosNeg ? div(n, 2) : n
    sidelength = Int(sqrt(nImg))
    nCols = Int(ceil(sqrt(nRecs)))

    folder = name * "reconstructions/"
    mkpath(folder)

    img = zeros(nCols*sidelength,nCols*sidelength)
    for i in 1:nRecs
        rec = reconstructions[i,:]
        if splitPosNeg
            rec = (rec[1:nImg] - rec[nImg+1:end])
        end
        col = ((i-1)%nCols)
        row = div(i-1,nCols)
        img[col*sidelength+1:(col+1)*sidelength, row*sidelength+1:(row+1)*sidelength] =
            reshape(rec,sidelength,sidelength)
    end
    img = separate_subimages(img, sidelength, 1)
    PyPlot.clf()
    PyPlot.figure(figsize=(5.0,4.0))
    PyPlot.imshow(img',cmap="Greys")
    ax = PyPlot.gca()
    ax.axis("off")
    PyPlot.colorbar()
    PyPlot.savefig(folder * "reconstruction" * app * ".svg")
    PyPlot.close_figs()
end

function plot_reconstruction_comparison(recs::Array{Float64}, xs::Array{Float64},
    name::String, s::Dict{String,Any})

    dt = s["dt"]
    folder = name * "reconstructions/"
    mkpath(folder)
    tmax = min(Int(floor(1000*s["presentationLength"]/s["snapshotLogInterval"])),size(recs,1))
    vars = mean((recs - xs).^2, dims=1)
    inds1 = sortperm(reshape(vars,:),lt=(>))
    inds2 = sortperm(reshape(mean(xs, dims=1),:),lt=(>))
    if s["n_z"] > 4
        inds = cat(inds1[1:2], inds1[end-1:end], inds2[1:2], dims=1)
    else
        inds = 1:s["n_z"]
    end
    for i in 1:length(inds)
        plot(dt .* collect(1:tmax) .* s["snapshotLogInterval"],
             cat(recs[1:tmax,inds[i]], xs[1:tmax,inds[i]], dims=2),
             linewidth=1.5, grid=false, legend=false, size=(1200,400));
        savefig(folder * "comparison$i.svg")
    end
end

function plot_spikes(spikes::Array{Bool}, name::String, s::Dict{String,Any})
    dt = s["dt"]
    data = []
    isis = []
    for n in 1:size(spikes)[2]
        sub = []
        last_t = 1
        for t in 1:min(Int(floor(1000*s["presentationLength"]/s["snapshotLogInterval"])),size(spikes)[1])
            if spikes[t,n]
                push!(sub, dt*t*0.001*s["snapshotLogInterval"])
                push!(isis, dt*(t - last_t)*0.001*s["snapshotLogInterval"])
                last_t = t
            end
        end
        push!(data, sub)
    end
    PyPlot.clf()
    PyPlot.figure(figsize=(12.0,4.0))
    PyPlot.eventplot(data);
    PyPlot.savefig(name * ".svg")
    PyPlot.close_figs()

    PyPlot.clf()
    PyPlot.hist(isis, bins=30, range=(0.0,0.05));
    PyPlot.savefig(name * "_ISI.svg")
    PyPlot.close_figs()
end

function plot_output(output::Array{Float64}, name::String)
    plot(output,title="Outputs",xlabel="\$t\$",ylabel="\$z(t)\$");
    savefig(name * ".svg")
end

function plot_zz_weights(weights::Array{Float64}, name::String)
    PyPlot.clf()
    PyPlot.figure(figsize=(10.0,8.0))
    PyPlot.imshow(weights',cmap="inferno")
    PyPlot.colorbar()
    PyPlot.savefig(name * ".svg")
    PyPlot.close_figs()
end

function plot_xz_weights(weights::Array{Float64}, name::String, s::Dict{String,Any})
    splitPosNeg = s["splitPosNegInput"]::Bool

    n = size(weights)[2]
    m = size(weights)[1]
    nImg = splitPosNeg ? div(n, 2) : n
    sidelength = Int(sqrt(nImg))
    nCols = Int(ceil(sqrt(m)))

    img = zeros(nCols*sidelength,nCols*sidelength)
    for i in 1:m
        weight = weights[i,:]
        if splitPosNeg
            weight = (weight[1:nImg] - weight[nImg+1:end])
        end
        col = ((i-1)%nCols)
        row = div(i-1,nCols)
        img[col*sidelength+1:(col+1)*sidelength, row*sidelength+1:(row+1)*sidelength] =
            reshape(weight,sidelength,sidelength)
    end
    img = separate_subimages(img, sidelength, 1)
    PyPlot.clf()
    PyPlot.figure(figsize=(5.0,4.0))
    PyPlot.imshow(img',cmap="Greys")
    ax = PyPlot.gca()
    ax.axis("off")
    PyPlot.colorbar()
    PyPlot.savefig(name * ".svg")
    PyPlot.close_figs()
end

function plot_decoder_weights(weights::Array{Float64}, name::String, s::Dict{String,Any})
    splitPosNeg = s["splitPosNegInput"]::Bool

    n = size(weights)[1]
    m = size(weights)[2]
    nImg = splitPosNeg ? div(n, 2) : n
    sidelength = Int(sqrt(nImg))
    nCols = Int(ceil(sqrt(m)))

    img = zeros(nCols*sidelength,nCols*sidelength)
    for i in 1:m
        weight = weights[:,i]
        if splitPosNeg
            weight = (weight[1:nImg] - weight[nImg+1:end])
        end
        col = ((i-1)%nCols)
        row = div(i-1,nCols)
        img[col*sidelength+1:(col+1)*sidelength, row*sidelength+1:(row+1)*sidelength] =
            reshape(weight,sidelength,sidelength)
    end
    img = separate_subimages(img, sidelength, 1)
    PyPlot.clf()
    PyPlot.figure(figsize=(5.0,4.0))
    PyPlot.imshow(img',cmap="Greys")
    ax = PyPlot.gca()
    ax.axis("off")
    PyPlot.colorbar()
    PyPlot.savefig(name * ".svg")
    PyPlot.close_figs()
end

function plot_biases(biases, name::String)
    heatmap(reshape(biases,1,:));
    savefig(name * ".svg")
end

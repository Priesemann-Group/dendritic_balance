using HDF5, Dates

# It's here, otherwise circular dependecies arise
mutable struct Decoder
    D :: Array{Float64,2}
    var :: Float64
    biases :: Array{Float64,1}
end

mutable struct Log
    settings :: Dict{String,Any}
    nSteps :: Int
    t :: Int # Current time
    snapshots :: Array{} # Snapshots of network parameters and tests
    temp :: Dict{String, Any} # Other quantities (likelihood etc.)
    decoder :: Decoder
end

function create_log(nSteps::Int, s::Dict{String,Any})::Log
    return Log(s, nSteps, 1, [], Dict(), create_decoder(s))
end

function save_log(log::Log, name::String)
    rm("$name/log.h5", force=true)
    h5open("$name/log.h5", "w") do file
        # temp dictionary
        g = g_create(file, "temp")
        for (k,v) in log.temp
            g[k] = v
        end

        # also save (simple) settings as attribute
        for k in keys(log.settings)
            if isa(log.settings[k], Number) || isa(log.settings[k], Bool)
                attrs(g)[k] = log.settings[k]
            end
            if k == "whitening_matrix"
                g = g_create(file, "whitening_matrix")
                g["whitening_matrix"] = log.settings[k]
            end
        end

        # snapshots
        for i in 1:length(log.snapshots)
            g = g_create(file, "snapshot$i")
            sn = log.snapshots[i]
            keys = fieldnames(Snapshot)
            for k in 1:length(keys)
                field = getfield(sn,keys[k])

                if occursin("spikes", string(keys[k]))
                    if occursin("x_spikes", string(keys[k])) && !log.settings["spiking_x_neurons"]
                        # don't plot x neurons if not spiking
                        continue
                    else
                        # python hdf5 can't read bool arrays :(
                        field = bool_array_to_ocurrences(field, log.settings)
                    end
                end

                g[string(keys[k])] = field
            end
        end
    end
end

""" Python HDF5 doesn't know Bool-Arrays. This function converts them to 'sparse'
representation.
"""
function bool_array_to_ocurrences(array::Array{Bool,2}, s::Dict{String,Any})::Array{Float64,2}
    dt = s["dt"]
    len = maximum(sum(array, dims=1))
    res = -ones(len, size(array,2))
    for i in 1:size(array,2)
        c = 0
        for t in 1:size(array,1)
            if array[t,i]
                c += 1
                res[c,i] = dt*t
            end
        end
    end
    return res
end

using MLDatasets
#using FFTW
import ImageMagick
using Images
using LinearAlgebra
#using HCubature
using FileIO
using Plots
using ProgressMeter
using MAT
using Statistics

function create_random_inputs(nPatterns::Int, s::Dict{String,Any})::Array{Float64,2}
    n = s["n_x"]::Int64
    return rand(Float64, nPatterns, n)
end

function create_constant_inputs(nPatterns::Int, s::Dict{String,Any})::Array{Float64,2}
    n = s["n_x"]::Int64
    return ones(Float64, nPatterns, n)
end

function create_bar_inputs(nPatterns::Int, s::Dict{String,Any}, corr::Float64, 
    noise::Float64)::Tuple{Array{Float64,2},Array{Int,2}}
    
    n = s["n_x"]
    sidelength = Int(sqrt(n))
    @assert (n == sidelength^2) "n has to be square number!"

    strengthOn = s["stimulusStrengthOn"]::Float64
    strengthOff = s["stimulusStrengthOff"]::Float64

    input = ones(nPatterns, n) * strengthOff
    labels = zeros(Int, nPatterns, sidelength*2)
    randos1 = rand(1:2,nPatterns) # hor/vert
    randos2 = rand(1:sidelength,nPatterns) # #bar
    randos3 = rand(nPatterns) # for correlation between har/vert bar
    randos4 = rand(1:2,nPatterns)
    randos5 = rand(1:sidelength,nPatterns)
    for i in 1:nPatterns
        square = zeros(sidelength,sidelength)
        if (randos1[i] == 1)
            square[:, randos2[i]] .= strengthOn
            labels[i,randos2[i]] = 1
        else
            square[randos2[i], :] .= strengthOn
            labels[i,sidelength-1+randos2[i]] = 1
        end
        if (randos3[i] <= corr) # same bars are shown
            square[:, randos2[i]] .= strengthOn
            square[randos2[i], :] .= strengthOn
            labels[i,randos2[i]] = 1
            labels[i,sidelength-1+randos2[i]] = 1
        else
            if (randos4[i] == 1)
                square[:, randos5[i]] .= strengthOn
                labels[i,randos5[i]] = 1
            else
                square[randos5[i], :] .= strengthOn
                labels[i,sidelength-1+randos5[i]] = 1
            end
        end

        input[i,:] .= reshape(square,:)
    end
    return input + randn(nPatterns, n) * noise, labels
end

function create_mnist_inputs(nPatterns::Int, s::Dict{String,Any}; nNumbers::Int=10, scale::Float64=16/28)::Array{Float64,2}
    n = s["n_x"]::Int64
    sidelength = Int(sqrt(n))
    @assert (n == (28*scale)^2) "n has to be $(28*scale)^2!"

    strengthOn = s["stimulusStrengthOn"]::Float64
    strengthOff = s["stimulusStrengthOff"]::Float64

    train_x, train_y = MNIST.traindata()
    l = size(train_x,3)

    input = ones(nPatterns, n)
    skip = 0
    for i in 1:nPatterns
        while train_y[(i + skip - 1) % l + 1] >= nNumbers
            skip += 1
        end
        square = convert(Array{Float64,2}, train_x[:, :, (i + skip - 1) % l + 1])
        square .*= strengthOn - strengthOff
        square .+= strengthOff
        square = imresize(square, ratio=scale)
        input[i,:] .= reshape(square,:)
    end

    return input
end

function create_mnist_test(nPatterns::Int, s::Dict{String,Any}; nNumbers::Int=10, scale::Float64=16/28)::Array{Float64,2}
    n = s["n_x"]::Int64
    sidelength = Int(sqrt(n))
    @assert (n == (28*scale)^2) "n has to be $(28*scale)^2!"

    strengthOn = s["stimulusStrengthOn"]::Float64
    strengthOff = s["stimulusStrengthOff"]::Float64

    test_x, test_y = MNIST.testdata()
    l = size(test_x,3)

    input = ones(nPatterns, n)
    skip = 0
    for i in 1:nPatterns
        while test_y[(i + skip - 1) % l + 1] >= nNumbers
            skip += 1
        end
        square = convert(Array{Float64,2}, test_x[:, :, (i + skip - 1) % l + 1])
        square .*= strengthOn - strengthOff
        square .+= strengthOff
        square = imresize(square, ratio=scale)
        input[i,:] .= reshape(square,:)
    end

    return input
end

""" Loads images used in Olshausen (1997).

 - img_size: side-length of the square cut-outs
 - nPixelSteps: to augment the dataset moves window (high value mean long setup)"""
function get_natural_scenes(nPatterns::Int, s::Dict{String,Any}, img_size::Int=16;
    nPixelSteps::Int=5, scale::Float64=1.0)::Array{Float64,2}

    strengthOn = s["stimulusStrengthOn"]::Float64
    strengthOff = s["stimulusStrengthOff"]::Float64
    splitPosNeg = s["splitPosNegInput"]::Bool
    nlShift = 0.15
    s["preprocessingNonlinearityShift"] = nlShift
    nlScale = 16.0
    s["preprocessingNonlinearityScale"] = nlScale

    @assert splitPosNeg "For scenes only splitting negative and positive parts is implemented."

    sl = img_size
    nImg = sl^2
    n = splitPosNeg ? nImg*2 : nImg

    input_dir = @__DIR__
    dic = matread(input_dir * "/scenes/IMAGES.mat")
    imgs = dic["IMAGES"] # (512,512,10)
    imgs = permutedims(imgs, [3, 1, 2]) # (10,512,512)

    # rescale
    resolution = Int(scale*512)
    temp = zeros(10,resolution,resolution)
    for i in 1:10
        temp[i,:,:] = imresize(imgs[i,:,:], ratio=scale)
    end
    imgs = temp

    split_input = Array{Float64}(undef, 0, nImg)
    for i in 0:nPixelSteps
        for j in 0:nPixelSteps
            # -8 as we leave away the border
            splitsX = div(Int((512-8)*scale)-i, sl)
            splitsY = div(Int((512-8)*scale)-j, sl)
            temp = split_images(imgs[:,5+i:splitsX*sl,5+j:splitsY*sl], sl)
            split_input = cat(split_input, temp, dims=1)
        end
    end

    input = zeros(nPatterns, n)

    nl(x) = nonlinearity(x, nlShift, nlScale)
    #nl(x) = max(0,x)
    for i in 1:nPatterns
        ind = rand(1:size(split_input,1))
        temp = split_input[ind,:]
        img = vcat(temp, -temp)
        img = map(nl, img)
        input[i,:] .= img
    end

    return input .* strengthOn
end


""" Loads speech data used in Brendel et al (2020)."""
function get_speech_data(nPatterns::Int, s::Dict{String,Any}; whiten::Bool=false, 
    whitening_matrix=nothing)

    splitPosNeg = s["splitPosNegInput"]::Bool
    strengthOn = s["stimulusStrengthOn"]::Float64
    dt = s["dt"]::Float64 # 0.05 ms in original publication
    tau = s["kernelTau"]::Float64 # 12.5 ms in original publication (here in steps)

    input_dir = @__DIR__
    dic = matread(input_dir * "/speech/speech.mat")
    sp = dic["speech"] 
    # normalize data magnitude
    data = 1.5 * sp["data"] ./ maximum(sp["data"])  # (25,:)
    frequencies = sp["CF"] # (25)

    block_length = Int(floor(size(data,2) / 50))

    input = zeros(nPatterns, 25)
    
    ind = rand(1:(size(data,2)-block_length))
    for i in 1:nPatterns
        input[i,:] = data[:,ind]
        ind += 1
        if i % block_length == 0
            ind = rand(1:(size(data,2)-block_length))
        end
    end
    
    
    if whiten
        input .-= mean(input, dims=1)
        W, input = whiten_images_cholesky(input, whitening_matrix)
        if splitPosNeg
            input = hcat(input, -input)
            nl(x) = max(0,x)
            map!(nl, input, input)
        end
        return input .* strengthOn, W
    end

    return input .* strengthOn
end

function fade(inputs::SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
 old_inputs::SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}, 
 t::Float64, fadeTime::Float64)::Array{Float64,1}
    f(t) = min(1.0,max(0.0,t / fadeTime))
    return inputs .* f(t) + old_inputs .* (1.0 - f(t))
end

function fade_images(x_inputs::Array{Float64,2},t::Int,s::Dict{String,Any})::Array{Float64,1}
    l = s["presentationLength"]::Int
    fadeLength = s["fadeLength"]::Float64
    
    indImg = div(t - 1,l) + 1

    inp = view(x_inputs, indImg, :)
    old_inp = view(x_inputs, max(1, indImg-1), :)
    x = fade(inp, old_inp, (((t-1) % l) + 1) / l, fadeLength)

    return x
end

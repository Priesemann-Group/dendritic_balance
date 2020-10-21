using MultivariateStats
using DelimitedFiles: readdlm, writedlm

""" Separates subimages in image with border for plotting"""
function separate_subimages(X::Array{Float64, 2}, sub_length::Int, border_width::Int)
    bw = border_width
    splits = split_images(X, sub_length)
    s = size(splits)
    X_new = zeros(s[1], s[2]+2*bw, s[3]+2*bw)
    X_new .= minimum(X)
    X_new[:,bw+1:end-bw,bw+1:end-bw] = splits
    return stitch_images(X_new)
end

""" Splits image into square tiles of a given side-length for single image"""
function split_images(X::Array{Float64, 2}, split_length::Int)
    image_slX = size(X,1)
    image_slY = size(X,2)

    sl = split_length
    splitsX = div(image_slX, sl)
    splitsY = div(image_slY, sl)
    temp = zeros(splitsX * splitsY, sl, sl)

    i = 0
    for sx in 1:splitsX
        for sy in 1:splitsY
            i += 1
            cut = X[sl*(sx-1)+1:sl*sx, sl*(sy-1)+1:sl*sy]
            temp[i,:,:] = cut
        end
    end
    return temp
end

""" Splits images into square tiles of a given side-length for a dataset"""
function split_images(X::Array{Float64, 3}, split_length::Int)
    n = size(X,1)
    image_slX = size(X,2)
    image_slY = size(X,3)

    sl = split_length
    dim = sl^2
    splitsX = div(image_slX, sl)
    splitsY = div(image_slY, sl)
    temp = zeros(n * splitsX * splitsY, dim)

    i = 0
    for ind_im in 1:n
        arr = X[ind_im, :, :]
        for sx in 1:splitsX
            for sy in 1:splitsY
                i += 1
                cut = arr[sl*(sx-1)+1:sl*sx, sl*(sy-1)+1:sl*sy]
                temp[i,:] = reshape(cut, :)
            end
        end
    end
    return temp
end

""" Takes the tiles of n (image_dim,image_dim) images and stiches them together"""
function stitch_images(X::Array{Float64,2}, n::Int, image_dim::Int, split_length::Int)
    m = size(X,1)
    cut_dim = size(X,2)

    sl = split_length
    splits = div(image_dim, dim)
    temp = zeros(n, image_dim, image_dim)

    i = 0
    for ind_im in 1:n
        for sx in 1:splits
            for sy in 1:splits
                i += 1
                cut = reshape(X[i, :], (sl, sl))
                temp[ind_im, sl*(sx-1)+1:sl*sx, sl*(sy-1)+1:sl*sy] = cut
            end
        end
    end
    return temp
end

function stitch_images(X::Array{Float64,3})
    m = Int(sqrt(size(X,1)))
    sl = size(X,2)

    image_dim = m * sl
    temp = zeros(image_dim, image_dim)

    i = 0
    for sx in 1:m
        for sy in 1:m
            i += 1
            cut = X[i, :, :]
            temp[sl*(sx-1)+1:sl*sx, sl*(sy-1)+1:sl*sy] = cut
        end
    end
    return temp
end


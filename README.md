# Dendritic balance enables local learning of efficient representations in networks of spiking neurons

These files accompany the results obtained in *Dendritic balance enables local learning of efficient representations in networks of spiking neurons*. For further details please refer to the manuscript posted on [arxiv](https://arxiv.org/abs/2010.12395).

## How to recreate results

### Figure 3

Run `julia mnist_().jl`, where `()` is to be replaced to execute the desired file.

To plot results run `python plot_mnist.py ../../mnist/logs/() ../../mnist_somatic/logs/()`, where `()` is to be replaced with the desired folder.

### Figure 4

Run `julia bars_().jl $i`, where `$i` ranges from 1-1050. We recommend only using the analytic and somatic implementations, as the all weights implementations are slow.

To plot the overview results first concat the resulting losses into one file `cat results* > cresults.txt`, in the log folders.
Then run `python plot_bars.py`.

### Figure 5

Run `julia scenes_rate_scan.jl $i` and `julia scenes_rate_scan_somatic.jl $i`, where `$i` ranges from 1-9.

To plot the overview results run `python plot_scenes_comparison.py ../../scenes_rate_scan/logs/ ../../scenes_rate_scan_somatic/logs/`.

### Figure 6

#### A

Run `julia scenes_timestep_scan.jl $i` and `julia scenes_timestep_scan_somatic.jl $i`, where `$i` ranges from 1-72.

To plot the overview results run `python plot_scenes_comparison.py ../../scenes_timestep_scan/logs/ ../../scenes_timestep_scan_somatic/logs/`.

#### B

Same as in figure 3.

## Requirements

The results in this paper were created using `Julia 1.3.1` and `Python 3.6` with `matplotlib`, `numpy` and `h5py`.
To run the experiments using natural images, please download `IMAGES.mat` from http://www.rctn.org/bruno/sparsenet/ and place it into `src/input_generation/scenes/`.

### Julia Packages

```julia
pkgs = ["BSON",
"Dates",
"DelimitedFiles",
"FileIO",
"HDF5",
"ImageMagick",
"Images",
"InteractiveUtils",
"JSON",
"LinearAlgebra",
"MAT",
"MLDatasets",
"MultivariateStats",
"Plots",
"Profile",
"ProgressMeter",
"PyPlot",
"SparseArrays"]

using Pkg

Pkg.add(pkgs)
```

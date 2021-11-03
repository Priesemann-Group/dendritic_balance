# Dendritic balance enables local learning of efficient representations in networks of spiking neurons

These files accompany the results obtained in *Dendritic balance enables local learning of efficient representations in networks of spiking neurons*. For further details please refer to the manuscript posted on [arxiv](https://arxiv.org/abs/2010.12395).

## How to recreate results

### Figure 3

Run `julia mnist_().jl`, where `()` is to be replaced to execute the desired file.

To plot results run `python plot_mnist.py ../../Fig3_mnist_all_weights_decay/logs/() ../../Fig3_mnist_somatic/logs/()`, where `()` is to be replaced with the desired folder.

### Figure 4

#### B

Run `julia B_bars_().jl $i`, where `$i` ranges from 1-1050. We recommend only using the analytic and somatic implementations, as the all weights implementations are slow.

To plot the overview results first concat the resulting losses into one file `cat results* > cresults.txt`, in the log folders.
Then run `python B_plot_bars.py`.

#### D

Run `julia D_scenes_rate_scan.jl $i` and `julia D_scenes_rate_scan_somatic.jl $i`, where `$i` ranges from 1-9.

To plot the overview results run `python D_plot_scenes_comparison.py ../../Fig4_scenes_rate_scan/logs/ ../../Fig4_scenes_rate_scan_somatic/logs/`.

### Figure 5

#### A

Run `julia A_scenes_timestep_scan.jl $i` and `julia A_scenes_timestep_scan_somatic.jl $i`, where `$i` ranges from 1-72.

To plot the overview results run `python A_plot_scenes_comparison.py ../../Fig5_scenes_somatic_timestep_scan/logs/ ../../Fig5_scenes_somatic_timestep_scan_somatic/logs/`.

#### B

Same as in figure 3.


## Requirements

The results in this paper were created using `Julia 1.3.1` and `Python 3.6` with `matplotlib`, `numpy` and `h5py`.
To run the experiments using natural images, please download `IMAGES.mat` from http://www.rctn.org/bruno/sparsenet/ and place it into `src/input_generation/scenes/`.
To run the experiments using speech data, please download `speech.mat` from https://github.com/machenslab/spikes/tree/master/UnsupervisedLearning/Figure5 and place it into `src/input_generation/speech/`

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

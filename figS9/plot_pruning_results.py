
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil
import matplotlib.gridspec as gridspec
import matplotlib as mpl
sys.path.insert(0, '../src/')
from plot_utils import *

params = {
    #'text.latex.preamble': ['\\usepackage{gensymb}'],
    #'image.origin': 'lower',
    #'image.interpolation': 'nearest',
    #'image.cmap': 'gray',
    #'axes.grid': False,
    #'savefig.dpi': 150,  # to adjust notebook inline plot size
    #'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    #'axes.titlesize': 8,
    'font.size': 13, # was 10
    #'legend.fontsize': 10, # was 10
    #'xtick.labelsize': 8,
    #'ytick.labelsize': 8,
    #'text.usetex': True,
    #'figure.figsize': [3.39, 2.10],
    #'font.family': 'serif',
}
matplotlib.rcParams.update(params)

def main():
    ARGS = sys.argv
    if (len(ARGS) <= 2):
        print("Please provide two folders as argument (1. logs folder of pruning network 2. log folder of somatic reference run).")
        return

    pruning_dics = []
    
    for fl in os.listdir(ARGS[1]):
        subfolder = ARGS[1] + '/' + fl

        dic = h5py.File(subfolder + '/log.h5' , 'r')
        pruning_dics.append(dic)

    dic_somatic = h5py.File(ARGS[2] + '/log.h5' , 'r')

    plot_results(pruning_dics, dic_somatic)


def plot_results(pruning_dics, dic_somatic):

    pruningfractions = []
    performances = []

    for dic in pruning_dics:
        temp = dic[u'temp']
        pruningfraction = temp.attrs.get("pruningFraction") *100
        print(pruningfraction)
        pruningfractions.append(pruningfraction)
        performance = np.mean(temp.get('test_decoder_loss')[-5:])
        performances.append(performance)
        print(temp.get('test_decoder_loss')[()])

    
    performances = np.array(performances)
    pruningfractions = np.array(pruningfractions)

    inds = np.argsort(pruningfractions)
    performances = performances[inds]
    pruningfractions = pruningfractions[inds]

    temp = dic_somatic[u'temp']
    reference = np.mean(temp.get('test_decoder_loss')[-10:])

    # create figure
    scale = 0.6
    fig = plt.figure(figsize=(scale*4, scale*3))
    
    plt.hlines(reference, xmin=0.0, xmax=100, linestyles='dashed', color='orange', label="SB reference")
    plt.plot(pruningfractions, performances, marker='o', markersize=3, color='darkolivegreen', label="DB")
    plt.legend()
    
    plt.ylim(0.0, 0.06)
    plt.xlabel('% dendr. weights pruned')
    plt.ylabel('decoder loss')
    plt.tight_layout()
    
    plt.savefig("pruning.svg")


main()


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


def conf(x, n_boot=10000, cil=0.95):
    mn = np.median(x)
    samples = np.random.choice(x, size=(len(x), n_boot), replace=True)
    medians = np.sort(np.apply_along_axis(np.median, 0, samples))
    ind1 = int((1 - cil)/2*n_boot)
    ind2 = int((1 - (1 - cil)/2)*n_boot)
    return (mn - medians[ind1], medians[ind2] - mn)

def main():
    ARGS = sys.argv
    if (len(ARGS) <= 1):
        print("Please provide the natural scenes log folder (Fig 4) as argument.")
        return

    folder = ARGS[1]
    subfolders = os.listdir(folder) 

    dics = [h5py.File(folder + "/" + subfolder + "/log.h5", 'r') for subfolder in subfolders]
    
    n_zs = []
    for dic in dics:
        snapshots = [dic[k] for k in sorted(dic.keys(), key=natural_keys) if 'snapshot' in k]
        snapshot = snapshots[-1]
        z = snapshot.get('z_outputs')[()]
        n_z = z.shape[0]
        n_zs.append(n_z)
    
    n_zs, dics = zip(*sorted(zip(n_zs, dics)))

    corrs = []
    for dic in dics:
        snapshots = [dic[k] for k in sorted(dic.keys(), key=natural_keys) if 'snapshot' in k]
        snapshot = snapshots[-1]
        temp = dic[u'temp']
        
        z = snapshot.get('z_outputs')[()]
        n_z = z.shape[0]
        T = z.shape[1]
        corr = []
        for i in range(n_z):
            for j in range(i+1,n_z):
                c = 1/T * np.dot(z[i,:] - np.mean(z[i,:]), z[j,:] - np.mean(z[j,:])) / (np.std(z[i,:]) * np.std(z[j,:]))
                corr.append(c)
        corrs.append(np.array(corr))
        
    fig, axs = plt.subplots(figsize=(0.6*4,0.6*3))
   
    mn = [np.mean(corr) for corr in corrs]
    sd = np.array([conf(corr) for corr in corrs])
    print(sd)
    p1 = plt.errorbar(n_zs, mn, yerr = sd.T, capsize=2, elinewidth=0.7, color='black')
    plt.xlabel("# z-neurons")
    plt.ylabel("pearson corr.")
    axs.set_xlim(215,30)
    
    plt.tight_layout()
    plt.savefig("corrs.svg")


main()

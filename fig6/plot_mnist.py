
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
        print("Please provide two folders as argument (1. dendritic, 2. somatic).")
        return

    plot_folder = str("/".join(ARGS[1].split("/")[0:-1])) + "/"
    filename = ARGS[1] + "/log.h5"
    f = str(ARGS[1].split("/")[-1])

    filename2 = ARGS[2] + "/log.h5"

    print("Plotting " + filename)
    dic = h5py.File(filename, 'r')
    dic2 = h5py.File(filename2, 'r')
    folder = plot_folder + f + "/python_plots/"
    #shutil.rmtree(folder, ignore_errors=True)

    try:
        os.mkdir(folder)
    except:
        pass

    plot_net(dic, dic2, folder)


def plot_net(dic, dic2, folder):
    # plot temp
    temp = dic[u'temp']

    # how to get elements of hdf5
    snapshots = [dic[k] for k in dic.keys() if 'snapshot' in k]
    for snapshot in snapshots:
        element = snapshot.get('xz_weights')[()]

    plot_results(dic, dic2, folder)

def plot_results(dic, dic2, folder):
    snapshots = [dic[k] for k in sorted(dic.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    snapshots = [dic2[k] for k in sorted(dic2.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot2 = snapshots[-1]
    snapshotstart = snapshots[0]
    temp = dic[u'temp']
    temp2 = dic2[u'temp']

    n_imgs = 5**2
    img_min = 0

    dt = temp.attrs.get("dt") 
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / temp.attrs.get("snapshotLogInterval"))


    # create figure
    scale = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(scale*3,scale*2))
    #plt.axis('off')

    times = dt * temp.get('t') / 1000

    # decoder loss
    loss = temp.get('test_decoder_loss')
    loss2 = temp2.get('test_decoder_loss')
    ax.plot(times, loss2, color='orange')
    ax.plot(times, loss, color='darkolivegreen')
    ax.legend(['SB', 'DB'])
    ax.set_xlabel(r'$t$ $[s]$')
    ax.set_ylabel('decoder loss')
    ax.ticklabel_format(axis='x',style='sci', scilimits=(0,1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #xmin, xmax = ax.get_xlim()
    #ax.set_xticks(np.round(np.linspace(xmin, xmax, 3), 2))

    plt.tight_layout()
    plt.savefig("../../plots/mnist_random_start.svg")
    
    fig, axs = plt.subplots(2, 2, figsize=(scale*6,scale*3.4))
    
    #spikes
    dt = temp.attrs.get("dt") 
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / temp.attrs.get("snapshotLogInterval"))
    tmin = 100
    tmax = tmin + 300*pres_len
    
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin/2 and spikes[t,n]*interval<tmax/2:
                sub.append(spikes[t,n]*interval/1000) 
        data.append(sub)

    axs[0,0].eventplot(data, color='cornflowerblue')
    axs[0,0].set_xlabel(r'$t$ $[s]$')
    axs[0,0].set_ylabel('# neuron')
    
    spikes = snapshot2.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin/2 and spikes[t,n]*interval<tmax/2:
                sub.append(spikes[t,n]*interval/1000) 
        data.append(sub)

    axs[1,0].eventplot(data, color='cornflowerblue')
    axs[1,0].set_xlabel(r'$t$ $[s]$')
    axs[1,0].set_ylabel('# neuron')
    
    spikes = snapshotstart.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin/2 and spikes[t,n]*interval<tmax/2:
                sub.append(spikes[t,n]*interval/1000) 
        data.append(sub)

    axs[0,1].eventplot(data, color='cornflowerblue')
    axs[0,1].set_xlabel(r'$t$ $[s]$')
    axs[0,1].set_ylabel('# neuron')
    
    plt.tight_layout()
    plt.savefig("../../plots/mnist_random_start_spikes.svg")


main()

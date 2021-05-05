import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil
sys.path.insert(0, '../src/')
from plot_utils import *


def main():
    ARGS = sys.argv
    if (len(ARGS) <= 4):
        print("Please provide four folders as argument (1. dendritic analytic, 2. dendritic simultaneous, 3. dendritic slow, 4. dendritic decay, 5. somatic).")
        return
    
    folder_dendritic = ARGS[1] 
    folder_dendritic_sim = ARGS[2] 
    folder_dendritic_del = ARGS[3] 
    folder_dendritic_dec = ARGS[4] 
    folder_somatic = ARGS[5] 
    
    names = ["DB", "DB simultaneous", "DB slow", "DB decay", "SB"]
    folders = [folder_dendritic, folder_dendritic_sim, folder_dendritic_del, folder_dendritic_dec, folder_somatic]
    
    file = "/log.h5"

    print("Plotting...")

    losses = {}
    ts = {}
    weights = {}
    for (folder, name) in zip(folders, names):
        dic = h5py.File(folder + file, 'r')
        temp = dic[u'temp']
        meta = temp.attrs
        
        losses[name] = temp["test_decoder_loss"]

        snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]
        weights[name] = snapshots[-1].get('xz_weights')[()]

        dt = meta["dt"]
        ts[name] = temp["t"] * dt / 1000

    
    fig = plt.figure(figsize=(6,9))

    gs = fig.add_gridspec(3,2)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
           fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    # all learningcurves combined
    for name in names:
        print(name)
        axs[0].plot(ts[name], losses[name], label=name)
    #axs[0].set_ylim((0.005,0.03))
    axs[0].set_ylabel("decoder loss")
    axs[0].set_xlabel("t [s]")
    axs[0].legend()

    # compare weights
    i = 1
    for name in names:
        def sbplt(fields, k, name):
            imsize = int(np.sqrt(fields.shape[0]))
            fields = np.transpose(fields)
            img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=2)

            im = axs[k].imshow(img, cmap='Greys')
            colorbar(im)
            axs[k].axis('off')
            axs[k].set_title(name)
            fig.add_subplot(axs[k])

        
        sbplt(weights[name], i, name)
        i += 1

    plt.tight_layout()
    plt.savefig("bars_compare_all.svg", dpi=500)


main()


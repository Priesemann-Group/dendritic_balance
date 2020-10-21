import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil
sys.path.insert(0, '../src/')
from plot_utils import *


def main():
    ARGS = sys.argv
    if (len(ARGS) <= 2):
        print("Please provide two folders as argument (1. dendritic logs, 2. somatic logs).")
        return
    
    N = 1  # rates, vars
    M = 9 # neurons, dt
    skip = 0
    folders_dendritic = [ARGS[1] + s for s in os.listdir(ARGS[1])]
    folders_somatic = [ARGS[2] + s for s in os.listdir(ARGS[2])]
    
    folders_dendritic = sorted(folders_dendritic)[skip:skip+N*M]
    folders_somatic = sorted(folders_somatic)[skip:skip+N*M]
    file = "/log.h5"

    print("Plotting...")

    losses_dendritic = {}
    losses_somatic = {}
    weights_dendritic = {}
    weights_somatic = {}
    ts_dendritic = {}
    ts_somatic = {}
    for folder in folders_dendritic:
        dic = h5py.File(folder + file, 'r')
        temp = dic[u'temp']
        meta = temp.attrs
        
        key = (round(meta["n_z"] * meta["rho"] ,1), meta["rho"], meta["n_z"])
        losses_dendritic[key] = temp["test_decoder_loss"]

        snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]
        weights_dendritic[key] = snapshots[-1].get('xz_weights')[()]

        ts_dendritic[key] = temp["t"]

    for folder in folders_somatic:
        dic = h5py.File(folder + file, 'r')
        temp = dic[u'temp']
        meta = temp.attrs
        
        key = (round(meta["n_z"] * meta["rho"],1), meta["rho"], meta["n_z"])
        losses_somatic[key] = temp["test_decoder_loss"]

        snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]
        weights_somatic[key] = snapshots[-1].get('xz_weights')[()]

        ts_somatic[key] = temp["t"]
    
    # all learningcurves combined
    fig, axs = plt.subplots(int(np.sqrt(M)), int(np.sqrt(M)), figsize=(int(np.sqrt(M))*2,int(np.sqrt(M))*1.6))
    axs = np.reshape(axs,-1)
    i = 0
    for key in sorted(list(losses_dendritic.keys())):
        print(key)
        try:
            axs[i].plot(ts_dendritic[key], losses_dendritic[key], 
                    label=["dendritic"])
        except:
            continue
        try:
            axs[i].plot(ts_somatic[key], losses_somatic[key], 
                    label=["somatic"])
        except:
            continue
        axs[i].set_ylim((0.02,0.04))
        axs[i].set_title(r"$N_z$={}, rate={}Hz".format(key[2], key[1]*1000))
        i+=1

    for j in range(0,int(np.sqrt(M))):
        for k in range(0,int(np.sqrt(M))):
            i = j * int(np.sqrt(M)) + k
            if k == 0:
                axs[i].set_ylabel("decoder loss")
            else: 
                axs[i].set_yticklabels([])
            if j == int(np.sqrt(M))-1:
                axs[i].set_xlabel("t")
            else:
                axs[i].set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig("../../plots/scancompare_loss_poprate.svg")

    # compare endpoints of learning
    fig, axs = plt.subplots(figsize=(0.6*4,0.6*3))
    loss_d = []
    loss_o = []
    ms = []
    for key in sorted(list(losses_dendritic.keys())):
        ms.append(key[2])
        loss_o.append(np.min(losses_dendritic[key]))
        loss_d.append(np.min(losses_somatic[key]))

    plt.plot(ms, loss_d, marker='o', markersize=3, color='orange', label="SB")
    plt.plot(ms, loss_o, marker='o', markersize=3, color='darkolivegreen', label="DB")
    plt.xlabel("# z-neurons")
    plt.ylabel("decoder loss")
    #plt.xscale("log")
    plt.legend()


    poprate = [k for k in losses_dendritic.keys()][0][0]
    print(poprate)
    def m2rho(m):
        return poprate / m * 1000

    def rho2m(rho):
        return poprate / rho / 1000

    secax = axs.secondary_xaxis('top', functions=(m2rho, rho2m))
    secax.set_xlabel('single neuron rate [Hz]')

    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis in [axs.xaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    axs.set_xticks([50, 100, 200])
    secax.set_xticks([20,10,5])

    plt.tight_layout()
    plt.savefig("../../plots/scancompare_loss_poprate_compact.svg")

    # compare weights
    fig, axs = plt.subplots(N * M, 2, figsize=(2*3,M*N*3))
    axs = np.reshape(axs,-1)

    i = 0
    for key in sorted(list(losses_dendritic.keys())):
        def sbplt(fields, k, name):
            half = fields.shape[0]//2
            fields = fields[:half,:] - fields[half:,:]
            imsize = int(np.sqrt(fields.shape[0]))
            fields = np.transpose(fields)
            img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3)

            im = axs[k].imshow(img, cmap='Greys')
            colorbar(im)
            axs[k].axis('off')
            axs[k].set_title(name)
            fig.add_subplot(axs[k])

        
        sbplt(weights_dendritic[key], 2*i, r"$N_z$={}, dendritic".format(key[2]))
        sbplt(weights_somatic[key], 2*i+1, r"$N_z$={}, somatic".format(key[2]))
        i += 1

    plt.tight_layout()
    plt.savefig("../../plots/scancompare_weights.svg", dpi=500)


main()


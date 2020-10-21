import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil
sys.path.insert(0, '../src/')
from plot_utils import *
from mpl_toolkits.mplot3d import Axes3D

def main():
    ARGS = sys.argv
    if (len(ARGS) <= 2):
        print("Please provide two folders as argument (1. dendritic logs, 2. somatic logs).")
        return
    
    N = 4  # rates, vars
    M = 18 # dt
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
    snaps_dendritic = {}
    snaps_somatic = {}
    for folder in folders_dendritic:
        dic = h5py.File(folder + file, 'r')
        temp = dic[u'temp']
        meta = temp.attrs
        
        key = (meta["fixedFinalSigmaValue"],meta["dt"])
        losses_dendritic[key] = temp["test_decoder_loss"]

        snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]
        snaps_dendritic[key] = (snapshots[-1], meta)
        weights_dendritic[key] = snapshots[-1].get('xz_weights')[()]

        ts_dendritic[key] = temp["t"]

    for folder in folders_somatic:
        dic = h5py.File(folder + file, 'r')
        temp = dic[u'temp']
        meta = temp.attrs
        
        key = (meta["fixedFinalSigmaValue"],meta["dt"])
        losses_somatic[key] = temp["test_decoder_loss"]

        snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]
        snaps_somatic[key] = (snapshots[-1], meta)
        weights_somatic[key] = snapshots[-1].get('xz_weights')[()]

        ts_somatic[key] = temp["t"]
        
    # all learningcurves combined
    fig, axs = plt.subplots(N, M, figsize=(0.5*M*3.8,0.5*N*3))
    axs = np.reshape(axs,-1)
    i = 0
    for key in sorted(list(losses_dendritic.keys())):
        print(key)
        try:
            axs[i].plot(ts_dendritic[key]*key[1], losses_dendritic[key], 
                    label=["dendritic"])
        except:
            continue
        try:
            axs[i].plot(ts_somatic[key]*key[1], losses_somatic[key], 
                    label=["somatic"])
        except:
            continue
        axs[i].set_ylim((0.02,0.07))
        axs[i].set_title(r"$\Delta t$={:.2f}ms $\Delta u$={:.2f}".format(key[1],key[0]**2))
        i+=1

    for k in range(0,M):
        for j in range(0,N):
            i = j * M + k
            if k == 0:
                axs[i].set_ylabel("decoder loss")
            else: 
                axs[i].set_yticklabels([])
            if j == N-1:
                axs[i].set_xlabel("t")
            else:
                axs[i].set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig("../../plots/scancompare_timstep_loss_poprate_2d.svg")
    plt.close()

    # compare endpoints of learning 3d
    loss_d = np.zeros((N,M))
    loss_o = np.zeros((N,M))
    dts = np.sort(np.unique([key[1] for key in list(losses_dendritic.keys())]))
    vs = np.sort(np.unique([key[0] for key in list(losses_dendritic.keys())]))
    
    for j in range(0,N):
        for k in range(0,M):
            key = (vs[j],dts[k])
            loss_o[j,k] = np.mean(losses_dendritic[key][-10:-1])
            loss_d[j,k] = np.mean(losses_somatic[key][-10:-1])

    """fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.log(dts), vs)
    
    # Plot the surface.
    ax.plot_surface(X, Y, loss_d, linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, loss_o, linewidth=0, antialiased=False)
    plt.xlabel(r"$\Delta t$ [ms]")
    plt.ylabel(r"$\Delta u$")
    plt.tight_layout()
    plt.show()"""

    # compare endpoints of learning
    fig, axs = plt.subplots(N,1,figsize=(0.6*4,0.6*3*N))

    for j in range(0,N):
        loss_d_1d = loss_d[j,:]
        loss_o_1d = loss_o[j,:]
        ms = dts

        axs[j].plot(ms, loss_d_1d, marker='o', markersize=3, color='orange', label="SB")
        axs[j].plot(ms, loss_o_1d, marker='o', markersize=3, color='darkolivegreen', label="DB")
        axs[j].set_title(r"$\Delta u={:.2f}$".format(vs[j]**2))
        axs[j].set_xlabel(r"$\Delta t$ [ms]")
        axs[j].set_ylabel("decoder loss")
        axs[j].set_xscale("log")
        axs[j].set_xticks([0.1,1,10])
        axs[j].legend()

    plt.tight_layout()
    plt.savefig("../../plots/scancompare_timstep_loss_poprate_compact_2d.svg")


    # spikes
    fig, axs = plt.subplots(M,2,figsize=(4,1.5*M))

    j = 0
    for k in range(0,M):
        key = (vs[j],dts[k])
        snapshot, meta = snaps_somatic[key]

        dt = meta["dt"]
        interval = meta["snapshotLogInterval"]
        pres_len = int(meta["presentationLength"] / meta["snapshotLogInterval"])
        tmin = 10 * pres_len
        tmax = 100 * pres_len

        spikes = snapshot.get('z_spikes')[()]
        data = []
        for t in range(spikes.shape[0]):
            sub = []
            for n in range(spikes.shape[1]):
                if spikes[t,n]>=tmin*dt and spikes[t,n]<tmax*dt:
                    sub.append((spikes[t,n]-tmin*dt)*interval/1000) 
            data.append(sub)
        axs[k,0].eventplot(data, color='cornflowerblue')
        axs[k,0].set_xlabel(r'$t$ $[s]$')
        axs[k,0].set_ylabel('# neuron')
        axs[k,0].set_title(r'$\Delta t={} ms$'.format(dt))


        snapshot, meta = snaps_dendritic[key]

        spikes = snapshot.get('z_spikes')[()]
        data = []
        for t in range(spikes.shape[0]):
            sub = []
            for n in range(spikes.shape[1]):
                if spikes[t,n]>=tmin*dt and spikes[t,n]<tmax*dt:
                    sub.append((spikes[t,n]-tmin*dt)*interval/1000) 
            data.append(sub)
        axs[k,1].eventplot(data, color='cornflowerblue')
        axs[k,1].set_xlabel(r'$t$ $[s]$')
        axs[k,1].set_ylabel('# neuron')
        axs[k,1].set_title(r'$\Delta t={} ms$'.format(dt))

    plt.tight_layout()
    plt.savefig("../../plots/scancompare_timstep_loss_poprate_spikes_2d.svg")
    

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

        
        sbplt(weights_dendritic[key], 2*i, "dt={}, dendritic".format(key))
        sbplt(weights_somatic[key], 2*i+1, "dt={}, somatic".format(key))
        i += 1

    plt.savefig("../../plots/scancompare_timstep_weights_2d.svg", dpi=500)


main()


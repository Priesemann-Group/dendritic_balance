
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
    if (len(ARGS) <= 4):
        print("Please provide 4 folders as argument (1. dendritic instant inh., 2. dendritic regular, 3. somatic instant inh, 4. somatic regular).")
        return

    plot_folder = str("/".join(ARGS[1].split("/")[0:-1])) + "/"
    filename1 = ARGS[1] + "/log.h5"
    filename2 = ARGS[2] + "/log.h5"
    filename3 = ARGS[3] + "/log.h5"
    filename4 = ARGS[4] + "/log.h5"
    f = str(ARGS[1].split("/")[-1])

    #filename2 = ARGS[2] + "/log.h5"

    dic_db_inst = h5py.File(filename1, 'r')
    dic_db_reg = h5py.File(filename2, 'r')
    dic_sb_inst = h5py.File(filename3, 'r')
    dic_sb_reg = h5py.File(filename4, 'r')
    folder = "."

    plot_results(dic_db_inst, dic_db_reg, dic_sb_inst, dic_sb_reg, folder)

def plot_results(dic_db_inst, dic_db_reg, dic_sb_inst, dic_sb_reg, folder):
    
    temp = dic_db_inst[u'temp']
    
    whitening_matrix = dic_db_inst["whitening_matrix"].get("whitening_matrix")[()]
    print(whitening_matrix)

    scale = 1.
    N = 100
    db_factor = 30 * 1.378 # revert normalization

    dt = temp.attrs.get("dt")
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / (temp.attrs.get("snapshotLogInterval")))
    
    tmin =  100*pres_len
    tmax = tmin + 400*pres_len # in steps
    
    fig, axs = plt.subplots(2, 5, figsize=(scale*13.3,scale*2.8))
    
    
    #signal
    snapshots = [dic_db_inst[k] for k in sorted(dic_db_inst.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    
    x = snapshot.get('x_outputs')[()]
    x = unwhiten(x, whitening_matrix)

    pcm = axs[0,0].imshow(x[:,tmin:tmax] * db_factor, aspect='auto', extent=[0,(tmax-tmin)*interval*dt/1000,25,0])
    axs[0,0].set_ylabel('# input')
    
    fig.colorbar(pcm, ax=[axs[1,0]], shrink=1.0, label="dB SP", location='bottom')
    
    ## DB instant
    snapshots = [dic_db_inst[k] for k in sorted(dic_db_inst.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    
    #spikes
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin*interval*dt and spikes[t,n]*interval<tmax*interval*dt:
                sub.append((spikes[t,n]*interval-tmin*dt*interval)/1000) 
        data.append(sub)

    axs[1,1].eventplot(data, color='cornflowerblue')
    axs[1,1].set_ylabel('# neuron')
    axs[1,1].set_xlabel(r'$t$ $[s]$')
    axs[1,1].set_xlim(0, (tmax-tmin)*dt*interval/1000 )
    
    #reconstruction
    x_hat = snapshot.get('reconstructions')[()]
    x_hat = unwhiten(x_hat * db_factor, whitening_matrix)
    
    for i in range(x_hat.shape[0]):
        x_hat[i,:] = np.convolve(x_hat[i,:], np.ones(N)/N, mode='same')

    axs[0,1].imshow(x_hat[:,tmin:tmax], aspect='auto', extent=[0,(tmax-tmin)*interval*dt/1000,25,0])
    axs[0,1].set_ylabel('# input')
    
    ## DB reg
    snapshots = [dic_db_reg[k] for k in sorted(dic_db_reg.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    
    #spikes
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin*interval*dt and spikes[t,n]*interval<tmax*interval*dt:
                sub.append((spikes[t,n]*interval-tmin*dt*interval)/1000) 
        data.append(sub)

    axs[1,2].eventplot(data, color='cornflowerblue')
    axs[1,2].set_ylabel('# neuron')
    axs[1,2].set_xlabel(r'$t$ $[s]$')
    axs[1,2].set_xlim(0, (tmax-tmin)*dt*interval /1000)
    
    #reconstruction
    x_hat = snapshot.get('reconstructions')[()]
    x_hat = unwhiten(x_hat * db_factor, whitening_matrix)
    
    for i in range(x_hat.shape[0]):
        x_hat[i,:] = np.convolve(x_hat[i,:], np.ones(N)/N, mode='same')

    axs[0,2].imshow(x_hat[:,tmin:tmax], aspect='auto', extent=[0,(tmax-tmin)*interval*dt/1000,25,0])
                  #vmin=np.min(x), vmax=np.max(x))
    axs[0,2].set_ylabel('# input')
    
    ## SB instant
    snapshots = [dic_sb_inst[k] for k in sorted(dic_sb_inst.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    
    #spikes
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin*interval*dt and spikes[t,n]*interval<tmax*interval*dt:
                sub.append((spikes[t,n]*interval-tmin*dt*interval)/1000) 
        data.append(sub)

    axs[1,3].eventplot(data, color='cornflowerblue')
    axs[1,3].set_ylabel('# neuron')
    axs[1,3].set_xlim(0, (tmax-tmin)*dt*interval/1000 )
    axs[1,3].set_xlabel(r'$t$ $[s]$')
    
    #reconstruction
    x_hat = snapshot.get('reconstructions')[()]
    x_hat = unwhiten(x_hat * db_factor, whitening_matrix)
    
    for i in range(x_hat.shape[0]):
        x_hat[i,:] = np.convolve(x_hat[i,:], np.ones(N)/N, mode='same')

    axs[0,3].imshow(x_hat[:,tmin:tmax], aspect='auto', extent=[0,(tmax-tmin)*interval*dt/1000,25,0]) 
                  #vmin=np.min(x), vmax=np.max(x))
    axs[0,3].set_ylabel('# input')
    
    ## SB reg
    snapshots = [dic_sb_reg[k] for k in sorted(dic_sb_reg.keys(), key=natural_keys) if 'snapshot' in k]
    snapshot = snapshots[-1]
    
    #spikes
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]*interval>=tmin*interval*dt and spikes[t,n]*interval<tmax*interval*dt:
                sub.append((spikes[t,n]*interval-tmin*dt*interval)/1000) 
        data.append(sub)

    axs[1,4].eventplot(data, color='cornflowerblue')
    axs[1,4].set_ylabel('# neuron')
    axs[1,4].set_xlim(0, (tmax-tmin)*dt*interval/1000 )
    axs[1,4].set_xlabel(r'$t$ $[s]$')
    
    #reconstruction
    x_hat = snapshot.get('reconstructions')[()]
    x_hat = unwhiten(x_hat * db_factor, whitening_matrix)
    
    for i in range(x_hat.shape[0]):
        x_hat[i,:] = np.convolve(x_hat[i,:], np.ones(N)/N, mode='same')

    axs[0,4].imshow(x_hat[:,tmin:tmax], aspect='auto', extent=[0,(tmax-tmin)*interval*dt/1000,25,0], 
                  vmin=np.min(x_hat), vmax=np.max(x_hat))
    axs[0,4].set_ylabel('# input')
    
    plt.tight_layout()
    plt.savefig(folder + "/speech.svg")
    
def unwhiten(data, W):
    l = data.shape[0]
    data = data[:l//2,:] - data[l//2:,:]
    W_inv = np.linalg.inv(W)
    data = np.dot(W_inv, data)
    data -= np.min(data)
    return data

main()

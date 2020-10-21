
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
    temp = dic[u'temp']
    temp2 = dic2[u'temp']

    n_imgs = 5**2
    img_min = 0

    dt = temp.attrs.get("dt") 
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / temp.attrs.get("snapshotLogInterval"))


    # create figure
    scale = 0.8
    fig = plt.figure(figsize=(12*scale, 5*scale))
    plt.axis('off')
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 4), (1, 0), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 4), (1, 1), colspan=1, rowspan=1)
    ax5 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=1)
    ax6 = plt.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1, sharex=ax5)

    times = dt * temp.get('t') / 1000

    # decoder loss
    loss = temp.get('test_decoder_loss')
    loss2 = temp2.get('test_decoder_loss')
    ax1.plot(times[1:-1], loss2[1:-1], color='orange')
    ax1.plot(times[1:-1], loss[1:-1], color='darkolivegreen')
    ax1.legend(['SB', 'DB'])
    ax1.set_xlabel(r'$t$ $[s]$')
    ax1.set_ylabel('decoder loss')
    ax1.ticklabel_format(axis='x',style='sci', scilimits=(0,1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #xmin, xmax = ax1.get_xlim()
    #ax1.set_xticks(np.round(np.linspace(xmin, xmax, 3), 2))

    # receptive fields
    fields = snapshot.get('xz_weights')[()]
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3)

    im = ax2.imshow(img, cmap='Greys')
    cbar = colorbar(im)
    axc = cbar.ax
    #axc.text(7,0.3,r'$F$',rotation=0)
    ax2.axis('off')
    fig.add_subplot(ax2)

    # originals
    recs = snapshot.get('x_outputs')[()].T
    tmin = img_min*pres_len + 20
    tmax = min(tmin+n_imgs*pres_len,recs.shape[0])
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=6)

    im = ax3.imshow(img, cmap='Greys', vmin=0, vmax=1)
    cbar = colorbar(im)
    axc = cbar.ax
    #axc.text(8.5,0.46,r'$\mathbf{x}$',rotation=0)
    ax3.axis('off')
    fig.add_subplot(ax3)

    # reconstruction
    recs = snapshot.get('reconstructions')[()].T
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=3)

    im = ax4.imshow(img, cmap='Greys')
    cbar = colorbar(im)
    axc = cbar.ax
    #axc.text(9.6,0.45,r'$\mathbf{\hat x}$',rotation=0)
    ax4.axis('off')
    fig.add_subplot(ax4)

    # reconstruction over time
    recs = snapshot.get('reconstructions')[()].T
    xs = snapshot.get('x_outputs')[()].T

    tmax = min(tmin+15*pres_len,recs.shape[0])
    var = np.mean(np.power(recs[tmin:tmax,:] - xs[tmin:tmax,:], 2), axis=0)
    inds = np.argsort(var)

    input_ind = 197
    ax5.plot(dt * interval * np.arange(tmax-tmin) / 1000, recs[tmin:tmax,input_ind], color='cornflowerblue', label=r"$\hat x$")
    ax5.plot(dt * interval * np.arange(tmax-tmin) / 1000, xs[tmin:tmax,input_ind], color='lightcoral', label=r"$x$")
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(reversed(handles), reversed(labels), loc='upper right')
    ax5.set_ylabel('pixel value')
    ax5.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    fig.add_subplot(ax5)

    # spikes
    spikes = snapshot.get('z_spikes')[()]
    data = []
    for t in range(spikes.shape[0]):
        sub = []
        for n in range(spikes.shape[1]):
            if spikes[t,n]>=tmin*dt and spikes[t,n]<tmax*dt:
                sub.append((spikes[t,n]-tmin*dt)*interval/1000) 
        data.append(sub)

    ax6.eventplot(data, color='cornflowerblue')
    ax6.set_xlabel(r'$t$ $[s]$')
    ax6.set_ylabel('# neuron')
    fig.add_subplot(ax6)


    plt.tight_layout()
    # resize images
    for ax in [ax2,ax4]:
        bbox = ax.get_position()
        p = bbox.get_points()
        p[0,1] -= bbox.bounds[3] * 0.1
        p[0,0] -= bbox.bounds[2] * 0.1
        p[:,0] -= bbox.bounds[2] * 0.1
        bbox.set_points(p)
        ax.set_position(bbox)
    for ax in [ax3]:
        bbox = ax.get_position()
        p = bbox.get_points()
        p[0,1] -= bbox.bounds[3] * 0.1
        p[0,0] -= bbox.bounds[2] * 0.1
        p[:,0] -= bbox.bounds[2] * 0.05
        bbox.set_points(p)
        ax.set_position(bbox)

    # remove space between ax5 & 6
    bbox = ax6.get_position()
    p = bbox.get_points()
    p[1,1] += bbox.bounds[3] * 0.37
    bbox.set_points(p)
    ax6.set_position(bbox)
    bbox = ax5.get_position()
    p = bbox.get_points()
    p[0,1] -= bbox.bounds[3] * 0.23
    bbox.set_points(p)
    ax5.set_position(bbox)
    plt.savefig(folder + "fig1.svg")
    plt.savefig("../../plots/mnist.svg")


main()

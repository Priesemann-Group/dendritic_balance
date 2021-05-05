
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
    snapshot_start = snapshots[0]
    snapshot_middle = snapshots[1]
    snapshot_end = snapshots[2]
    temp = dic[u'temp']
    temp2 = dic2[u'temp']

    n_imgs = 4**2
    img_min = 0

    dt = temp.attrs.get("dt") 
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / temp.attrs.get("snapshotLogInterval"))


    # create figure
    scale = 0.8
    fig = plt.figure(figsize=(12*scale, 11*scale))
    plt.axis('off')
    
    ax1  = plt.subplot2grid((5, 6), (0, 0), colspan=6, rowspan=1)
    
    ax21 = plt.subplot2grid((5, 6), (1, 0), colspan=1, rowspan=1)
    ax22 = plt.subplot2grid((5, 6), (1, 1), colspan=1, rowspan=1)
    ax23 = plt.subplot2grid((5, 6), (1, 2), colspan=1, rowspan=1)
    ax24 = plt.subplot2grid((5, 6), (1, 3), colspan=1, rowspan=1)
    ax25 = plt.subplot2grid((5, 6), (1, 4), colspan=1, rowspan=1)
    ax26 = plt.subplot2grid((5, 6), (1, 5), colspan=1, rowspan=1)
    
    ax31 = plt.subplot2grid((5, 6), (2, 0), colspan=1, rowspan=1)
    ax32 = plt.subplot2grid((5, 6), (2, 1), colspan=1, rowspan=1)
    ax33 = plt.subplot2grid((5, 6), (2, 2), colspan=1, rowspan=1)
    ax34 = plt.subplot2grid((5, 6), (2, 3), colspan=1, rowspan=1)
    ax35 = plt.subplot2grid((5, 6), (2, 4), colspan=1, rowspan=1)
    ax36 = plt.subplot2grid((5, 6), (2, 5), colspan=1, rowspan=1)
    
    ax41 = plt.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=1)
    ax42 = plt.subplot2grid((5, 6), (3, 2), colspan=2, rowspan=1)
    ax43 = plt.subplot2grid((5, 6), (3, 4), colspan=2, rowspan=1)
    
    ax51 = plt.subplot2grid((5, 6), (4, 0), colspan=2, rowspan=1, sharex=ax41)
    ax52 = plt.subplot2grid((5, 6), (4, 2), colspan=2, rowspan=1, sharex=ax42)
    ax53 = plt.subplot2grid((5, 6), (4, 4), colspan=2, rowspan=1, sharex=ax43)

    # decoder loss
    t_inhibitory_learning = snapshot_start.get('t')[()] * dt / 1000
    all_times = temp.get('t') * dt / 1000
    ind_t = np.where(all_times >= t_inhibitory_learning)[0][0]
    times = all_times - t_inhibitory_learning
    
    loss = temp.get('test_decoder_loss')
    loss2 = temp2.get('test_decoder_loss')
    
    from scipy.interpolate import make_interp_spline, BSpline
    times_long = np.linspace(times[ind_t], times[-1], 300)
    spl = make_interp_spline(times[ind_t:-1], loss[ind_t:-1], k=2)  # type: BSpline
    loss_smooth = spl(times_long)
    spl = make_interp_spline(times[ind_t:-1], loss2[ind_t:-1], k=2)  # type: BSpline
    loss2_smooth = spl(times_long)
    
    ax1.plot(times_long, loss2_smooth, color='orange')
    ax1.plot(times_long, loss_smooth, color='darkolivegreen')
    ax1.legend(['SB', 'DB'])
    ax1.set_xlabel(r'$t$ $[s]$')
    ax1.set_ylabel('decoder loss')
    ax1.ticklabel_format(axis='x',style='sci', scilimits=(0,1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.set_xticks([0.0,5*10**3,10**4])
    #ax1.set_xticklabels([r"$0$",r"$5\times 10^3$",r"$10^4$"])




    # receptive fields
    field_min = -0.1
    field_max = 1.0
    
    fields = snapshot_start.get('xz_weights')[()]
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax21.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax21.axis('off')
    ax21.set_title(r'$F$') #, y=-0.3)
    fig.add_subplot(ax21)
    #fig.colorbar(im, ticklocation='left')
    
    fields = snapshot_middle.get('xz_weights')[()]
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax23.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax23.axis('off')
    ax23.set_title(r'$F$') 
    fig.add_subplot(ax23)
    
    fields = snapshot_end.get('xz_weights')[()]
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax25.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax25.axis('off')
    ax25.set_title(r'$F$') 
    fig.add_subplot(ax25)
    
    # decoder weights
    fields = snapshot_start.get('decoder_weights')[()].T
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax22.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax22.axis('off')
    ax22.set_title(r'$D$') 
    fig.add_subplot(ax22)
    
    fields = snapshot_middle.get('decoder_weights')[()].T
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax24.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax24.axis('off')
    ax24.set_title(r'$D$') 
    fig.add_subplot(ax24)
    
    fields = snapshot_end.get('decoder_weights')[()].T
    fields = np.swapaxes(fields,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = ax26.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    ax26.axis('off')
    ax26.set_title(r'$D$') 
    fig.add_subplot(ax26)
    



    # originals
    recs = snapshot_start.get('x_outputs')[()].T
    tmin = img_min*pres_len + 20
    tmax = min(tmin+n_imgs*pres_len,recs.shape[0])
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=6)
    
    im = ax31.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax31.axis('off')
    ax31.set_title(r'$x$') 
    fig.add_subplot(ax31)
    #fig.colorbar(im, ticklocation='left', ticks=[0, 1])
    
    im = ax33.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax33.axis('off')
    ax33.set_title(r'$x$') 
    fig.add_subplot(ax33)
    
    im = ax35.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax35.axis('off')
    ax35.set_title(r'$x$') 
    fig.add_subplot(ax35)
    

    # reconstruction
    recs = snapshot_start.get('reconstructions')[()].T
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=3)
    im = ax32.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax32.axis('off')
    ax32.set_title(r'$\hat x$') 
    fig.add_subplot(ax32)
    
    recs = snapshot_middle.get('reconstructions')[()].T
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=3)
    im = ax34.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax34.axis('off')
    ax34.set_title(r'$\hat x$') 
    fig.add_subplot(ax34)
    
    recs = snapshot_end.get('reconstructions')[()].T
    recs = np.reshape(recs[tmin:tmax,:], (n_imgs,pres_len,recs.shape[1]))
    recs = np.mean(recs[:,int(pres_len*0.3):,:], axis=1)
    imsize = int(np.sqrt(recs.shape[1]))
    img = gallery(np.reshape(recs, (-1, imsize, imsize)), spacing=3)
    im = ax36.imshow(img, cmap='Greys', vmin=0, vmax=1)
    ax36.axis('off')
    ax36.set_title(r'$\hat x$') 
    fig.add_subplot(ax36)

    # reconstruction over time
    input_ind = 197
    recs = snapshot_start.get('reconstructions')[()].T
    tmax = min(tmin+15*pres_len,recs.shape[0])
    
    recs = snapshot_start.get('reconstructions')[()].T
    xs = snapshot_start.get('x_outputs')[()].T
    ax41.plot(dt * interval * np.arange(tmax-tmin) / 1000, recs[tmin:tmax,input_ind], 
             color='cornflowerblue', label=r"$\hat x$")
    ax41.plot(dt * interval * np.arange(tmax-tmin) / 1000, xs[tmin:tmax,input_ind], 
             color='lightcoral', label=r"$x$")
    handles, labels = ax41.get_legend_handles_labels()
    ax41.set_ylabel('pixel value')
    ax41.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    for side in ax41.spines:
        if side != "left":
            ax41.spines[side].set_visible(False)
    fig.add_subplot(ax41)
    
    recs = snapshot_middle.get('reconstructions')[()].T
    xs = snapshot_middle.get('x_outputs')[()].T
    ax42.plot(dt * interval * np.arange(tmax-tmin) / 1000, recs[tmin:tmax,input_ind], 
             color='cornflowerblue', label=r"$\hat x$")
    ax42.plot(dt * interval * np.arange(tmax-tmin) / 1000, xs[tmin:tmax,input_ind], 
             color='lightcoral', label=r"$x$")
    handles, labels = ax42.get_legend_handles_labels()
    ax42.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    for side in ax42.spines:
        if side != "left":
            ax42.spines[side].set_visible(False)
    fig.add_subplot(ax42)
    
    recs = snapshot_end.get('reconstructions')[()].T
    xs = snapshot_end.get('x_outputs')[()].T
    ax43.plot(dt * interval * np.arange(tmax-tmin) / 1000, recs[tmin:tmax,input_ind], 
             color='cornflowerblue', label=r"$\hat x$")
    ax43.plot(dt * interval * np.arange(tmax-tmin) / 1000, xs[tmin:tmax,input_ind], 
             color='lightcoral', label=r"$x$")
    handles, labels = ax43.get_legend_handles_labels()
    ax43.legend(reversed(handles), reversed(labels), loc='upper right')
    ax43.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    for side in ax43.spines:
        if side != "left":
            ax43.spines[side].set_visible(False)
    fig.add_subplot(ax43)

    # spikes
    def spikes_to_data(spikes):
        data = []
        for t in range(spikes.shape[0]):
            sub = []
            for n in range(spikes.shape[1]):
                if spikes[t,n]>=tmin*dt and spikes[t,n]<tmax*dt:
                    sub.append((spikes[t,n]-tmin*dt)*interval/1000) 
            data.append(sub)
        return data
        
    spikes = snapshot_start.get('z_spikes')[()]
    data = spikes_to_data(spikes)
    ax51.eventplot(data, color='cornflowerblue')
    ax51.plot([0.0,0.2],[0.0,0.0],color="black")
    ax51.annotate("200 ms", (0, 0))
    ax51.set_ylabel('# neuron')
    ax51.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax51.spines["bottom"].set_visible(False)
    ax51.spines["right"].set_visible(False)
    ax51.spines["top"].set_visible(False)
    fig.add_subplot(ax51)
    
    spikes = snapshot_middle.get('z_spikes')[()]
    data = spikes_to_data(spikes)
    ax52.eventplot(data, color='cornflowerblue')
    ax52.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax52.spines["bottom"].set_visible(False)
    ax52.spines["right"].set_visible(False)
    ax52.spines["top"].set_visible(False)
    fig.add_subplot(ax52)
    
    spikes = snapshot_end.get('z_spikes')[()]
    data = spikes_to_data(spikes)
    ax53.eventplot(data, color='cornflowerblue')
    ax53.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax53.spines["bottom"].set_visible(False)
    ax53.spines["right"].set_visible(False)
    ax53.spines["top"].set_visible(False)
    fig.add_subplot(ax53)

    plt.tight_layout()
    """# resize images
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
    ax5.set_position(bbox)"""
    
    plt.savefig("mnist.svg")


main()

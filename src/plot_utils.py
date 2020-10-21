
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

""" Creates colorbar on axis"""
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

""" Can be used to sort by human number order"""
def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

""" Arrange images in a grid with spacing """
def gallery(array, ncols=-1, spacing=2):
    if ncols == -1:
        ncols = np.math.ceil(np.sqrt(array.shape[0]))
    nrows = np.math.ceil(array.shape[0]/float(ncols))
    cell_w = array.shape[2]
    cell_h = array.shape[1]
    result = np.ones(((cell_h+spacing)*nrows + spacing, (cell_w+spacing)*ncols + spacing), dtype=array.dtype) * np.min(array)
    s = spacing
    for i in range(0, nrows):
        for j in range(0, ncols):
            if i*ncols+j < array.shape[0]:
                result[i*(cell_h+s)+s:(i+1)*(cell_h+s), j*(cell_w+s)+s:(j+1)*(cell_w+s)] = array[i*ncols+j]
    return result

from typing import Optional, Any, Tuple, Dict, List, Union
import math
import numpy as np
import matplotlib.colors as col
from matplotlib.colors import Normalize, LogNorm

from . import attenuate_power

# default settings
errorbar_kw_default = {'capsize':3, 'ls':''}
fill_between_kw_default = {'lw':0.0}

# This is from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
cb_magic_args = {'fraction':0.046, 'pad':0.04}
'''
You can correct for the case where image is too wide using this trick: im_ratio = data.shape[0]/data.shape[1] plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04) where data is your image

The shrink keyword argument, which defaults to 1.0, may also be useful for further fine tuned adjustments. I found that shrink=0.9 helped get it just right when I had two square subplots side by side
'''


gray = '#808080'
red = '#ff0000'
blue = '#0000ff'
green = '#00ff00'
anglecmap = col.LinearSegmentedColormap.from_list('anglecmap', [green, blue, gray, red, green], N=256, gamma=1)
anglecmap_r = col.LinearSegmentedColormap.from_list('anglecmap_r', [green, blue, gray, red, green][::-1], N=256, gamma=1)
anglecmap_shifted = col.LinearSegmentedColormap.from_list('anglecmap_shifted', [gray, blue, green, red, gray], N=256, gamma=1)
anglecmap_shifted_r = col.LinearSegmentedColormap.from_list('anglecmap_shifted_r', [gray, blue, green, red, gray][::-1], N=256, gamma=1)

# rivulet colors
color_w = '#3d5da9'
color_w_rgb = (61, 93, 169)
cmap_w_dict = {'red':   [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[0]/255, color_w_rgb[0]/255]],
               'green': [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[1]/255, color_w_rgb[1]/255]],
               'blue':  [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[2]/255, color_w_rgb[2]/255]]}
cmap_w = col.LinearSegmentedColormap('white_to_w', segmentdata=cmap_w_dict, N=256)


color_z = '#ff1a1a'
color_z_rgb = (255, 26, 26)
cmap_z_dict = {'red':   [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[0]/255, color_z_rgb[0]/255]],
               'green': [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[1]/255, color_z_rgb[1]/255]],
               'blue':  [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[2]/255, color_z_rgb[2]/255]]}
cmap_z = col.LinearSegmentedColormap('white_to_z', segmentdata=cmap_z_dict, N=256)

color_q = '#9c1ab2' # (dark version : #9c1ab2 | light version : #c320df

# condition colors
color_smallQ = '#008C00'
color_bigQ = '#C29A49'

cmap_zonly = 'PuOr_r'
cmap_wonly = 'viridis_r'

def force_aspect_ratio(ax, aspect=1):
    # old version, for images
    # im = ax.get_images()
    # extent =  im[0].get_extent()
    extent = [*ax.get_xlim(), *ax.get_ylim()]
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

    # # IF THIS DOES NOT WORK, TRY
    # ax.set_box_aspect(1.)

def set_yaxis_rad(ax):
    ax.set_yticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi], minor=False)
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], minor=False)
    ax.set_yticks([-3*math.pi/4, -math.pi/4, 0, math.pi/4, 3*math.pi/4], minor=True)
    ax.set_ylim(-math.pi, math.pi)

def set_yaxis_log(ax, maximum_amplitude:float, range_db:Union[int, float], text:bool=True,
                  step_minor=None):
    step_major = 40
    if range_db < 200:
        step_major = 20
    if range_db < 100:
        step_major = 20
    if range_db < 60:
        step_major = 10
    if range_db < 30:
        step_major = 5
    if step_minor is None:
        step_minor = 20
        if range_db < 200:
            step_minor = 10
        if range_db < 100:
            step_minor = 5
        if range_db < 60:
            step_minor = 2
        if range_db < 30:
            step_minor = 1
    # it seems unreasonable to have range_db > 100 or < 10
    att_db_major = np.arange(0, range_db+1, step_major)
    att_db_minor = np.arange(0, range_db+1, step_minor)
    cbticks_major = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_major]
    cbticklabels = ['0 dB' if att_db == 0 else f'-{att_db} dB' for att_db in att_db_major]
    cbticks_minor = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_minor]

    ax.set_yticks(cbticks_major, minor=False)
    ax.set_yticklabels(cbticklabels if text else [], minor=False)
    ax.set_yticks(cbticks_minor, minor=True)
    ax.set_yticklabels([], minor=True)

def set_ticks_log_cb(cb, maximum_amplitude:float, range_db:Union[int, float], text:bool=True):
    # LEGACY DO NOT USE
    set_yaxis_log(cb.ax, maximum_amplitude, range_db, text)
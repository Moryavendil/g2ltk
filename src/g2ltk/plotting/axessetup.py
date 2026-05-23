import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple, Optional
import math

from g2ltk.peakfinder import step
from g2ltk.fourier import attenuate_power


# This is from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
cb_magic_args = {'fraction':0.046, 'pad':0.04}
'''
You can correct for the case where image is too wide using this trick: 
im_ratio = data.shape[0]/data.shape[1] plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04) where data is your image

The shrink keyword argument, which defaults to 1.0, may also be useful for further fine tuned adjustments. 
I found that shrink=0.9 helped get it just right when I had two square subplots side by side
'''
def correct_limits(arr: np.ndarray) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2

def correct_extent(arr_x: np.ndarray, arr_y: np.ndarray, origin='upper') -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    if origin == 'upper':
        return xlim[0], xlim[1], ylim[1], ylim[0]
    elif origin == 'lower':
        return xlim[0], xlim[1], ylim[0], ylim[1]

def set_yaxis_rad(ax: plt.Axes):
    ax.set_yticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi], minor=False)
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], minor=False)
    ax.set_yticks([-3*math.pi/4, -math.pi/4, 0, math.pi/4, 3*math.pi/4], minor=True)
    ax.set_ylim(-math.pi, math.pi)

def set_yaxis_log(ax: plt.Axes, maximum_amplitude:float, range_db:Union[int, float], text:bool=True,
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

def set_bottom_xlabels(axs: list[list[plt.Axes]], label: str):
    for i in range(len(axs[-1])):
        axs[-1][i].set_xlabel(label)
        axs[0][i].xaxis.set_ticks_position('bottom')
        axs[0][i].xaxis.set_label_position('bottom')

def set_top_xlabels(axs: list[list[plt.Axes]], label: str):
    for j in range(len(axs[0])):
        axs[0][j].set_xlabel(label)
        axs[0][j].xaxis.set_ticks_position('top')
        axs[0][j].xaxis.set_label_position('top')

def set_top_xlabels(axs: list[list[plt.Axes]], label: str):
    for j in range(len(axs[0])):
        axs[0][j].set_xlabel(label)
        axs[0][j].xaxis.set_ticks_position('top')
        axs[0][j].xaxis.set_label_position('top')


def set_left_ylabels(axs: list[list[plt.Axes]], label: str):
    for i in range(len(axs)):
        axs[i][0].set_ylabel(label)
        axs[i][0].yaxis.set_ticks_position('left')
        axs[i][0].yaxis.set_label_position('left')


def set_right_ylabels(axs: list[list[plt.Axes]], label: Optional[str] = None):
    for i in range(len(axs)):
        if label is not None: axs[i][-1].set_ylabel(label)
        axs[i][-1].yaxis.set_ticks_position('right')
        axs[i][-1].yaxis.set_label_position('right')

def set_ylabel_right(ax: plt.Axes, label: Optional[str] = None):
    if label is not None: ax.set_ylabel(label)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')


def grids(axs: list[list[plt.Axes]], which='major', axis='both', **kwargs):
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].grid(which=which, axis=axis, **kwargs)
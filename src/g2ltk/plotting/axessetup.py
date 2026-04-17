import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import math

from g2ltk.fourier import attenuate_power

# This is from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
cb_magic_args = {'fraction':0.046, 'pad':0.04}
'''
You can correct for the case where image is too wide using this trick: 
im_ratio = data.shape[0]/data.shape[1] plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04) where data is your image

The shrink keyword argument, which defaults to 1.0, may also be useful for further fine tuned adjustments. 
I found that shrink=0.9 helped get it just right when I had two square subplots side by side
'''

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

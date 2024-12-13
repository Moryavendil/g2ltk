# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from tools.utility import activate_saveplot
%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
from scipy.signal import hilbert

from tools import datareading, utility
utility.configure_mpl()


# <codecell>

# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

# Acquisition selection
acquisition = '1Hz_start'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        framenumbers = np.arange(1100, 1600)
    elif acquisition=='1Hz_strong':
        framenumbers = np.arange(1600-500, 1600)
    elif acquisition=='2400mHz_start_vrai':
        framenumbers = np.arange(4769-500, 4769)
    elif acquisition=='10Hz_decal':
        framenumbers = np.arange(1500, 1800)


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)
frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')


# <codecell>

um_per_px = None
if dataset=='Nalight_cleanplate_20240708':
    um_per_px = 5000 / 888
mm_per_px = um_per_px/1000 if um_per_px is not None else None

density = 1.72
refractive_index = 1.26
lambda_Na_void = 0.589 # in um
gamma = 14
bsur2 = 600/2 # in mm

lambd = lambda_Na_void/refractive_index # in um
h0 = lambd * np.pi / (2 * np.pi)/2 # h0 = lambd/4 in um

from matplotlib.colors import LinearSegmentedColormap
cmap_Na = LinearSegmentedColormap.from_list("cmap_Na", ['black', "xkcd:bright yellow"])

vmin_Na = np.percentile(frames, 1)
vmax_Na = np.percentile(frames, 99)


# <codecell>

n_frame_ref = 1345

x_probe = 100
probespan = 5

interesting_probes = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        if n_frame_ref == 1345:
            interesting_probes = [100, 605, 1000, 1400, 1800]
        elif n_frame_ref == 1367:
            interesting_probes = [175, 630, 1030, 1400, 1800]
    if acquisition=='1Hz_strong':
        if n_frame_ref == 1566:
            interesting_probes = [160, 430, 660, 850, 1060, 1260, 1440, 1620, 1820, 2000]
    if acquisition=='2400mHz_start_vrai':
        x_probe = 1024
    if acquisition=='10Hz_decal':
        probespan = 5
        interesting_probes = [480, 1270, 1950]


# <codecell>

i_frame_ref = n_frame_ref - framenumbers[0]

DISP_OFFSET = 0
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        DISP_OFFSET = 820-533

z = np.arange(height)
frame_ref = frames[i_frame_ref]#[::-1, :][DISP_OFFSET:]


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(frame_ref, origin='lower', aspect='auto')
# for x_probe_interest in interesting_probes:
#     ax.axvline(x_probe_interest, color='k', ls='--', alpha=0.3)
#     ax.axvspan(x_probe_interest - probespan, x_probe_interest + probespan, color='k', alpha=.01)
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')


# <codecell>

ymax, ymin = None, None

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        ymax, ymin = 820, 820-533
    if acquisition=='10Hz_decal':
        if n_frame_ref == 1673:
            ymax, ymin = 540, None

probecolors = plt.cm.rainbow(np.linspace(0, 1, len(interesting_probes)))[::-1]

img = frame_ref[ymin:ymax,:]

i_probe = 0
x_probe = interesting_probes[i_probe]


# <codecell>

utility.activate_saveplot()
fgsize = utility.figsize(30, 60, unit='mm')
plt.figure(figsize=fgsize)
ax = plt.gca()
# ax = axes[0,0]
ax.imshow(img[::-1, :].T, extent=[-mm_per_px/2, (img.shape[0]+1/2)*mm_per_px, (img.shape[1]+1/2)*mm_per_px, -mm_per_px/2], origin='upper', aspect='equal',
          cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na)
# for x_probe_interest in interesting_probes:
#     ax.axvline(x_probe_interest, color='k', ls='--', alpha=0.3)
#     ax.axvspan(x_probe_interest - probespan, x_probe_interest + probespan, color='k', alpha=.01)

ax.set_ylim(10, 3)
ax.set_ylabel(r'$x$ [mm]')
ax.set_xticks(np.arange(4))
ax.set_xlim(3, 0)
ax.set_xlabel(r'$z$ [mm]')
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

utility.tighten_graph()
# utility.save_graphe('imgillustr')


# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'figure.titlesize' : 12,
                     'axes.labelsize': 12,'axes.titlesize': 12,
                     'legend.fontsize': 12})

from tools import datareading, rivuletfinding, datasaving, utility


# <codecell>

# Dataset selection
dataset = '20241104'
dataset_path = os.path.join('../', dataset)
print('Available acquisitions:', datareading.find_available_videos(dataset_path))

# Acquisition selection
acquisition = '100mid_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)

datareading.describe(dataset, acquisition, verbose=3)


# <codecell>

# Parameters definition
rivfinding_params = {
    'resize_factor': 2,
    'borders_min_distance': 8,
    'max_borders_luminosity_difference': 50,
    'max_rivulet_width': 100.,
}
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

framenumbers = np.arange(100)
roi = 250, None, 1150, None



# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')
x = np.arange(width * rivfinding_params['resize_factor']) / rivfinding_params['resize_factor']

rivs = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

frame = frames[80]

frames_verticleaned = frames - np.median(frames, axis=(0, 1), keepdims=True)
frames_verticleaned -= frames_verticleaned.min()
frame_verticleaned = frames_verticleaned[80]

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
axes[0].imshow(frame)
axes[1].imshow(frame_verticleaned)


# <codecell>

cos = rivuletfinding.cos_framewise(frame, **rivfinding_params)
borders = rivuletfinding.borders_via_peakfinder(frame, **rivfinding_params)
b1, b2 = borders
bol = rivuletfinding.bol_framewise_opti(frame, borders_for_this_frame=borders, **rivfinding_params)

cos2 = rivuletfinding.cos_framewise(frame_verticleaned, **rivfinding_params)
borders2 = rivuletfinding.borders_via_peakfinder(frame_verticleaned, **rivfinding_params)
b12, b22 = borders
bol2 = rivuletfinding.bol_framewise_opti(frame_verticleaned, borders_for_this_frame=borders, **rivfinding_params)


# <codecell>

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

ax = axes[0]
ax.imshow(frame)
ax.plot(x, cos)
ax.plot(x, b1)
ax.plot(x, b2)
ax.plot(x, bol)

ax = axes[1]
ax.imshow(frame_verticleaned)
ax.plot(x, cos2)
ax.plot(x, b12)
ax.plot(x, b22)
ax.plot(x, bol2)


# <codecell>




# <codecell>




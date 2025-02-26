# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from tools import set_verbose, datareading, datasaving, utility, rivuletfinding

utility.configure_mpl()


# <codecell>

### Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

### Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

### Acquisition selection
acquisition = '100mid_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

### Parameters definition

# conversion factor
px_per_mm = 33.6
px_per_um = px_per_mm * 1e3

# parameters to find the rivulet
rivfinding_params = {
    'resize_factor': 2,
    'borders_min_distance': 8,
    'max_borders_luminosity_difference': 50,
    'max_rivulet_width': 100.,
}

# portion of the video that is of interest to us
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

# framenumbers = np.arange(100)
if dataset == '20241104':
    roi = 250, None, 1150, None



# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

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




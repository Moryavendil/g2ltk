# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import datareading, datasaving, utility, rivuletfinding

utility.configure_mpl()


# <codecell>

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = datareading.find_dataset(None)
datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)


# <codecell>

### Acquisition selection
acquisition = 'm150'
acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)


# <codecell>

### Parameters definition

# parameters to find the rivulet
rivfinding_params = {
    'resize_factor': 2,
    'remove_median_bckgnd_zwise': True,
    'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
    'white_tolerance': 70,
    'borders_min_distance': 2.,
    'max_rivulet_width': 150.,
}

# portion of the video that is of interest to us
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y


# <codecell>

# take a few framenumbers, we just want to do a test
i_frametest = 354

i_line_test = 3947

framenumbers = np.arange(400)


# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

frame = frames[i_frametest].copy()
frame -= frame.min()

kwargs = {**rivfinding_params}
for key in rivuletfinding.default_kwargs.keys():
    if not key in kwargs.keys():
        kwargs[key] = rivuletfinding.default_kwargs[key]

from scipy.ndimage import gaussian_filter


# <codecell>

### Step 1: Alter the image
# 1.1 : resize (if needed)
l = datareading.resize_frame(frame, resize_factor=kwargs['resize_factor'])
l_origin = l.copy()
l_origin -= l_origin.min()

# 1.2 remove z median
if kwargs['remove_median_bckgnd_zwise']:
    l = l - np.median(l, axis=(0), keepdims=True)
l_bckgnd_rmved = l.copy()
l_bckgnd_rmved -= l_bckgnd_rmved.min()

# 1.3 gaussian blur
if kwargs['gaussian_blur_kernel_size'] is not None:
    sigma = kwargs['gaussian_blur_kernel_size']
    l = gaussian_filter(l, sigma=sigma)
l_filtered = l.copy()
l_filtered -= l_filtered.min()

# 1.4 remove minimum
l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255


# <codecell>

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
axes[0].imshow(l_origin, aspect='auto')
axes[1].imshow(l_bckgnd_rmved, aspect='auto')
axes[2].imshow(l_filtered, aspect='auto')
plt.tight_layout()


# <codecell>

l_line = l_filtered[:, i_line_test]


# <codecell>

### Step 2: Obtain the shadow representation
height, = l_line.shape

# 2.1 build the z coordinate ('horizontal' in real life)
z1D = np.arange(height)

# 2.2 discriminate the channel (white zone in the image)
max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
threshold_l = max_l - kwargs['white_tolerance']
is_channel = l >= threshold_l

# 2.3 identify the channel borders
top = np.argmax(is_channel, axis=0).max()
bot = height - np.argmax(is_channel[::-1], axis=0).min()

# 2.4 Obtain the z and shadow image for the channel
s_channel = 255 - l_line[top:bot] # shadowisity (255 - luminosity)
z_channel = z1D[top:bot]           # z coordinate


# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)

ax = axes
ax.plot(z1D, l_line)
ax.axvline(z1D[top], color='k', ls=':', alpha=.8)
ax.axvline(z1D[bot-1]+utility.step(z1D), color='k', ls=':', alpha=.8)

plt.tight_layout()


# <codecell>

### Step 3: Find the borders
bmax = rivuletfinding.bimax_by_peakfinder(z_channel, s_channel, distance=kwargs['borders_min_distance'], prominence=1)

z1, y1, z2, y2 = bmax

zdiff:np.ndarray = np.abs(z1 - z2)
space_ok = zdiff < kwargs['max_rivulet_width']
utility.log_info(f"space_ok (Dz < {kwargs['max_rivulet_width']}): {space_ok}")

ydiff:np.ndarray = np.abs(y1 - y2)
ydiff_ok = ydiff < kwargs['max_borders_luminosity_difference']
utility.log_info(f"ydiff_ok (Ds < {kwargs['max_borders_luminosity_difference']}): {ydiff_ok}")


# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)

ax = axes
ax.plot(z_channel, s_channel, color='k')
ax.scatter(z1, y1, color='b')
ax.axvline(z1, color='b', ls='--', alpha=.8)
ax.scatter(z2, y2, color='g')
ax.axvline(z2, color='g', ls='--', alpha=.8)


# <codecell>

### Step 4: Compute the BOL

rivulet_zone = (z_channel >= z1) * (z_channel <= z2)

# if np.sum(rivulet_zone) == 0:
#     return z1

z_rivulet_zone = z_channel[rivulet_zone]
l_rivulet_zone = 255 - s_channel[rivulet_zone]

# Center of mass
weights_offset = np.min(l_rivulet_zone) - 1e-5
weights = np.maximum(l_rivulet_zone - weights_offset, 0)
position = np.sum(z_rivulet_zone * weights) / np.sum(weights)

# FWHM
threshold = weights.max() / 2
fwhm = np.sum(weights > threshold) * utility.step(z1D)


# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)

ax = axes
ax.plot(z_rivulet_zone, l_rivulet_zone, color='k')
ax.axhline(weights_offset, color='gray', ls='--', alpha=.8)
ax.axvline(position, color='r', ls='--', alpha=.8)
ax.axvline(rivuletfinding.bol_linewise(z1D, l_line, **kwargs), color='k', ls=':', alpha=.8)


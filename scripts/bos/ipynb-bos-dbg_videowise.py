# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import set_verbose, datareading, datasaving, utility, rivuletfinding

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
acquisition = 'highQ_naturel'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

### Parameters definition

# conversion factor
px_per_mm = 220/80
px_per_um = px_per_mm * 1e3

# parameters to find the rivulet
rivfinding_params = {
    'resize_factor': 2,
    'white_tolerance': 70,
    'rivulet_size_factor': 2.,
    'remove_median_bckgnd_zwise': True,
    'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
}

# portion of the video that is of interest to us
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

if dataset=='250225':
    roi = [None, 10, 1980, -10]



# <codecell>

# take a few framenumbers, we just want to do a test
framenumbers = np.arange(200)


# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

kwargs = {**rivfinding_params}
for key in rivuletfinding.default_kwargs.keys():
    if not key in kwargs.keys():
        kwargs[key] = rivuletfinding.default_kwargs[key]

from scipy.ndimage import gaussian_filter

i_frametest = 100


# <codecell>

### Step 1: Alter the image
# 1.1 : resize (if needed)
l = datareading.resize_frames(frames, resize_factor=kwargs['resize_factor'])
l_origin = l.copy() - l.min() # DEBUG

# 1.2 remove z median
if kwargs['remove_median_bckgnd_zwise']:
    l = l - np.median(l, axis=(0, 1), keepdims=True)
l_bckgnd_rmved = l.copy() - l.min() # DEBUG

# 1.3 gaussian blur
if kwargs['gaussian_blur_kernel_size'] is not None:
    sigma = kwargs['gaussian_blur_kernel_size']
    for i_t in range(len(l)):
        l[i_t] = gaussian_filter(l[i_t], sigma=sigma)
l_filtered = l.copy() - l.min() # DEBUG

# 1.4 remove minimum
l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255


# <codecell>

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
axes[0].imshow(l_origin[i_frametest], aspect='auto')
axes[1].imshow(l_bckgnd_rmved[i_frametest], aspect='auto')
axes[2].imshow(l_filtered[i_frametest], aspect='auto')
plt.tight_layout()


# <codecell>

### Step 2: Obtain the shadow representation
length, height, width = l.shape

# 2.1 build the z coordinate ('horizontal' in real life)
x1D = np.arange(width)
z1D = np.arange(height)
z2D = np.repeat(z1D, width).reshape((height, width))
z3D = np.repeat(z2D[None, :, :], length, axis=0)

# 2.2 discriminate the channel (white zone in the image)
max_l = np.percentile(l, 95, axis=(0, 1), keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
threshold_l = max_l - kwargs['white_tolerance']
is_channel = l >= threshold_l

# 2.3 identify the channel borders
top = np.argmax(is_channel, axis=1).max()
bot = height - np.argmax(is_channel[::-1], axis=1).min()

# 2.4 Obtain the z and shadow image for the channel
s_channel = 255 - l[:, top:bot, :] # shadowisity (255 - luminosity)
z_channel = z3D[:, top:bot, :]           # z coordinate


# <codecell>

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
axes[0].imshow(l[i_frametest], aspect='auto')
axes[1].imshow(is_channel[i_frametest], aspect='auto')
axes[2].imshow(z_channel[i_frametest], aspect='auto')
axes[3].imshow(s_channel[i_frametest], aspect='auto')
plt.tight_layout()


# <codecell>

### Step 3: Obtain the border of the rivulet
# 3.1 get a threshold to know where is the reivulet
s_channel_max = np.amax(s_channel, axis=(1), keepdims=True)
s_channel_median = np.median(s_channel, axis=(1), keepdims=True)
# The threshold above which we count the rivulet
# This should be ap[prox. the half-max of the rivulet shadow
s_channel_threshold = (s_channel_max + s_channel_median) / 2

# 3.2 compute the upper-approximate the half-width of the rivulet
approx_rivulet_size = np.sum(s_channel >= s_channel_threshold, axis=1, keepdims=True) * kwargs['rivulet_size_factor']

# 3.3 the approximate position (resolution = size of the rivulet, a minima 1 pixel)
riv_pos_approx = np.argmax(s_channel, axis=1, keepdims=True) + z_channel[:, 0:1:, :]

# utility.log_info(f's {s_channel.shape} --(argmax)-> {np.argmax(s_channel, axis=1, keepdims=True).shape}')
# utility.log_info(f'+ z {z_channel[:, 0:1:, :].shape}')
# utility.log_info(f'= riv_pos_approx {riv_pos_approx.shape}')

# 3.4 identify the zone around the rivulet
z_top = np.maximum(riv_pos_approx - approx_rivulet_size, np.zeros_like(riv_pos_approx))
z_bot = np.minimum(riv_pos_approx + approx_rivulet_size, s_channel.shape[0] * np.ones_like(riv_pos_approx))
around_the_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

# utility.log_info(f'riv_pos_approx {riv_pos_approx.shape} + approx_rivulet_size {approx_rivulet_size.shape}')
# utility.log_info(f'? np.zeros_like(riv_pos_approx) {np.zeros_like(riv_pos_approx).shape}')
# utility.log_info(f'= z_top {z_top.shape}')



# <codecell>

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
ax = axes[0]
ax.imshow(s_channel[i_frametest], aspect='auto')
ax = axes[1]
ax.plot(x1D, s_channel_max[i_frametest, 0], color='r', ls='--', label='max')
ax.plot(x1D, s_channel_median[i_frametest, 0], color='b', ls=':', label='median')
ax.plot(x1D, s_channel_threshold[i_frametest, 0], color='k', ls='-', label='threshold')

ax = axes[2]
ax.imshow((s_channel >= s_channel_threshold)[i_frametest], aspect='auto')
axwidth = ax.twinx()
axwidth.plot(x1D, np.sum(s_channel >= s_channel_threshold, axis=1, keepdims=True)[i_frametest, 0], color='w', ls='-', label='threshold')
axwidth.set_ylim(0, np.sum(s_channel >= s_channel_threshold, axis=1, keepdims=True).max()*1.2)

ax = axes[3]
ax.imshow((s_channel * around_the_rivulet)[i_frametest], aspect='auto')
plt.tight_layout()


# <codecell>

### Step 4: compute the center fo mass
# 4.1 remove the background near the rivulet
s_bckgnd_near_rivulet = np.amin(s_channel, axis=1, where=around_the_rivulet, initial=255, keepdims=True) * (1-1e-5)

# 4.2 compute the weights to compute the COM
weights = (s_channel - s_bckgnd_near_rivulet) * around_the_rivulet

# 4.3 The COM rivulet with sub-pixel resolution
rivulet = np.sum(z_channel * weights, axis=1) / np.sum(weights, axis=1)


# <codecell>

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
ax = axes[0]
ax.imshow(s_channel[i_frametest], aspect='auto')

ax = axes[1]
ax.plot(x1D, s_bckgnd_near_rivulet[i_frametest, 0], color='k', ls='-', label='threshold')

ax = axes[2]
ax.imshow(weights[i_frametest], aspect='auto')
ax.plot(x1D, rivulet[i_frametest], color='k')



# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex=True, sharey=False)
ax = axes[0,0]
ax.imshow(s_channel[i_frametest], aspect='auto')
ax.plot(x1D, rivulet[i_frametest], color='k', label='step by step')
ax.plot(x1D, rivuletfinding.cos_videowise(frames, **kwargs)[i_frametest]*2, color='w', ls='--', label='videowise')
ax.plot(x1D, rivuletfinding.cos_framewise(frames[i_frametest], **kwargs)*2, color='r', ls=':', label='framewise')
ax.legend()


# <codecell>

### BENCHMARK
import time

t00 = time.time()
for i in range(len(frames)):
    rivuletfinding.cos_framewise(frames[i], **kwargs)
t01 = time.time()
time_frames = t01-t00
utility.log_info(f'Time (no parallel): {round(time_frames, 3)} seconds')

t10 = time.time()
rivuletfinding.cos_videowise(frames, **kwargs)
t11 = time.time()
time_video = t11-t10
utility.log_info(f'Time (parallel):    {round(time_video, 3)} seconds')



# <codecell>

rivuletfinding.cos_videowise(frames, **kwargs).shape


# <codecell>




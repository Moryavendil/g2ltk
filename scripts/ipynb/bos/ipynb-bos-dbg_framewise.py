# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

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
acquisition = None
acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)


# <codecell>

# Parameters definition
# =====================

# Data gathering
# --------------

### portion of the video that is of interest to us
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

# Rivulet detection
# -----------------

### parameters to find the rivulet
rivfinding_params = {
    'resize_factor': 2,
    'white_tolerance': 70,
    'rivulet_size_factor': 1.,
    'remove_median_bckgnd_zwise': True,
    'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
}


# <codecell>

# take a few framenumbers, we just want to do a test
framenumbers = np.arange(200)


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

i_frametest = 100

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

### Step 2: Obtain the shadow representation
height, width = l.shape

# 2.1 build the z coordinate ('horizontal' in real life)
x1D = np.arange(width)
z1D = np.arange(height)
z2D = np.repeat(z1D, width).reshape((height, width))

# 2.2 discriminate the channel (white zone in the image)
max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
threshold_l = max_l - kwargs['white_tolerance']
is_channel = l >= threshold_l

# 2.3 identify the channel borders
top = np.argmax(is_channel, axis=0).max()
bot = height - np.argmax(is_channel[::-1], axis=0).min()

# 2.4 Obtain the z and shadow image for the channel
s_channel = 255 - l[top:bot, :] # shadowisity (255 - luminosity)
z_channel = z2D[top:bot, :]           # z coordinate


# <codecell>

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
axes[0].imshow(l, aspect='auto')
axes[1].imshow(is_channel, aspect='auto')
axes[2].imshow(z_channel, aspect='auto')
axes[3].imshow(s_channel, aspect='auto')
plt.tight_layout()


# <codecell>

### Step 3: Obtain the border of the rivulet
# 3.1 get a threshold to know where is the reivulet
s_channel_max = np.amax(s_channel, axis=0, keepdims=True)
s_channel_median = np.median(s_channel, axis=0, keepdims=True)
# The threshold above which we count the rivulet
# This should be ap[prox. the half-max of the rivulet shadow
s_channel_threshold = (s_channel_max + s_channel_median) / 2

# 3.2 compute the upper-approximate the half-width of the rivulet
approx_rivulet_size = np.sum(s_channel >= s_channel_threshold, axis=0) * kwargs['rivulet_size_factor']

# 3.3 the approximate position (resolution = size of the rivulet, a minima 1 pixel)
riv_pos_approx = np.argmax(s_channel, axis=0, keepdims=True) + z_channel[0, :]

# 3.4 identify the zone around the rivulet
z_top = np.maximum(riv_pos_approx - approx_rivulet_size, np.zeros_like(riv_pos_approx))
z_bot = np.minimum(riv_pos_approx + approx_rivulet_size, s_channel.shape[0] * np.ones_like(riv_pos_approx))
around_the_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

print(np.argmax(s_channel, axis=0, keepdims=True).shape)
print(z_channel[0, :].shape)
print(riv_pos_approx.shape)
print(z_top.shape)
print(z_channel.shape)


# <codecell>

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
ax = axes[0]
ax.imshow(s_channel, aspect='auto')
ax = axes[1]
ax.plot(x1D, s_channel_max[0], color='r', ls='--', label='max')
ax.plot(x1D, s_channel_median[0], color='b', ls=':', label='median')
ax.plot(x1D, s_channel_threshold[0], color='k', ls='-', label='threshold')

ax = axes[2]
ax.imshow(s_channel >= s_channel_threshold, aspect='auto')
axwidth = ax.twinx()
axwidth.plot(x1D, np.sum(s_channel >= s_channel_threshold, axis=0), color='w', ls='-', label='threshold')
axwidth.set_ylim(0, np.sum(s_channel >= s_channel_threshold, axis=0).max()*1.2)

ax = axes[3]
ax.imshow(s_channel * (around_the_rivulet), aspect='auto')
plt.tight_layout()


# <codecell>

### Step 4: compute the center fo mass
# the background near the rivulet
s_bckgnd_near_rivulet = np.amin(s_channel, axis=0, where=around_the_rivulet, initial=255, keepdims=True) * (1-1e-5)

# the weights to compute the COM
weights = (s_channel - s_bckgnd_near_rivulet) * around_the_rivulet

# The COM rivulet with sub-pixel resolution
rivulet = np.sum(z_channel * weights, axis=0) / np.sum(weights, axis=0)


# <codecell>

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
ax = axes[0]
ax.imshow(weights, aspect='auto')

ax = axes[1]
# ax.plot(x1D, s_bckgnd_near_rivulet[0], color='r', ls='--', label='max')
# ax.plot(x1D, s_channel_median[0], color='b', ls=':', label='median')
ax.plot(x1D, s_bckgnd_near_rivulet[0], color='k', ls='-', label='threshold')

ax = axes[2]
ax.imshow(weights, aspect='auto')
ax.plot(x1D, rivulet, color='k')


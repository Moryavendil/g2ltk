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
acquisition = 'a340'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

### Parameters definition

# parameters to find the rivulet
rivfinding_params = {
    'resize_factor': 2,
    'remove_median_bckgnd_zwise': True,
    'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
    'white_tolerance': 70,
    'borders_min_distance': 5.,
    'borders_width': 6.,
    'max_rivulet_width': 150.,
}

# portion of the video that is of interest to us
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y



# <codecell>

# take a few framenumbers, we just want to do a test
framenumbers = np.arange(400)


# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)


# <codecell>

i_frametest = 354

frame = frames[i_frametest].copy()
frame -= frame.min()

borders_for_this_frame = None

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
x1D = np.arange(width) / kwargs['resize_factor']
z1D = np.arange(height) / kwargs['resize_factor']
z2D = np.repeat(z1D, width).reshape((height, width))

# 2.2 discriminate the channel (white zone in the image)
max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
threshold_l = max_l - kwargs['white_tolerance']
is_channel = l >= threshold_l

# 2.3 identify the channel borders
top = np.argmax(is_channel, axis=0).max()
bot = height - np.argmax(is_channel[::-1], axis=0).min()

# 2.4 Obtain the z and shadow image for the channel
l_channel = l[top:bot, :] # luminosity
z_channel = z2D[top:bot, :]           # z coordinate


# <codecell>

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
axes[0].imshow(l, aspect='auto')
axes[1].imshow(is_channel, aspect='auto')
axes[2].imshow(z_channel, aspect='auto')
axes[3].imshow(l_channel, aspect='auto')
plt.tight_layout()


# <codecell>

if borders_for_this_frame is None:
    borders_for_this_frame:np.ndarray = rivuletfinding.borders_framewise(frame, **kwargs)
z_top = borders_for_this_frame[0,:]
z_bot = borders_for_this_frame[1,:]


# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
ax = axes
ax.imshow(l_channel, aspect='auto', extent=utility.correct_extent(x1D, z1D))
ax.plot(x1D, z_top, color='w')
ax.plot(x1D, z_bot, color='w')


# <codecell>


inside_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

bckgnd_inside_rivulet = np.amin(l_channel, axis=0, where=inside_rivulet, initial=255, keepdims=True) - 1e-4

# the weights to compute the COM
weights = (l_channel - bckgnd_inside_rivulet) * inside_rivulet

# The BOL rivulet with sub-pixel resolution
# this handles the tricky size of 0-width rivulet (when the two borders are at the same point, it happens for some shitty videos
nonzerosum = np.sum(weights, axis=0) > 0

rivulet = np.empty(width, dtype=float)
rivulet[nonzerosum] = np.sum(z_channel * weights, axis=0)[nonzerosum] / np.sum(weights, axis=0)[nonzerosum]
rivulet[np.bitwise_not(nonzerosum)] = ((z_bot+z_top)/2)[np.bitwise_not(nonzerosum)]


# FWHM
threshold = np.max(weights, axis=0, keepdims=True) / 2

fwhm = np.empty(width, dtype=float)
fwhm[nonzerosum] = np.sum(weights > threshold, axis=0)[nonzerosum] * utility.step(z1D)
fwhm[np.bitwise_not(nonzerosum)] = ((z_bot+z_top)/2)[np.bitwise_not(nonzerosum)]


# <codecell>

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
ax = axes[0]
ax.imshow(inside_rivulet * l_channel, aspect='auto', extent=utility.correct_extent(x1D, z1D))
ax.plot(x1D, rivulet, color='k', ls='-')
ax.plot(x1D, rivulet+fwhm/2, color='w', ls='--')
ax.plot(x1D, rivulet-fwhm/2, color='w', ls='--')
ax.plot(x1D, z_top, color='w', ls=':')
ax.plot(x1D, z_bot, color='w', ls=':')

ax = axes[1]
ax.imshow(l_channel, aspect='auto', extent=utility.correct_extent(x1D, z1D))
ax.plot(x1D, rivulet, color='k', ls='-')
ax.plot(x1D, rivulet+fwhm/2, color='w', ls='--')
ax.plot(x1D, rivulet-fwhm/2, color='w', ls='--')
ax.plot(x1D, z_top, color='w', ls=':')
ax.plot(x1D, z_bot, color='w', ls=':')


ax = axes[2]
ax.plot(x1D, z_bot-z_top)
ax.plot(x1D, fwhm)



# <codecell>




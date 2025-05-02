# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt

from g2ltk import datareading, datasaving, utility, logging
utility.configure_mpl()


# <codecell>

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = '40evo'

datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)
dataset_path = datareading.generate_dataset_path(dataset)


# <codecell>

### Acquisition selection
acquisition = 'a280'
acquisition_path = datareading.generate_acquisition_path(acquisition, dataset)


# <codecell>

### Parameters definition

# conversion factor
px_per_mm = (1563/30+1507/30)/2
px_per_um = px_per_mm * 1e3
fr_per_s = datareading.get_acquisition_frequency(acquisition_path)
if dataset=='40evo':
    fr_per_s = (40 * 100)
fr_per_ms = fr_per_s / 1000

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
if dataset=='40evo' and acquisition=='a350':
    framenumbers = np.arange(450)



# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])
z = datareading.get_z_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
w_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)

z_raw = z_raw[:,::-1]
w_raw = w_raw[:,::-1]


# <codecell>

z_tmp = z_raw.copy()
w_tmp = w_raw.copy()


# <codecell>

from scipy.ndimage import gaussian_filter
blur_t_frame = 1 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 5 # blur in space (px).
sigma_x = blur_x_px


z_filtered = gaussian_filter(z_tmp, sigma=(sigma_t, sigma_x))
w_filtered = gaussian_filter(w_tmp, sigma=(sigma_t, sigma_x))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
imshow_kw = {'aspect': 'auto', 'origin': 'upper'}

ax = axes[0, 0]
ax.set_title('Z (normal)')
imz = ax.imshow(z_tmp, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[0, 1]
ax.set_title('Z (smoothed)')
imz = ax.imshow(z_filtered, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1, 0]
ax.set_title('W (normal)')
imz = ax.imshow(w_tmp, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1, 1]
ax.set_title('W (smoothed)')
imz = ax.imshow(w_filtered, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

plt.tight_layout()

apply_gaussianfiler = True
if apply_gaussianfiler:
    z_tmp = z_filtered.copy()
    w_tmp = w_filtered.copy()
    utility.log_info('Gaussian filtering correction made')
else:
    utility.log_info('No gaussian filtering correction made')


# <codecell>

fn = 280

img = datareading.get_frame(acquisition_path, framenumber = framenumbers[fn], subregion=roi)
img = img[:,::-1]

img_lmin = np.percentile(img.flatten(), 1)
img_lmax = np.percentile(img.flatten(), 99)

from matplotlib.colors import Normalize

img = Normalize(vmin=img_lmin, vmax=img_lmax, clip=True)(img)

x_mm = x / px_per_mm
z_mm = z / px_per_mm

Z_mm = z_tmp / px_per_mm
W_mm = w_tmp / px_per_mm

Z_main = Z_mm[fn]
W_main = W_mm[fn]


# <codecell>

Zoffset = 3
z_mm -= Zoffset
Z_main -= Zoffset


# <codecell>


fig, axs = plt.subplots(1,4, squeeze=False, sharey=True, sharex=True)

ax = axs[0,0]
ax.imshow(img.T, aspect='equal', extent=utility.correct_extent(z_mm, x_mm), cmap='binary_r')

ax = axs[0,1]
# ax.plot(Z, x_mm, color= 'w', lw=2)
# ax.plot(Z + W/2, x_mm, color='w', lw=2)
# ax.plot(Z - W/2, x_mm, color='w', lw=2)
ax.plot(Z_main, x_mm, color=utility.color_z, lw=1.5)
ax.plot(Z_main + W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.plot(Z_main - W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.set_aspect('equal')


ax = axs[0,2]
ax.plot(Z_main, x_mm, color=utility.color_z, lw=1.5)
ax.set_aspect('equal')

ax = axs[0,3]
ax.plot(1.5 + W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.plot(1.5 - W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.set_aspect('equal')

ax.set_ylim(25, 0)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['0', '1', '2', '3'])
ax.set_xlim(0,3)

# utility.save_graphe('testillustr')



# <codecell>

x1 = 6
x2 = 6+6.5

fig, axs = plt.subplots(1,4, squeeze=False, sharey=True, sharex=True)

ax = axs[0,0]
ax.imshow(img.T, aspect='equal', extent=utility.correct_extent(z_mm, x_mm), cmap='binary_r', interpolation='bilinear')

ax = axs[0,1]
# ax.plot(Z, x_mm, color= 'w', lw=2)
# ax.plot(Z + W/2, x_mm, color='w', lw=2)
# ax.plot(Z - W/2, x_mm, color='w', lw=2)
ax.plot(Z_main, x_mm, color=utility.color_z, lw=1.5)
ax.plot(Z_main + W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.plot(Z_main - W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.set_aspect('equal')


ax = axs[0,2]
ax.plot(Z_main, x_mm, color=utility.color_z, lw=1.5)
ax.set_aspect('equal')

ax = axs[0,3]
ax.plot(1.5 + W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.plot(1.5 - W_main / 2, x_mm, color=utility.color_w, lw=1.5)
ax.set_aspect('equal')

ax.set_ylim(x2, x1)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['0', '1', '2', '3'])
ax.set_xlim(0,3)


# <codecell>

x1 = 6
x2 = 6+6.5

lw = 2 

fig, axs = plt.subplots(1,3, squeeze=False, sharey=True, sharex=True)

ax = axs[0,0]
ax.imshow(img.T, aspect='equal', extent=utility.correct_extent(z_mm, x_mm), cmap='binary_r', interpolation='bilinear')

ax = axs[0,1]
ax.imshow(img.T, aspect='equal', extent=utility.correct_extent(z_mm, x_mm), vmin = img.min()-1, cmap='binary_r', interpolation='bilinear')
# ax.plot(Z, x_mm, color= 'w', lw=2)
# ax.plot(Z + W/2, x_mm, color='w', lw=2)
# ax.plot(Z - W/2, x_mm, color='w', lw=2)
ax.plot(Z_main, x_mm, color=utility.color_z, lw=lw)
ax.plot(Z_main + W_main / 2, x_mm, color=utility.color_w, lw=lw)
ax.plot(Z_main - W_main / 2, x_mm, color=utility.color_w, lw=lw)
ax.set_aspect('equal')


ax = axs[0,2]
ax.plot(Z_main, x_mm, color=utility.color_z, lw=lw)
ax.plot(Z_main + W_main / 2, x_mm, color=utility.color_w, lw=lw)
ax.plot(Z_main - W_main / 2, x_mm, color=utility.color_w, lw=lw)
ax.set_aspect('equal')

ax.set_ylim(x2, x1)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['0', '1', '2', '3'])
ax.set_xlim(0,3)

# utility.save_graphe('textimg')



# <codecell>




# <codecell>




# <codecell>




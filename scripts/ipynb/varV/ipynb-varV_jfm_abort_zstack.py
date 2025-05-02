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
acquisition = 'a260'
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

z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
w_raw = datasaving.fetch_or_generate_data('fwhmol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)

z_raw = z_raw[:,::-1]
w_raw = w_raw[:,::-1]


# <codecell>

z_tmp = z_raw.copy()
w_tmp = w_raw.copy()


# <codecell>

from scipy.ndimage import gaussian_filter
blur_t_frame = 4 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 20 # blur in space (px).
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

# Spatial cleaning : parabolic fit on the data
from scipy.signal import savgol_filter

# get temporal mean
z_xprofile = np.mean(z_tmp, axis=0)

# linear fit
rivs_fit_spatial = np.polyfit(x, z_xprofile, deg = 1)
utility.log_info(f'Position spatial drift linear estimation: {round(np.poly1d(rivs_fit_spatial)(x).max() - np.poly1d(rivs_fit_spatial)(x).min(),2)} px')

# savgol
savgol_width = len(x)//10 + (1 if len(x)%2 == 0 else 0)

"""
xcorrect can be 3 things
 * None: we do not want correction : it is biaised, or we are interested in the mean-time features
 * 'linear': we do a linear fit and remove everything from it
 * 'smoothed': we do a smoothing avec the average and remove that
 * 'total': total removal of the mean value
"""
xcorrect = 'linear'
if xcorrect is None:
    utility.log_info('No spatial correction made')
elif xcorrect == 'linear':
    utility.log_info('Linear spatial correction made')
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_spatial)(x), axis=0)
elif xcorrect == 'smoothed':
    utility.log_info(f'Smoothed spatial correction made (savgol-{savgol_width}-2)')
    z_tmp = z_tmp - savgol_filter(z_xprofile, savgol_width, 2)
elif xcorrect == 'total':
    utility.log_info('Total spatial correction made')
    z_tmp = z_tmp - z_xprofile
else:
    utility.log_warning(f'What do you mean by xcorrect={xcorrect} ?')
    utility.log_info('No spatial correction made')


# plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=utility.figsize('double'))
ax = axes[0]
ax.plot(x, z_xprofile, color='k', alpha=0.5, label='old time-averaged riv position')
ax.plot(x, np.poly1d(rivs_fit_spatial)(x), color='r', alpha=0.5, label=f'linear fit')
ax.plot(x, savgol_filter(z_xprofile, savgol_width, 2), color='b', alpha=0.5, label=f'smooth')
ax.set_ylabel('z (px)')
ax.legend()

ax = axes[1]
ax.plot(x, z_tmp.mean(axis=0), color='k', label='New time-averaged riv position')
ax.set_xlabel('x (px)')
ax.set_ylabel('z (px)')
ax.legend()
plt.tight_layout()


# <codecell>

# Temporal cleaning : parabolic fit on the data

# get spatial mean
z_tprofile = np.mean(z_tmp, axis=1)

# fit
rivs_fit_temporal = np.polyfit(t, z_tprofile, deg = 2)

# correct
do_tcorrect= False
if do_tcorrect:
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_temporal)(t), axis=1)
else:
    utility.log_info('No temporal correction made')

# plot
plt.figure(figsize=(8,3))
ax = plt.gca()
if do_tcorrect:
    ax.plot(t, z_tprofile, color='k', alpha=0.5, label='old space-averaged riv position')
    plt.plot(t, np.poly1d(rivs_fit_temporal)(t), color='r', alpha=0.5, label=f'paraboloidal fit')
ax.plot(t, z_tmp.mean(axis=1), color='k', label='space-averaged riv position')
ax.set_xlabel('t (s)')
ax.set_ylabel('z (px)')
ax.legend()
plt.tight_layout()


# <codecell>

Z = z_tmp.copy()
W = w_tmp.copy()

W = W / np.sqrt(1 + np.gradient(Z, x, axis=1)**2) # correction due to curvature

t_s = datareading.get_t_frames(acquisition_path, framenumbers) / fr_per_s
t_ms = t_s * 1000
x_mm = datareading.get_x_mm(acquisition_path, framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'], px_per_mm=px_per_mm)

Z_mm = Z / px_per_mm
W_mm = W / px_per_mm


# <codecell>

t1, t2 = None, None
x1 ,x2 = None, None
if dataset == '40evo':
    if acquisition == 'a260':
        L_X = 1/0.00290 / px_per_mm
        L_T = 100 / fr_per_ms
        
        t1 = int(61 + .25 * L_T*fr_per_ms)
        t2 = int(t1 + 2.5 * L_T*fr_per_ms)
        x1 = int(870 + L_X / 2  - .25 * L_X*px_per_mm*2 )
        x2 = int(x1 + 2.5 * L_X*px_per_mm*2)
        
Z_ = Z_mm[t1:t2, x1:x2]
W_ = W_mm[t1:t2, x1:x2]
t_ = t_ms[t1:t2]
x_ = x_mm[x1:x2]
t_ -= t_[0]
x_ -= x_[0]
X_, T_ = np.meshgrid(x_, t_)



# <codecell>


Zt = np.mean(Z_, axis = 0)
Zx = np.mean(Z_, axis = 1)
Wt = np.mean(W_, axis = 0)
Wx = np.mean(W_, axis = 1)

Zextr = .6 #.5 # Z_.max()
Wextr = .4

Zlim = [-Zextr, Zextr]
# Zlim = [-.5, .5]

cmap_z = 'PuOr_r'
cmap_w = 'viridis'


# <codecell>

fig, axes = plt.subplots(1, 5, squeeze=False, sharex=True, sharey=True)

ax = axes[0, 0]
ax.set_aspect('equal')
ni = 0
ax.plot(Z_[ni], x_, color=utility.color_z)
ax.plot(Z_[ni]+W_[ni]/2, x_, color=utility.color_w)
ax.plot(Z_[ni]-W_[ni]/2, x_, color=utility.color_w)

ax = axes[0, 1]
ax.set_aspect('equal')
ni = 25
ax.plot(Z_[ni], x_, color=utility.color_z)
ax.plot(Z_[ni]+W_[ni]/2, x_, color=utility.color_w)
ax.plot(Z_[ni]-W_[ni]/2, x_, color=utility.color_w)

ax = axes[0, 2]
ax.set_aspect('equal')
ni = 50
ax.plot(Z_[ni], x_, color=utility.color_z)
ax.plot(Z_[ni]+W_[ni]/2, x_, color=utility.color_w)
ax.plot(Z_[ni]-W_[ni]/2, x_, color=utility.color_w)

ax = axes[0, 3]
ax.set_aspect('equal')
ni = 75
ax.plot(Z_[ni], x_, color=utility.color_z)
ax.plot(Z_[ni]+W_[ni]/2, x_, color=utility.color_w)
ax.plot(Z_[ni]-W_[ni]/2, x_, color=utility.color_w)

ax = axes[0, 4]
ax.set_aspect('equal')
ni = 100
ax.plot(Z_[ni], x_, color=utility.color_z)
ax.plot(Z_[ni]+W_[ni]/2, x_, color=utility.color_w)
ax.plot(Z_[ni]-W_[ni]/2, x_, color=utility.color_w)


ax.set_ylim(ax.get_ylim()[::-1])


# <codecell>

fig, axes = plt.subplots(1, 1, squeeze=False, sharex=True, sharey=True, figsize=(6, 10))

import matplotlib as mpl

ax = axes[0, 0]
# ax.set_aspect('equal')
for p in range(4):
    ni = p*25
    midpos = (Z_[ni]).mean()
    
    cm = mpl.colormaps['hsv']
    
    # ax.plot(Z_[ni]-midpos, x_, color=utility.color_z)
    # ax.plot(Z_[ni]-midpos+W_[ni]/2, x_, color=cm(p/4), alpha=.5)
    # ax.plot(Z_[ni]-midpos-W_[ni]/2, x_, color=cm(p/4), alpha=.5)
    ax.plot(W_[ni]/2, x_, color=cm(p/4), alpha=.5)
    ax.plot(-W_[ni]/2, x_, color=cm(p/4), alpha=.5)


ax.set_ylim(ax.get_ylim()[::-1])


# <codecell>




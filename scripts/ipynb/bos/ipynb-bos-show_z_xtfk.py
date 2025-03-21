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
    'remove_median_bckgnd_zwise': True,
    'gaussian_blur_kernel_size': (1, 1), # (sigma_z, sigma_x)
    'white_tolerance': 70,
    'rivulet_size_factor': 1.,
}

# Data cleaning
# -------------

### Blurring
blur_t_frame = 1 # blur in time (frame).
blur_x_px = 1 # blur in space (px).
apply_gaussianfiler = True

### Spatial and temporal drift or 0-frequency correction
xcorrect = 'linear'
do_tcorrect= False

### 2D FFT parameters
fft_window_visu = 'hann'
zero_pad_factor_visu = (1, 1)
range_dB_visu = 80

# Data display
# ------------

# conversion factor
fr_per_s = datareading.get_acquisition_frequency(acquisition_path)
px_per_mm = 0.
px_per_um = px_per_mm * 1e3


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>

z_tmp = z_raw.copy()


# <codecell>

from scipy.ndimage import gaussian_filter
sigma_t = blur_t_frame
sigma_x = blur_x_px

z_filtered = gaussian_filter(z_tmp, sigma=(sigma_t, sigma_x))

if apply_gaussianfiler:
    z_tmp = z_filtered.copy()
    utility.log_info('Gaussian filtering correction made')
else:
    utility.log_info('No gaussian filtering correction made')

### plot

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False)
imshow_kw = {'aspect': 'auto', 'origin': 'upper'}

ax = axes[0, 0]
ax.set_title('Z (normal)')
imz = ax.imshow(z_raw, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[0, 1]
ax.set_title('Z (smoothed)')
imz = ax.imshow(z_filtered, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

plt.tight_layout()


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

### plot

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
if do_tcorrect:
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_temporal)(t), axis=1)
else:
    utility.log_info('No temporal correction made')

### plot

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

k_visu, f_visu = utility.fourier.rdual2d(x, t, zero_pad_factor=zero_pad_factor_visu)
Z_pw = utility.fourier.psd2d(Z, x, t, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)


# <codecell>

fig, axes = plt.subplots(1, 2, sharex='col', sharey='col', squeeze=False)
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto'}

ax = axes[0,0]
ax.set_title('Z (normal)')
imz = ax.imshow(Z, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[0,1]
vmax, vmin = utility.log_amplitude_range(Z_pw.max(), range_db=range_dB_visu)
im_zpw = ax.imshow(Z_pw, extent=utility.correct_extent(k_visu, f_visu), norm='log', vmax=vmax, vmin=vmin, cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cb = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [px^2/(px-1.frame-1)]')
utility.set_ticks_log_cb(cb, vmax, range_db=range_dB_visu)

ax.set_xlim(0, 1/10*1/20)
ax.set_ylim(-1/20, 1/5)


# <codecell>

### REAL UNITS

t_s = datareading.get_t_s(acquisition_path, framenumbers)
x_mm = datareading.get_x_mm(acquisition_path, framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'], px_per_mm=px_per_mm)

Z_mm = Z / px_per_mm


# <codecell>

k_visu_mm, f_visu_mm = utility.fourier.rdual2d(x_mm, t_s, zero_pad_factor=zero_pad_factor_visu)
Z_pw_mm = utility.fourier.psd2d(Z_mm, x_mm, t_s, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)


# <codecell>

fig, axes = plt.subplots(1, 2, figsize=utility.figsize('double'), sharex='col', sharey='col', squeeze=False)
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto'}

ax = axes[0,0]
imz = ax.imshow(Z_mm, extent=utility.correct_extent(x_mm, t_s), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [mm]')
ax.set_ylabel('$t$ [s]')
plt.colorbar(imz, ax=ax, label='$z$ [mm]')

ax = axes[0,1]
im_zpw = ax.imshow(Z_pw_mm, extent=utility.correct_extent(k_visu_mm, f_visu_mm), norm='log', vmax=Z_pw_mm.max(), vmin = Z_pw_mm.max()/10**(range_dB_visu / 10), cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [mm$^{-1}$]')
ax.set_ylabel(r'$f$ [Hz]')
cb = plt.colorbar(im_zpw, ax=ax, label=r'mm$^2$/(Hz.mm$^{-1}$)')
utility.set_ticks_log_cb(cb, vmax, range_db=range_dB_visu)


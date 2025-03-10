# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

# %matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import set_verbose, datareading, datasaving, utility
utility.configure_mpl()


# <codecell>

### Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# plt.close()


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
# acquisition = 'highQ_f0500a1200'
# acquisition = 'highQ_f0660a1200'
# acquisition = 'highQ_f3000a1200'
# acquisition = 'highQ_f4000a1200'
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

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

z_raw = datasaving.fetch_or_generate_data('cos', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)


# <codecell>

# Spatial cleaning : parabolic fit on the data

# get temporal mean
z_xprofile = np.mean(z_raw, axis=0)

# fit
rivs_fit_spatial = np.polyfit(x, z_xprofile, deg = 1)
print(f'Position spatial drift estimation: {round(np.poly1d(rivs_fit_spatial)(x).max() - np.poly1d(rivs_fit_spatial)(x).min(),2)} px')

# correct
do_xcorrect= True
z_x_treated = z_raw
if do_xcorrect:
    z_x_treated = z_raw - np.expand_dims(np.poly1d(rivs_fit_spatial)(x), axis=0)
else:
    print('No spatial correction made')


# plot
plt.figure(figsize=(8,3))
ax = plt.gca()
if do_xcorrect:
    ax.plot(x, z_xprofile, color='k', alpha=0.5, label='old time-averaged riv position')
    plt.plot(x, np.poly1d(rivs_fit_spatial)(x), color='r', alpha=0.5, label=f'linear fit')
ax.plot(x, z_x_treated.mean(axis=0), color='k', label='time-averaged riv position')
ax.set_xlabel('x (px)')
ax.set_ylabel('z (px)')
ax.legend()
plt.tight_layout()


# <codecell>

# Temporal cleaning : parabolic fit on the data

# get spatial mean
z_tprofile = np.mean(z_x_treated, axis=1)

# fit
rivs_fit_temporal = np.polyfit(t, z_tprofile, deg = 2)
print(f'Position temporal drift estimation: {round(np.poly1d(rivs_fit_temporal)(t).max() - np.poly1d(rivs_fit_temporal)(t).min(),2)} px')

# correct
do_tcorrect= False
z_xt_treated = z_x_treated
if do_tcorrect:
    z_xt_treated = z_x_treated - np.expand_dims(np.poly1d(rivs_fit_temporal)(t), axis=1)
else:
    print('No temporal correction made')


# plot
plt.figure(figsize=(8,3))
ax = plt.gca()
if do_tcorrect:
    ax.plot(t, z_tprofile, color='k', alpha=0.5, label='old space-averaged riv position')
    plt.plot(t, np.poly1d(rivs_fit_temporal)(t), color='r', alpha=0.5, label=f'paraboloidal fit')
ax.plot(t, z_xt_treated.mean(axis=1), color='k', label='space-averaged riv position')
ax.set_xlabel('t (s)')
ax.set_ylabel('z (px)')
ax.legend()
plt.tight_layout()


# <codecell>

from scipy.ndimage import gaussian_filter
blur_t_frame = 1 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 1 # blur in space (px).
sigma_x = blur_x_px

apply_gaussianfiler = True

z_filtered = gaussian_filter(z_xt_treated, sigma=(sigma_t, sigma_x))


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False)
imshow_kw = {'aspect': 'auto', 'origin': 'upper'}

ax = axes[0, 0]
ax.set_title('Z (normal)')
imz = ax.imshow(z_xt_treated, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
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

if apply_gaussianfiler:
    Z = z_filtered.copy()
else:
    Z = z_xt_treated.copy()


# <codecell>

fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=False, figsize=utility.figsize('simple'))
imshow_kw = {'origin':'upper', 'interpolation':'nearest', 'aspect':'auto'}

ax = axes[0,0]
imz = ax.imshow(Z, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

plt.tight_layout()


# <codecell>

k = utility.rdual(x)
f = utility.dual(t)

# Z_hat = utility.ft2d(Z, window='hann')
Z_pw = utility.fourier.psd2d(Z, x, t, window='hann')

range_db = 60


# <codecell>

fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
imshow_kw = {'origin':'upper', 
             'interpolation':'nearest', 
             'aspect':'auto'}
# ticks for fft
def get_cbticks(data, range_db):
    step_db = 5 * int(range_db / 25)
    z_ticks_db = np.arange(0, range_db, step_db)
    cbticks = [data.max() / 10**(att_db/10) for att_db in z_ticks_db]
    cbticklabels = ['ref' if att_db == 0 else f'-{att_db} dB' for att_db in z_ticks_db]
    return cbticks, cbticklabels

ax = axes
im_zpw = ax.imshow(Z_pw, extent=utility.correct_extent(k, f), norm='log', vmax=Z_pw.max(), vmin = Z_pw.max()/10**(range_db/10), cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cb = plt.colorbar(im_zpw, ax=ax, label='px^2/(px-1.frame-1)')
cbticks, cbticklabels = get_cbticks(Z_pw, range_db)
cb.ax.set_yticks(cbticks) ; cb.ax.set_yticklabels(cbticklabels)

ax.set_xlim(0, 1/20)
ax.set_ylim(-.1, .1)

# plt.tight_layout()


# <codecell>

### GO REAL UNITS
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

t_s = datareading.get_t_s(acquisition_path, framenumbers)
x_mm = datareading.get_x_mm(acquisition_path, framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'], px_per_mm=px_per_mm)

Z_mm = Z / px_per_mm
Z_pw_mm = utility.fourier.psd2d(Z_mm, x_mm, t_s, window='hann')

k_mm = utility.rdual(x_mm)
f_hz = utility.dual(t_s)


# <codecell>

fig, axes = plt.subplots(1, 2, figsize=utility.figsize('double'), sharex=False, sharey=False)
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto'}

ax = axes[0]
imz = ax.imshow(Z_mm, extent=utility.correct_extent(x_mm, t_s), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [mm]')
ax.set_ylabel('$t$ [s]')
plt.colorbar(imz, ax=ax, label='$z$ [mm]')

ax = axes[1]
im_zpw = ax.imshow(Z_pw_mm, extent=utility.correct_extent(k_mm, f_hz), norm='log', vmax=Z_pw_mm.max(), vmin = Z_pw_mm.max()/10**(range_db/10), cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [mm$^{-1}$]')
ax.set_ylabel(r'$f$ [Hz]')
ax.set_xlim(0, 1/5)
ax.set_ylim(-6, 6)
cb = plt.colorbar(im_zpw, ax=ax, label=r'mm$^2$/(Hz.mm$^{-1}$)')
cbticks, cbticklabels = get_cbticks(Z_pw_mm, range_db)
cb.ax.set_yticks(cbticks) ; cb.ax.set_yticklabels(cbticklabels)


# <codecell>




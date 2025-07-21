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
datareading.describe_dataset(dataset=dataset, videotype='mp4', makeitshort=True)


# <codecell>

# plt.close()


# <codecell>

### Acquisition selection
acquisition = '120_f750a500'
acquisition = '120_f1000a800'
acquisition = '120_f1500a600'
acquisition = '120_f1500a800'
acquisition = '120_f1750a800'
acquisition = '120_f2000a400'
acquisition = '120_f2000a800'
acquisition = '120_f2000a900'
acquisition = '120_f2250a400'
acquisition = '120_f2250a800'
acquisition = '120_f2500a500'
acquisition = '120_f2500a800'
acquisition = '120_f2500a900'
acquisition = '120_f2750a400'
acquisition = '120_f3000a400'
acquisition = '120_f3500a400'
acquisition = '120_f4000a400'
acquisition = '120_f4500a400'
acquisition = '120_f5000a400'
acquisition = '120_f5000a420'
acquisition = '120_f5000a440'
acquisition = '120_f5000a450'
acquisition = '120_f5000a460'
acquisition = '120_f5000a460_stable'
acquisition = '120_f5000a465_stable'
acquisition = '120_f5000a480'
acquisition = '120_f5000a500_stable'
acquisition = '120_f5000a700'
acquisition = '120_f5000a1000_stable'
acquisition = '120_f5500a500_stable'
acquisition = '120_f5500a700'
acquisition = '120_f5500a700_stable'
acquisition = '120_f6000a700_stable'
acquisition = '120_f6000a900_stable'
acquisition = '120_f6500a700_stable'
acquisition = '120_f7000a700_stable'
acquisition = '120_f8000a700_stable'
acquisition = '120_f9000a700_stable'
acquisition = '120_f10000a100'
acquisition = '120_f10000a200'
acquisition = '120_f10000a300'
acquisition = '120_f10000a400'
acquisition = '120_f10000a700_stable'
acquisition = '120_f10000a1000'
acquisition = '120_f12000a200'
acquisition = '120_f12000a600'
acquisition = '120_f12000a800'
acquisition = '120_f15000a800'
acquisition = '120_f20000a1000'
acquisition = '120_f30000a700'
acquisition = '120_f30000a1080'
acquisition = '120_f40000a900'
acquisition = '120_f50000a900'
acquisition = '120_f60000a900'
acquisition = '120_f80000a900'

# acquisition = '120_nat'
# acquisition = '120_nat_2'
# acquisition = '120_nat3'

# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
# acquisition = 'aaaaaaaaaaaa'
acquisition_path = datareading.generate_acquisition_path(acquisition, dataset=dataset)


# <codecell>

# Parameters definition
# =====================

# Data gathering
# --------------

framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y
if dataset=='pertu_250529':
    roi = None, 120//2, 3570//2, 180//2  #start_x, start_y, end_x, end_y
    if acquisition=='112_nat':
        framenumbers = np.arange(1420, datareading.get_number_of_available_frames(acquisition_path))
    if acquisition=='120_f1000a800':
        framenumbers = np.arange(4000+10, datareading.get_number_of_available_frames(acquisition_path))
    if acquisition=='120_f5500a500_stable':
        framenumbers = np.arange(4500-10, datareading.get_number_of_available_frames(acquisition_path))
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
blur_t_frame = 2 # blur in time (frame).
blur_x_px = 2 # blur in space (px).
apply_gaussianfiler = True

### Spatial and temporal drift or 0-frequency correction
xcorrect = 'smoothed' #'linear'
tcorrect = None

### 2D FFT parameters
fft_window_visu = 'hann'
zero_pad_factor_visu = (1, 1)
range_dB_visu = 80

# Data display
# ------------

# conversion factor
fr_per_s = datareading.get_acquisition_frequency(acquisition_path)
px_per_mm = 170/80
px_per_um = px_per_mm * 1e3


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)

length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])[2:]
x -= x[0]
t -= t[0]

utility.set_verbose('info')

z_raw = datasaving.fetch_or_generate_data('bos', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)[:,::-1][:,2:]


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


# <codecell>

# Spatial cleaning
from scipy.signal import savgol_filter

# get temporal mean
z_xprofile = np.mean(z_tmp, axis=0)

# linear fit
rivs_fit_spatial_linear = np.polyfit(x, z_xprofile, deg = 1)
utility.log_info(f'Position spatial drift linear estimation: {round(np.poly1d(rivs_fit_spatial_linear)(x).max() - np.poly1d(rivs_fit_spatial_linear)(x).min(), 2)} px')

# parabolic fit
rivs_fit_spatial_parabolic = np.polyfit(x, z_xprofile, deg = 2)

# savgol
savgol_width = len(x)//10 + (1 if len(x)%2 == 0 else 0)

"""
xcorrect can be 4 things
 * None: we do not want correction : it is biaised, or we are interested in the mean-time features
 * 'linear': we do a linear fit and remove everything from it
 * 'parabolic': we do a parabolic fit and remove everything from it
 * 'smoothed': we do a smoothing avec the average and remove that
 * 'total': total removal of the mean value
"""
if xcorrect is None:
    utility.log_info('No spatial correction made')
elif xcorrect == 'linear':
    utility.log_info('Linear spatial correction made')
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_spatial_linear)(x), axis=0)
elif xcorrect == 'parabolic':
    utility.log_info('Parabolic spatial correction made')
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_spatial_parabolic)(x), axis=0)
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
ax.plot(x, np.poly1d(rivs_fit_spatial_linear)(x), color='r', alpha=0.5, label=f'linear fit')
ax.plot(x, np.poly1d(rivs_fit_spatial_parabolic)(x), color='g', alpha=0.5, label=f'linear fit')
ax.plot(x, savgol_filter(z_xprofile, savgol_width, 2), color='b', alpha=0.5, label=f'smooth')
ax.set_ylabel('z [px]')
ax.legend()

ax = axes[1]
ax.plot(x, z_tmp.mean(axis=0), color='k', label='New time-averaged riv position')
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')
ax.legend()


# <codecell>

Z = z_tmp.copy()

k_visu, f_visu = utility.fourier.rdual2d(x, t, zero_pad_factor=zero_pad_factor_visu)
Z_pw = utility.fourier.rpsd2d(Z, x, t, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)


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

ax.set_xlim(0, 1/10)
ax.set_ylim(-1/10, 1/10)


# <codecell>

n_xsamples = 30

x_samples  = np.linspace(x.min(), x.max(), n_xsamples, endpoint=True)
i_x_samples = [np.argmin((x-x_samples[i])**2) for i in range(len(x_samples))]

f = utility.rdual(datareading.get_t_s(acquisition_path, framenumbers=framenumbers))

tab = np.empty((n_xsamples, len(f)), dtype=float)

for i, i_x_sample in enumerate(i_x_samples):
    tab[i] = utility.psd1d(Z[:, i_x_sample], f, window='hann')

print(acquisition)


# <codecell>

fmax = 12.5


# <codecell>

fig, axs = plt.subplots(2, 1, squeeze=False, figsize=utility.figsize('double', ratio=1.2))

vmax = tab.max()
vmin = utility.attenuate_power(vmax, 160)

ax = axs[0, 0]
ax.imshow(tab, extent=utility.correct_extent(f, x), aspect='auto', norm='log', vmax=vmax, vmin=vmin)
ax.set_xlim(0, fmax)

ax = axs[1, 0]
ax.plot(f, tab[-1])
ax.set_yscale('log')
# ax.set_xlim(0, .2)
ax.set_ylim(vmin, vmax*4)
ax.set_xlim(0, fmax)


# <codecell>




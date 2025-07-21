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

### Acquisition selection
# acquisition = '112_nat'
acquisition = '120_f750a500'

utility.set_verbose('subtrace')

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

# utility.set_verbose('info')

z_raw = datasaving.fetch_or_generate_data('bos', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)[:,::-1][:,2:]

print(t.shape)
print(x.shape)
print(z_raw.shape)


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

# Temporal cleaning

# get spatial mean
z_tprofile = np.mean(z_tmp, axis=1)

# linear fit
rivs_fit_temporal_linear = np.polyfit(t, z_tprofile, deg = 1)
utility.log_info(f'Position temporal drift linear estimation: {round(np.poly1d(rivs_fit_temporal_linear)(t).max() - np.poly1d(rivs_fit_temporal_linear)(t).min(), 2)} px')

# correct
if tcorrect == 'linear':
    z_tmp = z_tmp - np.expand_dims(np.poly1d(rivs_fit_temporal_linear)(t), axis=1)
else:
    utility.log_info('No temporal correction made')

### plot

fig, axes = plt.subplots(2, 1, sharex=True, figsize=utility.figsize('double'))
ax = axes[0]
ax.plot(t, z_tprofile, color='k', alpha=0.5, label='old time-averaged riv position')
ax.plot(t, np.poly1d(rivs_fit_temporal_linear)(t), color='r', alpha=0.5, label=f'linear fit')
ax.set_ylabel('z [px]')
ax.legend()

ax = axes[1]
ax.plot(t, np.mean(z_tmp, axis=1), color='k', label='New time-averaged riv position')
ax.set_xlabel('t [s]')
ax.set_ylabel('z [px]')
ax.legend()


# <codecell>

Z = z_tmp.copy()

k_visu, f_visu = utility.fourier.rdual2d(x, t, zero_pad_factor=zero_pad_factor_visu)
Z_pw = utility.fourier.rpsd2d(Z, x, t, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)


# <codecell>

# fig, axes = plt.subplots(1, 2, sharex='col', sharey='col', squeeze=False)
# imshow_kw = {'origin':'upper',
#              'interpolation':'nearest',
#              'aspect':'auto'}
# 
# ax = axes[0,0]
# ax.set_title('Z (normal)')
# imz = ax.imshow(Z, extent=utility.correct_extent(x, t), cmap='viridis', **imshow_kw)
# ax.set_xlabel('$x$ [px]')
# ax.set_ylabel('$t$ [frame]')
# plt.colorbar(imz, ax=ax, label='$z$ [px]')
# 
# ax = axes[0,1]
# vmax, vmin = utility.log_amplitude_range(Z_pw.max(), range_db=range_dB_visu)
# im_zpw = ax.imshow(Z_pw, extent=utility.correct_extent(k_visu, f_visu), norm='log', vmax=vmax, vmin=vmin, cmap='viridis', **imshow_kw)
# ax.set_xlabel(r'$k$ [px$^{-1}$]')
# ax.set_ylabel(r'$f$ [frame$^{-1}$]')
# cb = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [px^2/(px-1.frame-1)]')
# utility.set_ticks_log_cb(cb, vmax, range_db=range_dB_visu)
# 
# ax.set_xlim(0, 1/10*1/20)
# ax.set_ylim(-1/20, 1/5)


# <codecell>

tsample = 4000

zt = Z[tsample]


# <codecell>

fig, axs = plt.subplots(3, 1, squeeze=False, figsize=utility.figsize('double', ratio=1.2), sharex=True)

ax = axs[0,0]
ax.plot(x, zt)

ax = axs[1,0]
ax.plot(x, np.gradient(zt, x, edge_order=2), lw=2)
yy2 = utility.gradient_findiff_regular(zt, x, order=2)
ax.plot(x, yy2, lw=1)
yy4 = utility.gradient_findiff_regular(zt, x, order=4)
ax.plot(x, yy4, lw=1)
yy6 = utility.gradient_findiff_regular(zt, x, order=6)
ax.plot(x, yy6, lw=1)


ax = axs[2,0]
ax.plot(x, np.gradient(np.gradient(zt, x, edge_order=2), x, edge_order=2), lw=2)
yyy2 = utility.laplacian_findiff_regular(zt, x, order=2)
ax.plot(x, yyy2, lw=1)
yyy4 = utility.laplacian_findiff_regular(zt, x, order=4)
ax.plot(x, yyy4, lw=1)
yyy6 = utility.laplacian_findiff_regular(zt, x, order=6)
ax.plot(x, yyy6, lw=1)


# <codecell>

xmin = 800

x_sample = x[x > xmin]

c_byhand = 1.42
tau_indep = int(np.rint(x.max() / c_byhand)+44)
tau_indep = tau_indep //3
print(f'time indep : {tau_indep} frames')

N_smaples = int(t.max()) // tau_indep

print(f'{int(t.max())} frames = {N_smaples}*{tau_indep} + {t.max() % tau_indep}')

tau_indep = int(t.max()) // N_smaples

print(f'Real tau indep : {tau_indep}')

times_xsamples = np.arange(0, int(t.max()), tau_indep)
print(f'Taus {times_xsamples}')

z_samples = np.empty((len(times_xsamples), x_sample.shape[0]), dtype=float)
zder1_samples = np.empty_like(z_samples)
zder2_samples = np.empty_like(z_samples)
for i, tau in enumerate(times_xsamples):
    z_samples[i] = Z[tau][x > xmin]
    zder1_samples[i] = utility.gradient_findiff_regular(z_samples[i], x_sample, order=6)
    zder2_samples[i] = utility.laplacian_findiff_regular(z_samples[i], x_sample, order=6)

print(zder2_samples.shape)


# <codecell>

from scipy.stats import gaussian_kde, norm, iqr


# <codecell>

plt.figure()
ax = plt.gca()

# dset = z_samples.flatten()
# dset = zder1_samples.flatten()
dset = zder2_samples.flatten()

ax.hist(dset, bins=200, density=True)

A = min(np.std(dset), iqr(dset)/1.34)
h = 1.06*A*len(dset)**(-1/5)
h = 0.05
g = gaussian_kde(dset, h)
xa_z = np.linspace(dset.min(), dset.max(), 201, endpoint=True)
ax.plot(xa_z, g.evaluate(xa_z))
# ax.plot(xa_z, norm.pdf(xa_z, dset.mean(), np.std(dset)))


# <codecell>

testx = np.linspace(0, 2*np.pi, 10000, endpoint=False)

testy = np.sin(testx)


# <codecell>

plt.figure()
ax = plt.gca()

xa = np.linspace(-1, 1, 1001, endpoint=True)[1:-1]

ax.plot()
ax.hist(testy.flatten(), bins=200, density=True)

ax.plot(xa, 1/(np.pi*np.sqrt((1+xa)*(1-xa))))
ax.plot(xa, gaussian_kde(testy.flatten()).evaluate(xa))


# <codecell>




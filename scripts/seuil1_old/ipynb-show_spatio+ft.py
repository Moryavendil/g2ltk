# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'figure.titlesize' : 12,
                     'axes.labelsize': 12,'axes.titlesize': 12,
                     'legend.fontsize': 12})

from tools import datareading, rivuletfinding, datasaving, utility


# <codecell>

# Dataset selection
dataset = '20241104'
dataset_path = os.path.join('../', dataset)
print('Available acquisitions:', datareading.find_available_gcv(dataset_path))

# Acquisition selection
acquisition = '50mid_gcv'
acquisition_path = os.path.join(dataset_path, acquisition)

datareading.describe(dataset, acquisition, verbose=3)


# <codecell>

# Parameters definition
rivfinding_params = {
    'resize_factor': 2,
    'borders_min_distance': 8,
    'max_borders_luminosity_difference': 50,
    'max_rivulet_width': 100.,
}
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

# framenumbers = np.arange(100)
roi = 250, None, 1150, None



# <codecell>

# Data fetching
length, height, width = datareading.get_geometry(acquisition_path, framenumbers = framenumbers, subregion=roi)

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
w_raw = utility.w_from_borders(datasaving.fetch_or_generate_data('borders', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params))


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
w_x_treated = w_raw
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
w_xt_treated = w_x_treated
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
blur_t_frame = 3 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 3 # blur in space (px).
sigma_x = blur_x_px

apply_gaussianfiler = True

z_filtered = gaussian_filter(z_xt_treated, sigma=(sigma_t, sigma_x))
w_filtered = gaussian_filter(w_xt_treated, sigma=(sigma_t, sigma_x))


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
imshow_kw = {'aspect': 'auto', 'origin': 'upper'}

ax = axes[0, 0]
ax.set_title('Z (normal)')
imz = ax.imshow(z_xt_treated, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[0, 1]
ax.set_title('W (normal)')
imw = ax.imshow(w_xt_treated, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$w$ [px]')

ax = axes[1, 0]
ax.set_title('Z (smoothed)')
imz = ax.imshow(z_filtered, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1, 1]
ax.set_title('W (smoothed)')
imw = ax.imshow(w_filtered, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$w$ [px]')

plt.tight_layout()


# <codecell>

if apply_gaussianfiler:
    Z = z_filtered.copy()
    W = w_filtered.copy()
else:
    Z = z_xt_treated.copy()
    W = w_xt_treated.copy()


# <codecell>

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
imshow_kw = {'origin':'upper', 'interpolation':'nearest', 'aspect':'auto'}

ax = axes[0]
imz = ax.imshow(Z, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1]
imw = ax.imshow(W, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **imshow_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$w$ [px]')

plt.tight_layout()


# <codecell>

k = utility.rdual(x)
f = utility.dual(t)

Z_hat = utility.ft2d(Z, window='hann')
W_hat = utility.ft2d(W, window='hann')

Z_pw = np.abs(Z_hat)**2
W_pw = np.abs(W_hat)**2

range_db = 100


# <codecell>

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
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

ax = axes[0]
im_zpw = ax.imshow(Z_pw, extent=utility.correct_extent_fft(k, f), norm='log', vmax=Z_pw.max(), vmin = Z_pw.max()/10**(range_db/10), cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cb = plt.colorbar(im_zpw, ax=ax, label='amp²')
cbticks, cbticklabels = get_cbticks(Z_pw, range_db)
cb.ax.set_yticks(cbticks) ; cb.ax.set_yticklabels(cbticklabels)

ax = axes[1]
im_wpw = ax.imshow(W_pw, extent=utility.correct_extent_fft(k, f), norm='log', vmax=W_pw.max(), vmin = W_pw.max()/10**(range_db/10), cmap='viridis', **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cb = plt.colorbar(im_wpw, ax=ax, label='amp²')
cbticks, cbticklabels = get_cbticks(W_pw, range_db)
cb.ax.set_yticks(cbticks) ; cb.ax.set_yticklabels(cbticklabels)

ax.set_xlim(0, 1/20)
ax.set_ylim(-.1, .1)

# plt.tight_layout()


# <codecell>

### GO REAL UNITS
px_per_mm = 33.6
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

tt = datareading.get_t_s(acquisition_path, framenumbers)
xx = datareading.get_x_mm(acquisition_path, framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'], px_per_mm=px_per_mm)


# <codecell>




# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook
# %matplotlib qt

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


from tools import datareading, rivuletfinding, datasaving, utility, log_info


# <markdowncell>

# # MEASURE DES VARIABLES
# Automated measurements of lots of things


# <codecell>

# Dataset selection
dataset = '20241104'
dataset_path = os.path.join('../', dataset)
print('Available acquisitions:', datareading.find_available_gcv(dataset_path))

# Acquisition selection
acquisition = 'rest_gcv'
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

t = datareading.get_t_frames(acquisition_path, framenumbers=framenumbers).astype(float)
x = datareading.get_x_px(acquisition_path, framenumbers = framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'])

z_raw = datasaving.fetch_or_generate_data('bol', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params)
w_raw = utility.w_from_borders(datasaving.fetch_or_generate_data('borders', dataset, acquisition, framenumbers=framenumbers, roi=roi, **rivfinding_params))


# <markdowncell>

# ## data cleaning
# We do various cleaning steps to obtaon a cleaner $Z$ and $W$


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
plt.show()
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
blur_t_frame = 2 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 3 # blur in space (px).
sigma_x = blur_x_px

apply_gaussianfiler = True

z_filtered = gaussian_filter(z_xt_treated, sigma=(sigma_t, sigma_x))
w_filtered = gaussian_filter(w_xt_treated, sigma=(sigma_t, sigma_x))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
realplot_kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'auto'}

ax = axes[0, 0]
ax.set_title('Z (normal)')
imz = ax.imshow(z_xt_treated, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[0, 1]
ax.set_title('W (normal)')
imw = ax.imshow(w_xt_treated, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$w$ [px]')

ax = axes[1, 0]
ax.set_title('Z (smoothed)')
imz = ax.imshow(z_filtered, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1, 1]
ax.set_title('W (smoothed)')
imw = ax.imshow(w_filtered, extent=utility.correct_extent_spatio(x, t), cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$w$ [px]')

# plt.tight_layout()


# <codecell>

if apply_gaussianfiler:
    Z = z_filtered.copy()
    W = w_filtered.copy()
else:
    Z = z_xt_treated.copy()
    W = w_xt_treated.copy()


# <codecell>

log_info(f'Mean z: {Z.mean()} px')
log_info(f'Mean w: {W.mean()} px')


# <codecell>

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
realplot_kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'auto', 'extent': utility.correct_extent_spatio(x, t)}

ax = axes[0]
imz = ax.imshow(Z, cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imz, ax=ax, label='$z$ [px]')

ax = axes[1]
imw = ax.imshow(W, cmap='viridis', **realplot_kw)
ax.set_xlabel('$x$ [px]')
ax.set_ylabel('$t$ [frame]')
plt.colorbar(imw, ax=ax, label='$w$ [px]')

# plt.tight_layout()


# <codecell>

### GO REAL UNITS
px_per_mm = 33.6
acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")

# ms
acquisition_frequency /= 1000

unit_t = 'ms'
unit_x = 'mm'

t /= acquisition_frequency
x /= px_per_mm
Z /= px_per_mm
W /= px_per_mm

zscalemax = np.percentile(Z.flatten(), 99.)
zmin, zmax = -zscalemax, zscalemax
wmidscale = W.mean()
wmin, wmax = np.percentile(W.flatten(), 1.), np.percentile(W.flatten(), 99.)



# <codecell>

import matplotlib.colors as mcolors

cmap_ref = plt.cm.bwr

w_prop_above_mean = (W > wmidscale).sum() / (W.shape[0]*W.shape[1])
colors_above = cmap_ref(np.linspace(0.5, 1, int(256 * w_prop_above_mean), endpoint=True))
colors_below = cmap_ref(np.linspace(0, 0.5, 256 - int(256 * w_prop_above_mean), endpoint=False))
colors = np.vstack((colors_below, colors_above))
cmap_w = mcolors.LinearSegmentedColormap.from_list('customseismic', colors)


# <codecell>


# plt.rcParams['text.usetex'] = True


# <codecell>

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
realplot_kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'auto', 'extent': utility.correct_extent_spatio(x, t)}

fig.suptitle(f'{acquisition} ({dataset})')

ax = axes[0]
ax.set_title(rf'$z(x, t)$')
imz = ax.imshow(Z, cmap=cmap_ref, vmin=zmin, vmax=zmax, **realplot_kw)
ax.set_xlabel(fr'$x$ [{unit_x}]')
ax.set_ylabel(fr'$t$ [{unit_t}]')
plt.colorbar(imz, ax=ax, label=fr'$z$ [{unit_x}]')

ax = axes[1]
ax.set_title(rf'$w(x, t)$')
imw = ax.imshow(W, cmap=cmap_w, vmin=wmin, vmax=wmax, **realplot_kw)
ax.set_xlabel(fr'$x$ [{unit_x}]')
ax.set_ylabel(fr'$t$ [{unit_t}]')
plt.colorbar(imw, ax=ax, label=fr'$w$ [{unit_x}]')

plt.tight_layout()
utility.save_graphe(f'{acquisition}_spatio', imageonly=True)


# <codecell>




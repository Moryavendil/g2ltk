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
        
        n_period_t = 2.5
        n_period_x = 2.5
        
        t1 = int(61 + .25 * L_T*fr_per_ms)
        t2 = int(t1 + n_period_t * L_T*fr_per_ms)
        x1 = int(870 + L_X / 2  - .25 * L_X*px_per_mm*2 )
        x2 = int(x1 + n_period_x * L_X*px_per_mm*2)
        
Z_ = Z_mm[t1:t2, x1:x2].T
W_ = W_mm[t1:t2, x1:x2].T
Z_ -= Z_.mean()
t_ = t_ms[t1:t2]
x_ = x_mm[x1:x2]
t_ -= t_[0]
x_ -= x_[0]
T_, X_ = np.meshgrid(t_, x_)



# <codecell>


u0 = 300 / 1e3 # mm / ms
Fa = 14.45304192 / px_per_mm
Za = 15.83437829 / px_per_mm
Wa = 5.87467402 / px_per_mm


# <codecell>


Zt = np.mean(Z_, axis = 1)
Zx = np.mean(Z_, axis = 0)
Wt = np.mean(W_, axis = 1)
Wx = np.mean(W_, axis = 0)

Zextr = .6 #.5 # Z_.max()
Wextr = .4

Zlim = [-Zextr, Zextr]
# Zlim = [-.5, .5]
zticks = [-.5, 0, .5]
zticklabels=['-0.5', '0', '0.5']

wticks = [0, .2, .4]
wticklabels=['0', '0.2', '0.4']

tticks_major = L_T * np.arange(0, t_[-1] / L_T+5)
tticks_minor = L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+5)

xticks_major = L_X * np.arange(0, x_[-1] / L_X+5)
xticks_minor = L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+5)


cmap_z = 'PuOr_r'
cmap_w = 'viridis_r'


# <codecell>

from scipy.ndimage import map_coordinates

crossline_start_index = [int((1 + 1/np.sqrt(2))/2*len(t_)), int((1 - 1/np.sqrt(2))/2*len(x_))]
crossline_end_index = [int((1 - 1/np.sqrt(2))/2*len(t_)), int((1 + 1/np.sqrt(2))/2*len(x_))]

t1_crossline_index, x1_crossline_index = crossline_start_index
t2_crossline_index, x2_crossline_index = crossline_end_index
dlength = int(np.hypot(t2_crossline_index - t1_crossline_index, x2_crossline_index - x1_crossline_index)) + 1
t_crossline, x_crossline = np.linspace(t1_crossline_index, t2_crossline_index, dlength), np.linspace(x1_crossline_index, x2_crossline_index, dlength)


d_crossline = np.hypot((np.interp(t_crossline, np.arange(len(t_)), t_) - np.interp(t1_crossline_index, np.arange(len(t_)), t_)) * u0,
                       (np.interp(x_crossline, np.arange(len(x_)), x_) - np.interp(x1_crossline_index, np.arange(len(x_)), x_)) * 1.)
# d_crossline = np.hypot(t_crossline - t1_crossline_index, x_crossline - x1_crossline_index)
w_crossline = map_coordinates(W_, np.vstack((x_crossline, t_crossline))).astype(float)
z_crossline = map_coordinates(Z_, np.vstack((x_crossline, t_crossline))).astype(float)



L_d = np.hypot(L_T*u0, L_X)/2
dticks_major = L_d * np.arange(0, d_crossline[-1] / L_d+1)
dticks_minor = L_d/4 + L_d/4 * np.arange(0, 4 * d_crossline[-1] / L_d+1)


# <codecell>

logging.set_verbose('debug')

utility.deactivate_saveplot(style='jfm')


# <codecell>

lw_levellines = 1.25


# <codecell>

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(constrained_layout=False, figsize=utility.figsize('wide', ratio=1.))

gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1],
                      hspace=0.075, wspace=0.075,
                      left=0.15, right=.98, top=.98, bottom=0.15)

### TOP PLOT
ax = fig.add_subplot(gs[0, 0])

# Plot
ax.plot(t_, Zx, '-k', lw=1)

# Colormap
polygon = ax.fill_between(t_, Zx, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(Zx.min(), Zx.max(), 100)[::-1].reshape(1, -1).T,
                      vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

# ampli
ax.axhline( Fa, ls='--', color=utility.color_z, zorder=-5, alpha=.6)
ax.axhline(-Fa, ls='--', color=utility.color_z, zorder=-5, alpha=.6)

# Limits
ax.set_xticks(tticks_major)
ax.set_xticks(tticks_minor, minor=True)
ax.set_xticklabels([])
ax.set_xlim(t_[0], t_[0] + L_T * (n_period_t+0.001))

ax.set_yticks(zticks)
ax.set_yticklabels(zticklabels)
ax.set_ylim(Zlim)
ax.set_ylabel(r'$\left\langle z \right\rangle_x(t)$ [mm]')

### RIGHT PLOT
ax = fig.add_subplot(gs[1, 1])

# Plot
ax.plot(Zt, x_, '-k', lw=1)

# Colormap
polygon = ax.fill_betweenx(x_, Zt, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(Zt.min(), Zt.max(), 100).reshape(1, -1),
                      vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

# ampli
ax.axvline( Za, ls='--', color=utility.color_z, zorder=-5, alpha=.6)
ax.axvline(-Za, ls='--', color=utility.color_z, zorder=-5, alpha=.6)

# Limits
ax.set_yticks(xticks_major)
ax.set_yticks(xticks_minor, minor=True)
ax.set_yticklabels([])
ax.set_ylim(x_[0] + L_X * (n_period_x+0.001), x_[0])

ax.set_xticks(zticks)
ax.set_xticklabels(zticklabels)
ax.set_xlim(Zlim)
ax.set_xlabel(r'$\left\langle z \right\rangle_t(x)$ [mm]')

### CENTER PLOT
ax = fig.add_subplot(gs[1, 0], frameon=True)

# main image
imr = ax.imshow(Z_, extent=utility.correct_extent(t_, x_), vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto', interpolation='bicubic')

# Level lines

# levels= [10]
# levels = [10, 33]
levels = [20]
# levels= [33]

levels_up = [np.percentile(W_.flatten(), level) for level in (100 - np.sort(levels)[::-1])]
levels_down = [np.percentile(W_.flatten(), level) for level in np.sort(levels)]
# Fill
ax.contourf(t_, x_, W_, levels = [levels_up[0]] + [np.inf],
            colors = 'k', alpha=.2)
ax.contourf(t_, x_, W_, levels = [-np.inf] + [levels_down[-1]],
            colors = 'w', linestyles='--', alpha=.3)
# Borders
ax.contour(t_, x_, W_, levels = levels_up,
           colors = 'k', linewidths=lw_levellines)
ax.contour(t_, x_, W_, levels = levels_down,
           colors = 'k', linestyles='--', linewidths=lw_levellines)

# Limits
ax.set_yticks(xticks_major)
ax.set_yticks(xticks_minor, minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(x_[0] + L_X * (n_period_x+0.001), x_[0])
ax.set_ylabel(r'Position $x$ [mm]')

ax.set_xticks(tticks_major)
ax.set_xticks(tticks_minor, minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(t_[0], t_[0] + L_T * (n_period_t+0.001))
ax.set_xlabel(r'Time $t$ [ms]')

# utility.force_aspect_ratio(ax)

### TOP - RIGHT PLOT
# colorbar
ax = fig.add_subplot(gs[0, 1], frameon=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes('left', size='20%', pad=0.05)
fig.colorbar(imr, cax=cax, orientation='vertical')
cax.set_ylabel(r'$z$ [mm]')
cax.set_yticks(zticks)
cax.set_yticklabels(zticklabels)
ax.set_axis_off()

utility.save_graphe('phaseplot_z')


# <codecell>

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(constrained_layout=False, figsize=utility.figsize('wide', ratio=1.))

gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1],
                      hspace=0.075, wspace=0.075,
                      left=0.15, right=.98, top=.98, bottom=0.15)

### TOP PLOT
ax = fig.add_subplot(gs[0, 0])

# Plot

# Colormap
polygon = ax.fill_between(d_crossline, w_crossline, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(w_crossline.min(), w_crossline.max(), 100)[::-1].reshape(1, -1).T,
                      vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

for lvl in levels_down:
    ax.axhline(lvl, ls='--', color='k', lw=lw_levellines)
for lvl in levels_up:
    ax.axhline(lvl, ls='-', color='k', lw=lw_levellines)


# Colormap
polygon = ax.fill_between(d_crossline, w_crossline, 2., lw=0, color='w', zorder=4)

# ampli
ax.axhline(W_.mean() + Wa, ls='--', color=utility.color_w, zorder=5, alpha=.6)
ax.axhline(W_.mean() - Wa, ls='--', color=utility.color_w, zorder=5, alpha=.6)

ax.plot(d_crossline, w_crossline, '-k', zorder=8, lw=1)

# Limits
ax.set_xticks(dticks_major)
ax.set_xticks(dticks_minor, minor=True)
ax.set_xlim(d_crossline[0], d_crossline[-1])
ax.set_xlabel(r'$x + v_w\,t$ [mm]')

ax.set_yticks(wticks)
ax.set_yticklabels(wticklabels)
ax.set_ylim(0, Wextr)
ax.set_ylabel(r'$w(x + v_w\,t)$ [mm]')

### RIGHT PLOT
ax = fig.add_subplot(gs[1, 1])

# Colormap
polygon = ax.fill_betweenx(d_crossline, w_crossline, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(w_crossline.max(), w_crossline.min(), 100)[::-1].reshape(1, -1),
                      vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)


for lvl in levels_down:
    ax.axvline(lvl, ls='--', color='k', lw=lw_levellines)
for lvl in levels_up:
    ax.axvline(lvl, ls='-', color='k', lw=lw_levellines)
polygon = ax.fill_betweenx(d_crossline, w_crossline, 2, lw=0, color='w', zorder=4)

# ampli
ax.axvline(W_.mean() + Wa, ls='--', color=utility.color_w, zorder=5, alpha=.6)
ax.axvline(W_.mean() - Wa, ls='--', color=utility.color_w, zorder=5, alpha=.6)

ax.plot(w_crossline, d_crossline, '-k', lw=1, zorder=8)

# Limits
ax.set_yticks(dticks_major)
ax.set_yticks(dticks_minor, minor=True)
ax.set_ylim(d_crossline[0], d_crossline[-1])
ax.set_ylabel(r'$x - v_w\,t$ [mm]') # Advected position 

ax.set_xticks(wticks)
ax.set_xticklabels(wticklabels)
ax.set_xlim(0, Wextr)
ax.set_xlabel(r'$w(x + v_w\,t)$ [mm]')

### CENTER PLOT
ax = fig.add_subplot(gs[1, 0], frameon=True)

# main image
imr = ax.imshow(W_, extent=utility.correct_extent(t_, x_), vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto', interpolation='bicubic')

# sample line
ax.plot([t_[t1_crossline_index], t_[t2_crossline_index]], [x_[x1_crossline_index], x_[x2_crossline_index]], color='k', marker='o', mfc='k', ms=4)
# ax.scatter([t_[t1_crossline_index], t_[t2_crossline_index]], [x_[x1_crossline_index], x_[x2_crossline_index]], color='k',s=12, zorder=4)

# Limits
ax.set_yticks(xticks_major)
ax.set_yticks(xticks_minor, minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(x_[0] + L_X * (n_period_x+0.001), x_[0])
ax.set_ylabel(r'Position $x$ [mm]')

ax.set_xticks(tticks_major)
ax.set_xticks(tticks_minor, minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(t_[0], t_[0] + L_T * (n_period_t+0.001))
ax.set_xlabel(r'Time $t$ [ms]')

# utility.force_aspect_ratio(ax)

### TOP - RIGHT PLOT
# colorbar
ax = fig.add_subplot(gs[0, 1], frameon=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes('left', size='20%', pad=0.05)
fig.colorbar(imr, cax=cax, orientation='vertical')
cax.set_ylabel(r'$w$ [mm]')
cax.set_yticks(wticks)
cax.set_yticklabels(wticklabels)
ax.set_axis_off()

utility.save_graphe('phaseplot_w')


# <codecell>

# plt.close()


# <codecell>




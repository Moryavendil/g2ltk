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
        
        t1 = int(61 + .26 * L_T*fr_per_ms)
        t2 = int(t1 + 2.5 * L_T*fr_per_ms)
        x1 = int(870 + L_X / 2  - .24 * L_X*px_per_mm*2 )
        x2 = int(x1 + 2.5 * L_X*px_per_mm*2)
        
Z_ = Z_mm[t1:t2, x1:x2]
W_ = W_mm[t1:t2, x1:x2]
t_ = t_ms[t1:t2]
x_ = x_mm[x1:x2]
t_ -= t_[0]
x_ -= x_[0]
X_, T_ = np.meshgrid(x_, t_)

Zt = np.mean(Z_, axis = 0)
Zx = np.mean(Z_, axis = 1)

ZT = np.tile(Zt, (len(t_), 1))
ZX = np.tile(Zx, (len(x_), 1)).T



# <codecell>


Wt = np.mean(W_, axis = 0)
Wx = np.mean(W_, axis = 1)

Zextr = .6 #.5 # Z_.max()
Wextr = .4

cmap_z = 'PuOr_r'
cmap_w = 'viridis'


from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(constrained_layout=False, figsize=utility.figsize('wide', ratio=1))

gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1],
                      hspace=0.075, wspace=0.075,
                      left=0.15, right=0.95, top=0.95, bottom=0.1)

### TOP PLOT
ax = fig.add_subplot(gs[0, 0])

# Plot
ax.plot(x_, Zt, '-k', lw=1)

# Colormap
polygon = ax.fill_between(x_, Zt, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(Zt.min(), Zt.max(), 100)[::-1].reshape(1, -1).T,
                      vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

# Limits
ax.set_xlim(x_[0], x_[-1])
ax.set_xticklabels([])
ax.set_ylim(-Zextr, Zextr)
ax.set_ylabel(r'$\left\langle \zeta \right\rangle_t$ [px]')

### RIGHT PLOT
ax = fig.add_subplot(gs[1, 1])

# Plot
ax.plot(Zx, t_, '-k', lw=1)

# Colormap
polygon = ax.fill_betweenx(t_, Zx, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(Zx.min(), Zx.max(), 100).reshape(1, -1),
                      vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

# Limits
ax.set_ylim(t_[-1], t_[0])
ax.set_yticklabels([])
ax.set_xlim(-Zextr, Zextr)
ax.set_xlabel(r'$\left\langle \zeta \right\rangle_x$ [px]')

### CENTER PLOT
ax = fig.add_subplot(gs[1, 0], frameon=True)

# main image
imr = ax.imshow(Z_, extent=utility.correct_extent(x_, t_), vmin=-Zextr, vmax=Zextr, cmap=cmap_z, aspect='auto', interpolation='bicubic')

# Level lines

# levels= [10]
# levels = [10, 33]
levels = [20]
# levels= [33]

levels_up_w = [np.percentile(W_.flatten(), level) for level in (100 - np.sort(levels)[::-1])]
levels_down_w = [np.percentile(W_.flatten(), level) for level in np.sort(levels)]
# Borders
ax.contour(x_, t_, W_, levels = levels_up_w,
           colors = 'k')
ax.contour(x_, t_, W_, levels = levels_down_w,
           colors = 'k', linestyles='--')
# Fill
ax.contourf(x_, t_, W_, levels = [levels_up_w[0]] + [np.inf],
           colors = 'k', alpha=.2)
ax.contourf(x_, t_, W_, levels = [-np.inf] + [levels_down_w[-1]],
           colors = 'w', linestyles='--', alpha=.3)

# Limits
ax.set_xticks(L_X/2 * np.arange(0, 2*x_[-1] / L_X+1))
ax.set_xticks(L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(x_[0], x_[-1])
ax.set_xlabel(r'Position $x$ [mm]')

ax.set_yticks(L_T/2 * np.arange(0, 2*t_[-1] / L_T+1))
ax.set_yticks(L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(t_[-1], t_[0])
ax.set_ylabel(r'Time $t$ [ms]')

### TOP - RIGHT PLOT
# colorbar
ax = fig.add_subplot(gs[0, 1], frameon=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes('left', size='20%', pad=0.05)
fig.colorbar(imr, cax=cax, orientation='vertical')
cax.set_ylabel(r'$\zeta$ [px]')
ax.set_axis_off()

# utility.save_graphe('phaseplot_z')


# <codecell>

from scipy.ndimage import map_coordinates

crossline_start_index = [int((1 + 1/np.sqrt(2))/2*len(t_)), int((1 - 1/np.sqrt(2))/2*len(x_))]
crossline_end_index = [int((1 - 1/np.sqrt(2))/2*len(t_)), int((1 + 1/np.sqrt(2))/2*len(x_))]

t1_crossline_index, x1_crossline_index = crossline_start_index
t2_crossline_index, x2_crossline_index = crossline_end_index
dlength = int(np.hypot(t2_crossline_index - t1_crossline_index, x2_crossline_index - x1_crossline_index)) + 1
t_crossline, x_crossline = np.linspace(t1_crossline_index, t2_crossline_index, dlength), np.linspace(x1_crossline_index, x2_crossline_index, dlength)

u0 = 300 / 1e3 # mm / ms

d_crossline = np.hypot((np.interp(t_crossline, np.arange(len(t_)), t_) - np.interp(t1_crossline_index, np.arange(len(t_)), t_)) * u0,
                       (np.interp(x_crossline, np.arange(len(x_)), x_) - np.interp(x1_crossline_index, np.arange(len(x_)), x_)) * 1.)
# d_crossline = np.hypot(t_crossline - t1_crossline_index, x_crossline - x1_crossline_index)
w_crossline = map_coordinates(W_, np.vstack((t_crossline, x_crossline))).astype(float)


# <codecell>

fig = plt.figure(constrained_layout=False, figsize=utility.figsize('wide', ratio=1/1.2))

gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], width_ratios=[3],
                      hspace=.25, wspace=0.075,
                      left=0.15, right=0.85, top=0.95, bottom=0.1)

### TOP PLOT
ax = fig.add_subplot(gs[0, 0])

# main image
imr = ax.imshow(W_, extent=utility.correct_extent(x_, t_), vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto', interpolation='bicubic')
ax.plot([x_[x1_crossline_index], x_[x2_crossline_index]], [t_[t1_crossline_index], t_[t2_crossline_index]], color='k')
ax.scatter(x_[x1_crossline_index], t_[t1_crossline_index], color='k', s=12, zorder=4)
ax.scatter(x_[x2_crossline_index], t_[t2_crossline_index], color='k', s=12, zorder=4)

# imr = ax.imshow(W_, vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto', interpolation='bicubic')
# ax.scatter(crossline_start_index[1], crossline_start_index[0], color='r', s=10)
# ax.scatter(crossline_end_index[1], crossline_end_index[0], color='r', s=10)
# ax.plot([crossline_start_index[1], crossline_end_index[1]], [crossline_start_index[0], crossline_end_index[0]], color='k')

# Limits
ax.set_xticks(L_X/2 * np.arange(0, 2*x_[-1] / L_X+1))
ax.set_xticks(L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(x_[0], x_[-1])
ax.set_xlabel(r'Position $x$ [mm]')

ax.set_yticks(L_T/2 * np.arange(0, 2*t_[-1] / L_T+1))
ax.set_yticks(L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(t_[-1], t_[0])
ax.set_ylabel(r'Time $t$ [ms]')

def force_aspect_ratio(ax, aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

force_aspect_ratio(ax)

ax = fig.add_subplot(gs[1, 0])

# Plot
ax.plot(d_crossline, w_crossline, '-k', lw=1)

for lvl in levels_down_w:
    ax.axhline(lvl, ls='--', color='k')
for lvl in levels_up_w:
    ax.axhline(lvl, ls='-', color='k')

# Colormap
polygon = ax.fill_between(d_crossline, w_crossline, lw=0, color='none')
verts = np.vstack([p.vertices for p in polygon.get_paths()])
gradient = plt.imshow(np.linspace(w_crossline.min(), w_crossline.max(), 100)[::-1].reshape(1, -1).T,
                      vmin=0, vmax=Wextr, cmap=cmap_w, aspect='auto',
                      extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

# Limits
L_d = np.hypot(L_T*u0, L_X)/2
ax.set_xticks(L_d * np.arange(0, d_crossline[-1] / L_d+1))
ax.set_xticks(L_d/4 + L_d/4 * np.arange(0, 4 * d_crossline[-1] / L_d+1), minor=True)
ax.set_xlim(d_crossline[0], d_crossline[-1])
ax.set_xlabel(r'Counter-advected position $x + u_0\,t$ [mm]')

ax.set_ylim(0, Wextr)
ax.set_ylabel(r'$W$ [px]')

# utility.save_graphe('phaseplot_w')


# <codecell>




# <codecell>

from scipy.signal import hilbert2

def phase_wr_hilbert(sig):
    sig -= sig.mean()

    # technique 4 : hilbert2 on symmetrized matrix
    hh, ww = sig.shape
    sig_x4 = np.zeros((2 * hh, 2 * ww), dtype=sig.dtype)
    sig_x4[:hh, :ww] = sig[:, :]
    sig_x4[hh:, :ww] = sig[::-1, :]
    sig_x4[:hh, ww:] = sig[:, ::-1]
    sig_x4[hh:, ww:] = sig[::-1, ::-1]
    hil2D_x4 = hilbert2(sig_x4)
    phase_wrapped_hil2D_x4 = np.angle(hil2D_x4)
    
    # SYMMETRIZED VERSION
    # phase_wrapped_hil2D = phase_wrapped_hil2D4[:hh, :ww] # no flip (?)
    phase_wrapped_hil2D = phase_wrapped_hil2D_x4[hh:, :ww][::-1, :] # flip (?)  # PARFOIS SIL FAUT INVERSER LE BAZARD ???
    
    return phase_wrapped_hil2D


# <codecell>

import matplotlib.colors as col
import hsluv

def make_segmented_cmap():
    white = '#ffffff'
    black = '#000000'
    gray = '#808080'
    red = '#ff0000'
    blue = '#0000ff'
    green = '#00ff00'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [gray, red, green, blue, gray], N=256, gamma=1)
    return anglemap

anglecmap = make_segmented_cmap()
anglecmap = 'twilight_shifted'


# <codecell>

utility.activate_saveplot(True)


# <codecell>

fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True, sharey=True, figsize=utility.figsize('double', ratio=1))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto', 
             'extent': utility.correct_extent(x_, t_)}

phase_Z0 = -phase_wr_hilbert(ZX.copy())
phase_Z1 = -phase_wr_hilbert(ZT.copy())
phase_W1 = -phase_wr_hilbert(W_.copy())

def contourupdown(ax, mat, levels_up, levels_down):
    # Borders
    ax.contour(x_, t_, mat, levels = levels_up,
               colors = 'k')
    ax.contour(x_, t_, mat, levels = levels_down,
               colors = 'k', linestyles='--')
    # Fill
    ax.contourf(x_, t_, mat, levels = [levels_up[0]] + [np.inf],
                colors = 'k', alpha=.2)
    ax.contourf(x_, t_, mat, levels = [-np.inf] + [levels_down[-1]],
                colors = 'w', linestyles='--', alpha=.3)


ax = axes[0,0]
ax.set_title(r'$\left\langle z\right\rangle_x \approx Z_0 e^{i\omega t} + c.c.$')
im = ax.imshow(ZX, vmin=-Zextr, vmax=Zextr, **imshow_kw, cmap=cmap_z)
# contourupdown(ax, ZX, [.25], [-.25])
cbwa = plt.colorbar(im, ax=ax)
force_aspect_ratio(ax)

ax = axes[0,1]
ax.set_title(r'$\arg(\left\langle z\right\rangle_x) \approx \arg(Z_0) + \omega t$ (Hilbert)')
im = ax.imshow(phase_Z0, cmap=anglecmap, **imshow_kw)
# contourupdown(ax, ZX, [.25], [-.25])
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[1,0]
ax.set_title(r'$\left\langle z\right\rangle_t \approx Z_1 e^{-ikx} + c.c.$')
im = ax.imshow(ZT, vmin=-Zextr, vmax=Zextr, **imshow_kw, cmap=cmap_z)
# contourupdown(ax, ZT, [.25], [-.25])
cbwa = plt.colorbar(im, ax=ax)
force_aspect_ratio(ax)

ax = axes[1,1]
ax.set_title(r'$\arg(\left\langle z\right\rangle_t) \approx \arg(Z_1) - k x$ (Hilbert)')
im = ax.imshow(phase_Z1, cmap=anglecmap, **imshow_kw)
# contourupdown(ax, ZT, [.25], [-.25])
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[2,0]
ax.set_title(r'$w \approx W_1 e^{i(\omega t-kx)} + c.c.$')
im = ax.imshow(W_, vmin=0, vmax=Wextr, cmap=cmap_w, **imshow_kw)
# contourupdown(ax, W_, [.1], [-.09])
cbwa = plt.colorbar(im, ax=ax)
force_aspect_ratio(ax)

ax = axes[2,1]
ax.set_title(r'$\arg(w) \approx \arg(W_1) + \omega t - k x$ (Hilbert)')
im = ax.imshow(phase_W1, cmap=anglecmap, **imshow_kw)
# contourupdown(ax, W_, [.1], [-.09])
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

# Limits
ax.set_xticks(L_X/2 * np.arange(0, 2*x_[-1] / L_X+1))
ax.set_xticks(L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(x_[0], x_[-1])

ax.set_yticks(L_T/2 * np.arange(0, 2*t_[-1] / L_T+1))
ax.set_yticks(L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(t_[-1], t_[0])


axes[0,0].set_ylabel(r'Time $t$ [ms]')
axes[1,0].set_ylabel(r'Time $t$ [ms]')
axes[2,0].set_ylabel(r'Time $t$ [ms]')

axes[-1,0].set_xlabel(r'Position $x$ [mm]')
axes[-1,1].set_xlabel(r'Position $x$ [mm]')
utility.save_graphe('phase comput')


# <codecell>

fig, axes = plt.subplots(2, 2, squeeze=False, figsize=utility.figsize('wide', ratio=1.2), sharex=True, sharey=True)
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'vmin':-np.pi, 'vmax':np.pi,
             'cmap':anglecmap,
             'extent': utility.correct_extent(x_, t_)}

def wrap(phase):
    return (phase + np.pi) % (2 * np.pi) - np.pi

phase_sum = wrap(phase_Z0 + phase_Z1 )

phase_sum_shift = wrap(phase_sum - np.pi/2)

delta = wrap(phase_W1 - phase_sum_shift)

ax = axes[0,0]
# ax.set_title(r'$\arg(\left\langle z\right\rangle_x) \approx \arg(Z_0) + \omega t$ (Hilbert)')
# ax.set_title(r'$\arg(w) \approx \arg(W_1) + \omega t - k x$ (Hilbert)')
ax.set_title(r'$\arg(\left\langle z\right\rangle_x) + \arg(\left\langle z\right\rangle_t) $'+'\n'+
             r'$\approx \arg(Z_0) + \arg(Z_1) + \omega t - kx$')
im = ax.imshow(phase_sum, **imshow_kw)
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[0,1]
ax.set_title(r'$\arg(w) $'+'\n'+
             r'$\approx \arg(W_1) + \omega t - k x$')
im = ax.imshow(phase_W1, **imshow_kw)
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[1,0]
ax.set_title(r'$\arg(\left\langle z\right\rangle_x) + \arg(\left\langle z\right\rangle_t) - \pi/2 $'+'\n'+
             r'$\approx \arg(Z_0) + \arg(Z_1) + \omega t - kx - \pi/2$')
im = ax.imshow(phase_sum_shift, **imshow_kw)
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[1,1]
ax.set_title(r'$\arg(w) - \left( \arg(\left\langle z\right\rangle_x) + \arg(\left\langle z\right\rangle_t) - \pi/2 \right) $'+'\n'+
             r'$\approx \arg(W_1) - \left( \arg(Z_0) + \arg(Z_1) - \pi/2\right)$')
im = ax.imshow(delta, **imshow_kw)
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

# Limits
ax.set_xticks(L_X/2 * np.arange(0, 2*x_[-1] / L_X+1))
ax.set_xticks(L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(x_[0], x_[-1])

ax.set_yticks(L_T/2 * np.arange(0, 2*t_[-1] / L_T+1))
ax.set_yticks(L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(t_[-1], t_[0])

axes[0,0].set_ylabel(r'Time $t$ [ms]')
axes[1,0].set_ylabel(r'Time $t$ [ms]')
axes[-1,0].set_xlabel(r'Position $x$ [mm]')
axes[-1,1].set_xlabel(r'Position $x$ [mm]')

plt.tight_layout()
utility.save_graphe('phase comparison')


# <codecell>

fig, axes = plt.subplots(2, 1, squeeze=False, sharex=True, sharey=True, figsize=utility.figsize('wide', ratio=1/1.5))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'vmin':-np.pi, 'vmax':np.pi,
             'cmap':'seismic',
             'extent': utility.correct_extent(x_, t_)}

ax = axes[0,0]
ax.set_title(r'$\arg(w) - \left( \arg(\left\langle z\right\rangle_x) + \arg(\left\langle z\right\rangle_t) - \pi/2 \right) $'+'\n'+
             r'$\approx \arg(W_1) - \left( \arg(Z_0) + \arg(Z_1) - \pi/2\right)$')
im = ax.imshow(delta, **imshow_kw)
cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

ax = axes[1,0]
ax.set_title(r'$\arg(w) - \left( \arg(\left\langle z\right\rangle_x) + \arg(\left\langle z\right\rangle_t) - \pi/2 \right) $'+'\n'+
             r'$\approx \arg(W_1) - \left( \arg(Z_0) + \arg(Z_1) - \pi/2\right)$')
im = ax.imshow(delta, **imshow_kw)

contourupdown(ax, W_, levels_up_w, levels_down_w)

cbwa = plt.colorbar(im, ax=ax)
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
cbwa.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
force_aspect_ratio(ax)

axes[0,0].set_ylabel(r'Time $t$ [ms]')
axes[1,0].set_ylabel(r'Time $t$ [ms]')
axes[-1,0].set_xlabel(r'Position $x$ [mm]')

# Limits
ax.set_xticks(L_X/2 * np.arange(0, 2*x_[-1] / L_X+1))
ax.set_xticks(L_X/4 + L_X/4 * np.arange(0, 4 * x_[-1] / L_X+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_xlim(x_[0], x_[-1])

ax.set_yticks(L_T/2 * np.arange(0, 2*t_[-1] / L_T+1))
ax.set_yticks(L_T/4 + L_T/4 * np.arange(0, 4 * t_[-1] / L_T+1), minor=True)
# ax.set_xticklabels([('0' if i==0 else fr'${i}\,\lambda$') for i in np.arange(0, x_[-1] / L_X+1).astype(int)])
ax.set_ylim(t_[-1], t_[0])

plt.tight_layout()
utility.save_graphe('residual')


# <codecell>

plt.close()


# <codecell>




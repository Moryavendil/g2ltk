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

### Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

### Acquisition selection
acquisition = 'a260'
acquisition_path = os.path.join(dataset_path, acquisition)


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

len(Z_mm)


# <codecell>

# roll = (0, 40)
# 
# Z_mm = np.roll(Z_mm, roll)
# W_mm = np.roll(W_mm, roll)
# print('rolled')


# <codecell>

Z_mm = Z_mm[85:485, 680:3430]
W_mm = W_mm[85:485, 680:3430]
Z_mm = Z_mm[:, ::-1]
W_mm = W_mm[:, ::-1]


# <codecell>

plt.figure()
plt.imshow(Z_mm, aspect='auto')


# <codecell>

zero_pad_factor_visu = (10, 10)
fft_window_visu = 'boxcar'
range_dB_visu = 40

k_visu_mm, f_visu_mm = utility.fourier.dual2d(x_mm, t_s, zero_pad_factor=zero_pad_factor_visu)
Z_pw_mm = utility.fourier.psd2d(Z_mm, x_mm, t_s, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)
W_pw_mm = utility.fourier.psd2d(W_mm, x_mm, t_s, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu)

Z_ang = np.angle(utility.fourier.ft2d(Z_mm, x_mm, t_s, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu))
W_ang = np.angle(utility.fourier.ft2d(W_mm, x_mm, t_s, window=fft_window_visu, zero_pad_factor=zero_pad_factor_visu))


# <codecell>

f0 = 40
k0 = 0.1495
Z_pw_mm[:, 0] *= 2 # for aesthetic purpose


# <codecell>

import matplotlib.colors as col
import hsluv

def make_segmented_cmap():
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    green = '#00ff00'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, green, blue, black], N=256, gamma=1)
    return anglemap

def make_anglemap( N = 256, use_hpl = True ):
    h = np.ones(N) # hue
    h[:N//2] = 11.6 # red 
    h[N//2:] = 258.6 # blue
    s = np.full(N, 100) # saturation
    # l = np.linspace(0, 100, N//2) # luminosity
    # l = np.hstack( (l,l[::-1] ) )

    h[:N//8] = 120 # green
    h[N//8:3*N//8] = 0 # red 
    h[3*N//8:5*N//8] = 120 # green
    h[5*N//8:7*N//8] = 240 # blue
    h[7*N//8:] = 120 # green

    s = np.full(N, 100)
    s[:N//8] = 0 # bw
    s[7*N//8:] = 0 # bw

    ktow = np.linspace(0, 100, N//8) # luminosity
    gtow = np.linspace(65, 100, N // 8) # luminosity
    gtow_red = np.linspace(60, 100, N // 8)
    l = np.hstack((gtow, gtow_red[::-1], gtow_red, gtow[::-1], gtow, gtow[::-1], gtow, gtow[::-1]))

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s[ii], l[ii]) )
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s[ii], l[ii]) )
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0
    return col.ListedColormap( colorlist )

amp_cmap = 'binary'

angle_cmap = 'hsv'
angle_cmap = 'twilight'
angle_cmap = make_segmented_cmap()
angle_cmap =  make_anglemap( use_hpl = False )


# <codecell>

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False, figsize=utility.figsize('double'))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto'}

from matplotlib.colors import Normalize, LogNorm

# vmax, vmin = utility.log_amplitude_range(Z_pw_mm.max(), range_db=range_dB_visu)
# norm_z = LogNorm(vmax=vmax, vmin=vmin, clip=True)(Z_pw_mm)
vmax, vmin = Z_pw_mm.max(), Z_pw_mm.max()/10*0
norm_z = Normalize(vmax=vmax, vmin=vmin, clip=True)

# vmax, vmin = utility.log_amplitude_range(W_pw_mm.max(), range_db=range_dB_visu)
# norm_z = LogNorm(vmax=vmax, vmin=vmin, clip=True)(Z_pw_mm)
vmax, vmin = W_pw_mm.max(), W_pw_mm.max()/10*0
norm_w = Normalize(vmax=vmax, vmin=vmin, clip=True)

def plotline(ax):
    kw = {'color': 'gray', 'lw': 0.5, 'alpha': .5}
    ax.axhline(0, **kw)
    ax.axhline(f0, **kw)
    ax.axvline(0, **kw)
    ax.axvline(k0, **kw)
    

ax = axes[0,0]
im_zpw = ax.imshow(Z_pw_mm, extent=utility.correct_extent(k_visu_mm, f_visu_mm), norm=norm_z, cmap=amp_cmap, **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cbz = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [px^2/(px-1.frame-1)]')
# utility.set_ticks_log_cb(cbz, vmax, range_db=range_dB_visu)
plotline(ax)

ax = axes[0,1]
im_wpw = ax.imshow(W_pw_mm, extent=utility.correct_extent(k_visu_mm, f_visu_mm), norm=norm_w, cmap=amp_cmap, **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cbw = plt.colorbar(im_wpw, ax=ax, label=r'$|\hat{w}|^2$ [px^2/(px-1.frame-1)]')
# utility.set_ticks_log_cb(cbw, vmax, range_db=range_dB_visu)
plotline(ax)


ax = axes[1,0]
im_zpwa = ax.imshow(Z_ang, extent=utility.correct_extent(k_visu_mm, f_visu_mm), alpha=norm_z(Z_pw_mm), cmap=angle_cmap, **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cbza = plt.colorbar(im_zpwa, ax=ax, label=r'$\arg \hat{z}$ [rad]')
cbza.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
plotline(ax)

ax = axes[1,1]
im_zpwa = ax.imshow(W_ang, extent=utility.correct_extent(k_visu_mm, f_visu_mm), alpha=norm_w(W_pw_mm), cmap=angle_cmap, **imshow_kw)
ax.set_xlabel(r'$k$ [px$^{-1}$]')
ax.set_ylabel(r'$f$ [frame$^{-1}$]')
cbwa = plt.colorbar(im_zpwa, ax=ax, label=r'$\arg \hat{w}$ [rad]')
cbwa.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], minor=False)
plotline(ax)

ax.set_xlim(0, .175)
ax.set_ylim(-50, 50)
plt.tight_layout()


# <codecell>

plt.close()


# <codecell>




# <codecell>




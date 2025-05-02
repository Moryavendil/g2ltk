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
acquisition = 'a330'
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

# Step 1: Gaussian blur to remove high-frequency noise

from scipy.ndimage import gaussian_filter
blur_t_frame = 2 # blur in time (frame).
sigma_t = blur_t_frame
blur_x_px = 5 # blur in space (px).
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

# Step 2: W correction due to Z slope

w_slopecorrected = w_tmp / np.sqrt(1 + np.gradient(z_tmp, axis=1)**2)

plt.figure()
wbins = np.linspace(0, w_tmp.max(), 101, endpoint=True)
ax = plt.gca()
ax.hist(w_tmp.flatten(), alpha=.5, bins=wbins, color='b')
ax.hist(w_slopecorrected.flatten(), alpha=.5, bins=wbins, color='k')

w_tmp = w_slopecorrected


# <codecell>

# Step 3: Spatial cleaning of z, we correct the angle of camera
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

# Step 4: Temporal cleaning : we can correct a drift in time

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

t_s = datareading.get_t_frames(acquisition_path, framenumbers) / fr_per_s
t_ms = t_s * 1000
x_mm = datareading.get_x_mm(acquisition_path, framenumbers, subregion=roi, resize_factor=rivfinding_params['resize_factor'], px_per_mm=px_per_mm)

Z_mm = Z / px_per_mm
W_mm = W / px_per_mm


# <codecell>

from matplotlib import cm
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

def make_cmap_from_hsl(h=None, s=None, l=None, N:int=256, use_hpl:bool=True):
    if h is None:
        h = np.linspace(0, 360, N)
    if s is None:
        s = np.full(N, 100)
    if l is None:
        l = np.full(N, 100)

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s[ii], l[ii]) )
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s[ii], l[ii]) )
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0
    return col.ListedColormap( colorlist )

def cmap_continuous(N:int=256, use_hpl:bool=True):
    # Hue
    hue_yellow = 60
    hue_green = 120
    hue_cyan = 180
    hue_magenta = 360
    opptoqu = np.linspace(hue_cyan, hue_magenta, N//4)
    qutocen = np.linspace(hue_magenta, hue_yellow+360, N//4) % 360
    centoqu = np.linspace(hue_yellow, hue_green, N//4)
    qutoopp = np.linspace(hue_green, hue_cyan, N//4)
    h = np.hstack((opptoqu, qutocen, centoqu, qutoopp))

    # Saturation
    s = np.full(N, 100)

    # Luminosity
    l = np.full(N, 65) + 5 * np.cos(np.linspace(0, 4, N)*2*np.pi)

    return make_cmap_from_hsl(h=h, l=l, s=s, N=N, use_hpl=use_hpl)

def cmap_whiteinterrupt(N:int=256, use_hpl:bool=True):

    # discrete
    # Hue
    h = np.ones(N) # hue
    h[:N//8] = 120 # green
    h[N//8:3*N//8] = 0 # red 
    h[3*N//8:5*N//8] = 120 # green
    h[5*N//8:7*N//8] = 240 # blue
    h[7*N//8:] = 120 # green
    # Saturation
    s = np.full(N, 100)
    s[:N//8] = 0 # bw
    s[7*N//8:] = 0 # bw
    # Luminosity
    lummax = 100
    gtow = np.linspace(65, lummax, N // 8) # luminosity
    gtow_red = np.linspace(60, lummax, N // 8)
    l = np.hstack((gtow, gtow_red[::-1], gtow_red, gtow[::-1], gtow, gtow[::-1], gtow, gtow[::-1]))

    return make_cmap_from_hsl(h=h, l=l, s=s, N=N, use_hpl=use_hpl)

def cmap_saturinterrupt(N:int=256, use_hpl:bool=True):

    # discrete
    # Hue
    h = np.ones(N) # hue
    h[:N//8] = 120 # green
    h[N//8:3*N//8] = 0 # red 
    h[3*N//8:5*N//8] = 120 # green
    h[5*N//8:7*N//8] = 240 # blue
    h[7*N//8:] = 120 # green

    # Saturation
    stokw = np.linspace(100, 0, N // 8)
    s = np.hstack((np.full(N//8, 0), stokw[::-1], stokw, stokw[::-1], stokw, stokw[::-1], stokw, np.full(N//8, 0)))

    # Luminosity
    # l = np.full(N, 65)
    lummax = 95
    gtow = np.linspace(65, lummax, N // 8) # luminosity
    gtow_red = np.linspace(60, lummax, N // 8)
    l = np.hstack((gtow, gtow_red[::-1], gtow_red, gtow[::-1], gtow, gtow[::-1], gtow, gtow[::-1]))

    return make_cmap_from_hsl(h=h, l=l, s=s, N=N, use_hpl=use_hpl)

amp_cmap = 'binary'

# angle_cmap = 'hsv'
# angle_cmap = 'twilight'
angle_cmap = make_segmented_cmap()
# angle_cmap = cmap_continuous(use_hpl = False)


# <codecell>

### ATTENUATION
b = .58
b_u = 0.02
nu = 1.
mu = 1 / (b ** 2 / (12 * nu))
mu_u = mu * 2*np.sqrt((b_u/b)**2)
print(f'mu = {round(mu, 3)} pm {round(mu_u, 3)} Hz')

### BULK SPEED
g = 9.81e3
u0 = g / mu
u0_u = u0 * mu_u/mu
print(f'u0 = {round(u0, 3)} pm {round(u0_u, 3)} mm/s')

### CAPILLARY SPEED
gamma = 17
rho = 1.72e-3
w0 = 0.17723893195382692
w0_u = np.sqrt(0.001713682346235076**2 + (1/px_per_mm/5)**2) # combine the standard deviation and the pixel size
# w0_u = np.sqrt(w0_u**2 + (1*mm_per_px)**2) # we might be overestimating w of something like 1 px because of bad lighting condition
print(f'w0 = {round(w0, 3)} pm {round(w0_u, 3)} mm')


Gamma = np.pi * gamma / (2 * rho)
vc = np.sqrt(Gamma / w0)
vc_u = vc * max(0.01, np.sqrt((1/2 * w0_u/w0)**2))
print(f'vc = {round(vc, 3)} pm {round(vc_u, 3)} mm/s')

f0 = 40
omega0 = 2 * np.pi * f0

k_plot = np.linspace(0, 10, 1000)

omz_plot = (u0 - vc) * k_plot
omz_u_plot = k_plot * np.sqrt(u0_u**2 + vc_u**2)

omw_plot = (u0 + vc* w0 * k_plot) * k_plot
omw_u_plot = k_plot * np.sqrt(u0_u**2 + vc* w0 * k_plot * np.sqrt((w0_u/w0)**2 + (vc_u/vc)**2) )

k_th = k_plot[np.argmin(((omw_plot - omz_plot) - omega0)**2)]
print(f'k_th = {k_th}')


# <codecell>

plt.figure()
plt.plot(k_plot, omw_plot)
plt.plot(k_plot, omz_plot)
plt.plot(k_plot, omw_plot - omz_plot)
plt.axhline(omega0)


# <markdowncell>

# ## ##  # T# F#  # 1#  # :#  # v# i# s# u# l# a# i# z# a# t# i# o# n


# <codecell>

zero_pad_factor = (1,1)
window='hamming'

shift = (-0.000*np.pi, -0.002*np.pi)
# shift = (0, 0)

q, f = utility.fourier.dual2d(x_mm, t_s, zero_pad_factor=zero_pad_factor)
Z_pw = utility.fourier.psd2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift)
W_pw = utility.fourier.psd2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift)

Z_ang = np.angle(utility.fourier.ft2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift))
W_ang = np.angle(utility.fourier.ft2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift))

# We flip the space axis (k -> -k) for visualization purpose
q = -q[::-1]
Z_pw = Z_pw[:, ::-1]
W_pw = W_pw[:, ::-1]
Z_ang = Z_ang[:, ::-1]
W_ang = W_ang[:, ::-1]

Q, F = np.meshgrid(q, f)

# # We remove low frequency noise
# to_remove_z = (Q/(0.149/2))**2 + (F/(40/2))**2 < 1
# Z_pw[to_remove_z] = 0
# to_remove_w = (Q/(0.149*4/5))**2 + (F/(40*3/4))**2 < 1
# W_pw[to_remove_w] = 0


# <codecell>

range_db = 80

qmin = 0
qmax = 0.35
fmin = -60
fmax = 100

f0 = 40
ftickspacing = 10
q0 = 0.15 # q0 = 0.149
qtickspacing = 0.05

from matplotlib.colors import Normalize, LogNorm
zpwmax, zpwmin = utility.log_amplitude_range(Z_pw.max(), range_db=range_db)
norm_z = LogNorm(vmax=zpwmax, vmin=zpwmin, clip=True)

wpwmax, wpwmin = utility.log_amplitude_range(W_pw[:, np.argmin((q - q0/2)**2):].max(), range_db=range_db)
norm_w = LogNorm(vmax=wpwmax, vmin=wpwmin, clip=True)

norm_ang = Normalize(vmin=-np.pi, vmax=np.pi, clip=True)


# <codecell>

phase_F = Z_ang[np.argmin((f - 40)**2), np.argmin((q - 0.)**2)]
phase_Z = Z_ang[np.argmin((f - 0)**2), np.argmin((q - 0.149)**2)]
phase_W = W_ang[np.argmin((f - 40)**2), np.argmin((q - 0.149)**2)]

print(f'F: {phase_F} | Z: {phase_Z} | W: {phase_W}')
print('(0-shift: F: 0.03389428359994625 | Z: 0.10300031927051154 | W: -1.4741358488993797)')
print(f'(ideal: 0 | 0 | {-np.pi/2})')


# <codecell>

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=utility.figsize('double'))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'extent': utility.correct_extent(q, f)}

fig.suptitle(f'Cutoff: -{range_db} dB')

ax = axes[0,0]
im_zpw = ax.imshow(Z_pw, norm=norm_z, cmap=utility.cmap_z, **imshow_kw)
cbz = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, zpwmax, range_db=range_db)

ax.fill_between(k_plot, omz_plot - omz_u_plot/2, omz_plot + omz_u_plot/2, color=utility.color_z, **utility.fill_between_kw_default)
ax.plot(k_plot, omz_plot, color=utility.color_z)

ax.fill_between(k_plot, omw_plot - omw_u_plot/2, omw_plot + omw_u_plot/2, color=utility.color_w, **utility.fill_between_kw_default)
ax.plot(k_plot, omw_plot, color=utility.color_w)

ax.axvline(k_th/(2*np.pi))


ax = axes[0,1]
im_wpw = ax.imshow(W_pw, norm=norm_w, cmap=utility.cmap_w, **imshow_kw)
cbz = plt.colorbar(im_wpw, ax=ax, label=r'$|\hat{w}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, wpwmax, range_db=range_db)

ax.fill_between(k_plot, omz_plot - omz_u_plot/2, omz_plot + omz_u_plot/2, color=utility.color_z, **utility.fill_between_kw_default)
ax.plot(k_plot, omz_plot, color=utility.color_z)

ax.fill_between(k_plot, omw_plot - omw_u_plot/2, omw_plot + omw_u_plot/2, color=utility.color_w, **utility.fill_between_kw_default)
ax.plot(k_plot, omw_plot, color=utility.color_w)

ax = axes[1,0]
im_zph = ax.imshow(Z_ang, cmap=angle_cmap, norm=norm_ang, alpha=norm_z(Z_pw), **imshow_kw)
cb_zph = plt.colorbar(im_zph, ax=ax, label=r'$\arg(\hat{z})$ [rad]')
utility.set_yaxis_rad(cb_zph.ax)

ax = axes[1,1]
im_wph = ax.imshow(W_ang, cmap=angle_cmap, norm=norm_ang, alpha=norm_w(W_pw), **imshow_kw)
cb_wph = plt.colorbar(im_wph, ax=ax, label=r'$\arg(\hat{w})$ [rad]')
utility.set_yaxis_rad(cb_wph.ax)

ax.set_yticks(np.arange(-10*f0, 11*f0, f0), minor=False)
ax.set_yticks(np.arange(-50*ftickspacing, 51*ftickspacing, ftickspacing), minor=True)
ax.set_ylim(fmin, fmax)
for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel(r'$\omega/(2\pi)$ [Hz]')

ax.set_xticks(np.arange(-10 * q0, 11 * q0, q0), minor=False)
ax.set_xticks(np.arange(-50 * qtickspacing, 51 * qtickspacing, qtickspacing), minor=True)
ax.set_xlim(qmin, qmax)
for i in range(axes.shape[1]):
    axes[-1, i].set_xlabel(r'$k/(2\pi)$ [mm$^{-1}$]')



# <markdowncell>

# ## ##  # T# F#  # 2# :#  # p# l# o# t#  # n# o# n# l# i# n# a# r#  # m# o# d# e


# <codecell>

zero_pad_factor = (2,2) # Protoypage
# zero_pad_factor = (8,8) # real plot
window='blackman'

shift = (-0.000*np.pi, -0.002*np.pi)
# shift = (0, 0)

q, f = utility.fourier.dual2d(x_mm, t_s, zero_pad_factor=zero_pad_factor)
Z_pw = utility.fourier.psd2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift)
W_pw = utility.fourier.psd2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift)

Z_ang = np.angle(utility.fourier.ft2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift))
W_ang = np.angle(utility.fourier.ft2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor, shift=shift))

# We flip the space axis (k -> -k) for visualization purpose
q = -q[::-1]
Z_pw = Z_pw[:, ::-1]
W_pw = W_pw[:, ::-1]
Z_ang = Z_ang[:, ::-1]
W_ang = W_ang[:, ::-1]

Q, F = np.meshgrid(q, f)

# We remove low frequency noise
to_remove_z = (Q/(0.149/2))**2 + (F/(40/2))**2 < 1
Z_pw[to_remove_z] = 0
to_remove_w = (Q/(0.149*4/5))**2 + (F/(40*3/4))**2 < 1
W_pw[to_remove_w] = 0


# <codecell>

range_db = 80

qmin = 0
qmax = 0.35
fmin = -60
fmax = 100

f0 = 40
ftickspacing = 10
q0 = 0.15 # q0 = 0.149
qtickspacing = 0.05

from matplotlib.colors import Normalize, LogNorm
zpwmax, zpwmin = utility.log_amplitude_range(Z_pw.max(), range_db=range_db)
norm_z = LogNorm(vmax=zpwmax, vmin=zpwmin, clip=True)

wpwmax, wpwmin = utility.log_amplitude_range(W_pw[:, np.argmin((q - q0/2)**2):].max(), range_db=range_db)
norm_w = LogNorm(vmax=wpwmax, vmin=wpwmin, clip=True)

norm_ang = Normalize(vmin=-np.pi, vmax=np.pi, clip=True)


# <codecell>

fig, axes = plt.subplots(1, 2, squeeze=False, sharex=True, sharey=True, figsize=utility.figsize('double', ratio=2.5))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'extent': utility.correct_extent(q, f)}

fig.suptitle(f'Cutoff: -{range_db} dB')

ax = axes[0,0]
im_zpw = ax.imshow(Z_pw, norm=norm_z, cmap=utility.plotting.cmap_z, **imshow_kw)
cbz = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, zpwmax, range_db=range_db)

ax = axes[0,1]
im_wpw = ax.imshow(W_pw, norm=norm_w, cmap=utility.plotting.cmap_w, **imshow_kw)
cbz = plt.colorbar(im_wpw, ax=ax, label=r'$|\hat{w}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, wpwmax, range_db=range_db)

ax.set_yticks(np.arange(-10*f0, 11*f0, f0), minor=False)
ax.set_yticks(np.arange(-50*ftickspacing, 51*ftickspacing, ftickspacing), minor=True)
ax.set_ylim(fmin, fmax)
for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel(r'$\omega/(2\pi)$ [Hz]')

ax.set_xticks(np.arange(-10 * q0, 11 * q0, q0), minor=False)
ax.set_xticks(np.arange(-50 * qtickspacing, 51 * qtickspacing, qtickspacing), minor=True)
ax.set_xlim(qmin, qmax)
for i in range(axes.shape[1]):
    axes[-1, i].set_xlabel(r'$k/(2\pi)$ [mm$^{-1}$]')

# utility.save_graphe('FT_nonlinear')


# <markdowncell>

# ## ##  # T# F#  # 3#  # :#  # p# l# o# t#  # c# o# n# d# i# t# i# o# n#  # d# e#  # r# e# s# o# n# a# n# c# e


# <codecell>

zero_pad_factor = (2,2) # Protoypage
# zero_pad_factor = (8,8) # real plot
window='boxcar'
shift = (-0.000*np.pi, -0.002*np.pi)

q, f = utility.fourier.dual2d(x_mm, t_s, zero_pad_factor=zero_pad_factor)
Z_pw = utility.fourier.psd2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor)
W_pw = utility.fourier.psd2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor)

Z_ang = np.angle(utility.fourier.ft2d(Z_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor))
W_ang = np.angle(utility.fourier.ft2d(W_mm, x_mm, t_s, window=window, zero_pad_factor=zero_pad_factor))

# keep only revelevant part to lighten memory usage
iqmin, iqmax = 3*len(q)//8, len(q)-3*len(q)//8
ifmin, ifmax = 3*len(f)//8, len(f)-3*len(f)//8

q = q[iqmin:iqmax]
f = f[ifmin:ifmax]
Z_pw = Z_pw[ifmin:ifmax, iqmin:iqmax]
W_pw = W_pw[ifmin:ifmax, iqmin:iqmax]
Z_ang = Z_ang[ifmin:ifmax, iqmin:iqmax]
W_ang = W_ang[ifmin:ifmax, iqmin:iqmax]

# We flip the space axis (k -> -k) for visualization purpose
q = -q[::-1]
Z_pw = Z_pw[:, ::-1]
W_pw = W_pw[:, ::-1]
Z_ang = Z_ang[:, ::-1]
W_ang = W_ang[:, ::-1]

# clean 0-freq component of w
W_pw[np.argmin((f + f0/2)**2):np.argmin((f - f0/2)**2), :np.argmin((q - q0/2)**2)] *= 0


# <codecell>

range_db = 20

qmin = 0
qmax = 0.3
fmin = -60
fmax = 60

f0 = 40
ftickspacing = 10
q0 = 0.15
qtickspacing = 0.05

from matplotlib.colors import Normalize, LogNorm
zpwmax, zpwmin = utility.log_amplitude_range(Z_pw.max(), range_db=range_db)
norm_z = LogNorm(vmax=zpwmax, vmin=zpwmin, clip=True)

wpwmax, wpwmin = utility.log_amplitude_range(W_pw[:, np.argmin((q - q0/2)**2):].max(), range_db=range_db)
norm_w = LogNorm(vmax=wpwmax, vmin=wpwmin, clip=True)

norm_ang = Normalize(vmin=-np.pi, vmax=np.pi, clip=True)


# <codecell>

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=utility.figsize('double'))
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'extent': utility.correct_extent(q, f)}

fig.suptitle(f'Cutoff: -{range_db} dB')

ax = axes[0,0]
im_zpw = ax.imshow(Z_pw, norm=norm_z, cmap=utility.plotting.cmap_z, **imshow_kw)
cbz = plt.colorbar(im_zpw, ax=ax, label=r'$|\hat{z}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, zpwmax, range_db=range_db)

ax = axes[0,1]
im_wpw = ax.imshow(W_pw, norm=norm_w, cmap=utility.plotting.cmap_w, **imshow_kw)
cbz = plt.colorbar(im_wpw, ax=ax, label=r'$|\hat{w}|^2$ [mm$^2$/Hz/mm$^{-1}$]')
utility.set_ticks_log_cb(cbz, wpwmax, range_db=range_db)

ax = axes[1,0]
im_zph = ax.imshow(Z_ang, cmap=angle_cmap, norm=norm_ang, alpha=norm_z(Z_pw), **imshow_kw)
cb_zph = plt.colorbar(cm.ScalarMappable(norm=norm_ang, cmap=angle_cmap), ax=ax, label=r'$\arg(\hat{z})$ [rad]')
utility.set_yaxis_rad(cb_zph.ax)

ax = axes[1,1]
im_wph = ax.imshow(W_ang, cmap=angle_cmap, norm=norm_ang, alpha=norm_w(W_pw), **imshow_kw)
cb_wph = plt.colorbar(im_wph, ax=ax, label=r'$\arg(\hat{w})$ [rad]')
utility.set_yaxis_rad(cb_wph.ax)

ax.set_yticks(np.arange(-10*f0, 11*f0, f0), minor=False)
ax.set_yticks(np.arange(-10*ftickspacing/2, 11*ftickspacing, ftickspacing), minor=True)
ax.set_ylim(fmin, fmax)
for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel(r'$\omega/(2\pi)$ [Hz]')

ax.set_xticks(np.arange(-10 * q0, 11 * q0, q0), minor=False)
ax.set_xticks(np.arange(-10 * qtickspacing / 2, 11 * qtickspacing, qtickspacing), minor=True)
ax.set_xlim(qmin, qmax)
for i in range(axes.shape[1]):
    axes[-1, i].set_xlabel(r'$k/(2\pi)$ [mm$^{-1}$]')



# <codecell>

img_zpw = utility.plotting.cmap_z(norm_z(Z_pw))
img_zpw[:, :, -1] = ~np.isclose(np.min(img_zpw, axis=2), 1)

img_wpw = utility.plotting.cmap_w(norm_w(W_pw))
img_wpw[:, :, -1] = ~np.isclose(np.min(img_wpw, axis=2), 1)

img_zph = angle_cmap(norm_ang(Z_ang), alpha=norm_z(Z_pw))
# img_zph[:, :, -1] = norm_z(Z_pw)

img_wph = angle_cmap(norm_ang(W_ang), alpha=norm_w(W_pw))
# img_wph[:, :, -1] = norm_z(W_pw)


# <codecell>

utility.activate_saveplot(False, style='jfm')


# <codecell>

fig, axes = plt.subplots(1, 2, squeeze=False, sharex=True, sharey=True, figsize=utility.figsize('double', ratio=2.5), layout='constrained')
imshow_kw = {'origin':'upper',
             'interpolation':'nearest',
             'aspect':'auto',
             'extent': utility.correct_extent(q, f)}

# fig.suptitle(f'Cutoff: -{range_db} dB')

ax = axes[0,0]
ax.set_title(f'Power spectrum (cutoff at -{range_db} dB)')
ax.imshow(img_zpw, **imshow_kw)
ax.imshow(img_wpw, **imshow_kw)
cbw = plt.colorbar(cm.ScalarMappable(norm=norm_w, cmap=utility.plotting.cmap_w), ax=ax, label=r'Power [mm$^2$/Hz/mm$^{-1}$]', 
                   fraction=0.05, pad=0.04)
cbz = plt.colorbar(cm.ScalarMappable(norm=norm_z, cmap=utility.plotting.cmap_z), ax=ax, ticks=[], 
                   fraction=0.05, pad=0.04)
utility.set_ticks_log_cb(cbz, zpwmax, range_db=range_db)
cbz.ax.set_yticklabels([])
utility.set_ticks_log_cb(cbw, wpwmax, range_db=range_db)

ax = axes[0,1]
ax.set_title(f'Phase (cutoff at -{range_db} dB)')
ax.imshow(img_wph, **imshow_kw)
ax.imshow(img_zph, **imshow_kw)
utility.force_aspect_ratio(ax)
cbph = plt.colorbar(cm.ScalarMappable(norm=norm_ang, cmap=angle_cmap), ax=ax, label=r'Phase $\varphi$', fraction=0.05, pad=0.04)
utility.set_yaxis_rad(cbph.ax)

ax.set_yticks(np.arange(-10*f0, 11*f0, f0), minor=False)
ax.set_yticks(np.arange(-50*ftickspacing, 51*ftickspacing, ftickspacing), minor=True)
ax.set_ylim(fmin, fmax)
for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel(r'$\omega/(2\pi)$ [Hz]')

ax.set_xticks(np.arange(-10 * q0, 11 * q0, q0), minor=False)
ax.set_xticks(np.arange(-50 * qtickspacing, 51 * qtickspacing, qtickspacing), minor=True)
ax.set_xlim(qmin, qmax)
for i in range(axes.shape[1]):
    axes[-1, i].set_xlabel(r'$k/(2\pi)$ [mm$^{-1}$]')


utility.force_aspect_ratio(axes[0,0])
utility.force_aspect_ratio(axes[0,1])

# utility.save_graphe('FT_phase')


# <codecell>

plt.close()


# <codecell>




# <codecell>




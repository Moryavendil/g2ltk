# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from tools.utility import figsize
%matplotlib notebook

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

SAVEPLOT = True
mm_per_in = .1 / 2.54 if SAVEPLOT else .1/2

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.signal import find_peaks, savgol_filter, hilbert

from tools import datareading, utility

utility.configure_mpl()


# <codecell>

# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

# Acquisition selection
acquisition = '10Hz_decal'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

interframes = 1

if dataset == 'Nalight_cleanplate_20240708':
    if acquisition=='10Hz_decal':
        framenumbers = np.arange(2478, datareading.get_number_of_available_frames(acquisition_path), interframes)
        roi = None, None, None, 200  #start_x, start_y, end_x, end_y


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)
frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')


# <codecell>

um_per_px = None
if dataset=='Nalight_cleanplate_20240708':
    um_per_px = 5000 / 888
mm_per_px = um_per_px/1000 if um_per_px is not None else None

density = 1.72
refractive_index = 1.26
lambda_Na_void = 0.589 # in um
gamma = 14
bsur2 = 600/2 # in mm

lambd = lambda_Na_void/refractive_index # in um
h0 = lambd * np.pi / (2 * np.pi)/2 # h0 = lambd/4 in um

from matplotlib.colors import LinearSegmentedColormap
cmap_Na = LinearSegmentedColormap.from_list("cmap_Na", ['black', "xkcd:bright yellow"])

# vmin_Na = np.percentile(frames, 1)
# vmax_Na = np.percentile(frames, 99)


# <codecell>

def extremapoints(n_frame):
    p1 = [0, 68]
    p2 = [width-1, 180]

    if dataset=='Nalight_cleanplate_20240708':
        if acquisition=='10Hz_decal':
            frames = [2477, 2497, 2566, 3084, 5011]
            heights = [76, 68, 68, 68, 55]
            p1[1] = np.interp(n_frame, frames, heights)
    return [p1, p2]


# <codecell>

from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import make_smoothing_spline
from scipy.signal import hilbert


# <markdowncell>

# ## Estimate the relative heights


# <codecell>

sig_raw = np.zeros((length, width))
sig_smoothed = np.zeros((length, width))
sig_norm = np.zeros((length, width))

from tools.fringeswork import findminmaxs, find_cminmax, normalize_for_hilbert

for i_frame_ref, n_frame in enumerate(framenumbers):
    frame = frames[i_frame_ref]

    p1, p2 = extremapoints(n_frame)
    # These are in _pixel_ coordinates
    x1, y1 = p1
    x2, y2 = p2
    dlength = int(np.hypot(x2 - x1, y2 - y1))+1
    x, y = np.linspace(x1, x2, dlength), np.linspace(y1, y2, dlength)

    d = np.hypot(x-x1, y-y1)

    # bourrin, minimal
    l = map_coordinates(frame, np.vstack((y, x)))

    sig_raw[i_frame_ref] = l[:sig_norm.shape[1]]

    d_smoothed = d.copy()
    l_smoothed = savgol_filter(l, 151, 2)

    sig_smoothed[i_frame_ref] = l_smoothed[:sig_norm.shape[1]]

    # Find the peaks (grossier)
    prominence = 5
    distance = 100
    forcedmins=None
    forcedmaxs=None
    if dataset=='Nalight_cleanplate_20240708':
        if acquisition=='10Hz_decal':
            if n_frame==2566:
                forcedmins = [61]

    l_smoothed_normalized = normalize_for_hilbert(l_smoothed, prominence=prominence, distance=distance, forcedmins=forcedmins, forcedmaxs=forcedmaxs)

    sig_norm[i_frame_ref] = l_smoothed_normalized[:sig_norm.shape[1]]


# <codecell>

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
imshow_kw = {'aspect':'auto', 'origin':'lower', 'interpolation':'nearest'}
imshow_kw_rawunits = {'extent': utility.correct_extent(np.arange(sig_raw.shape[1]), framenumbers, origin='lower'), **imshow_kw}
ax = axes[0]
ax.set_title('Signal raw')
ax.imshow(sig_raw, **imshow_kw_rawunits)
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

ax = axes[1]
ax.set_title('Signal smoothed')
ax.imshow(sig_raw, **imshow_kw_rawunits)
# ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

ax = axes[2]
ax.set_title('Signal smoothed, normalized')
ax.imshow(sig_norm, **imshow_kw_rawunits)
# ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')


# <codecell>

from scipy.ndimage import gaussian_filter

blur_fn = .5
blur_px = 5

sig_blurred = gaussian_filter(sig_norm, sigma = (blur_fn, blur_px))


# <codecell>

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
ax = axes[0]
ax.set_title('Signal smoothed, normalized')
ax.imshow(sig_norm, cmap='seismic', vmin=-np.abs(sig_norm).max(), vmax=np.abs(sig_norm).max(), **imshow_kw_rawunits)
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

ax = axes[1]
ax.set_title('Signal smoothed, normalized, blurred')
ax.imshow(sig_blurred, cmap='seismic', vmin=-np.abs(sig_blurred).max(), vmax=np.abs(sig_blurred).max(), **imshow_kw_rawunits)
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')


# <codecell>

sig = sig_blurred
sig = sig[:, ::-1]


# <codecell>

from scipy.signal import hilbert2

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



# <codecell>

# phase_wrapped = -phase_wrapped_hil2D # sometimes you need to switch
phase_wrapped = phase_wrapped_hil2D

phase_unwrapped_HV = np.unwrap(np.unwrap(phase_wrapped,axis=1), axis=0)
phase_unwrapped_VH = np.unwrap(np.unwrap(phase_wrapped,axis=0), axis=1)

from skimage.restoration import unwrap_phase
phase_unwrapped_skimage = unwrap_phase(phase_wrapped)


# <codecell>

phase_unwrapped = phase_unwrapped_skimage


# <codecell>

phase_real = phase_unwrapped.copy()
phase_real -= int(np.rint(phase_real[-1, -1] / np.pi)) * np.pi
print(phase_real.min())
phase_real += (18-3) * 2*np.pi
phase_real -= 3 * 2*np.pi

g = 9.81
nu = 1.
lcap = 910

### DIMENSION
x = np.arange(sig_raw.shape[1]) * um_per_px
t = framenumbers.copy() / acquisition_frequency
X, T = np.meshgrid(x, t)
H = phase_real  / (2 * np.pi) * lambd /2

V = g/(2 * nu) * H**2

dHdX = (H[1:-1, 2:]-H[1:-1, :-2])/(X[1:-1, 2:]-X[1:-1, :-2])
dHdT = (H[2:, 1:-1]-H[:-2, 1:-1])/(T[2:, 1:-1]-T[:-2, 1:-1])
Hnew = H[1:-1, 1:-1]
Tnew = T[1:-1, 1:-1]
Xnew = X[1:-1, 1:-1]
Vnew = V[1:-1, 1:-1]

tnew = t[1:-1]
xnew = x[1:-1]


# <codecell>

tsamples = np.linspace(50, 100, 11) - utility.step(tnew)

i_samples = [np.argmin((tnew - tsample)**2) for tsample in tsamples]
tsamples = tnew[i_samples]
fn_samples = framenumbers[1+np.array(i_samples)]

tt = tsamples - 50

dhdx = [np.mean(dHdX[i-1:i+2, :]) for i in i_samples]
dhdx_u = [np.std(dHdX[i-1:i+2, :]) for i in i_samples]
dhdx, dhdx_u = np.array(dhdx), np.array(dhdx_u)

dhdt = [np.mean(dHdT[i-1:i+2, :]) for i in i_samples]
dhdt_u = [np.std(dHdT[i-1:i+2, :]) for i in i_samples]
dhdt, dhdt_u = np.array(dhdt), np.array(dhdt_u)

h = [np.mean(Hnew[i-1:i+2, :]) for i in i_samples]
h_u = [np.std(Hnew[i-1:i+2, :]) for i in i_samples]
h, h_u = np.array(h), np.array(h_u)
h_u += h0

tsamples_smooth = np.linspace( tnew.min(), tnew.max()-.5, 251)
i_samples_smooth = [np.argmin((tnew - tsample)**2) for tsample in tsamples_smooth]


blur_fn = 5
blur_px = 10
dHdT_forsmooth = gaussian_filter(dHdT, sigma=(blur_fn, blur_px))

dhdt_smooth   = [np.mean(dHdT_forsmooth[max(0, i-blur_fn*2):min(i+blur_fn*2+1, len(dHdT_forsmooth)-1), :]) for i in i_samples_smooth]
dhdt_u_smooth = [np.std( dHdT_forsmooth[max(0, i-blur_fn*2):min(i+blur_fn*2+1, len(dHdT_forsmooth)-1), :]) for i in i_samples_smooth]
dhdt_smooth, dhdt_u_smooth = np.array(dhdt_smooth), np.array(dhdt_u_smooth)


# <codecell>

nframessnapmeasured = [2490, 2739, 2989, 3238, 3487, 3736, 3985, 4234, 4483, 4732, 4982]

npeaks = np.array([10, 12, 11, 9, 9, 8, 8, 8, 7, 7, 8])

snap   = [0.96, 0.54, 0.35, 0.38, 0.335, 0.32, 0.32, 0.34, 0.28, 0.36, 0.37]
snap_u = [0.47, 0.15, 0.07, 0.03, 0.06,  0.05, 0.05, 0.06, 0.10, 0.04, 0.08]
snap, snap_u = np.array(snap), np.array(snap_u)
snap *= 1e-9
snap_u *= 1e-9


# <codecell>

if not np.all(nframessnapmeasured == fn_samples):
    utility.log_error('snap measured at wrong frames !!')

snap_color = '#FF0000'
grav_color = '#0000FF'


# <codecell>

snap_u *= 2 / np.sqrt(npeaks)
dhdx_u *= 2 / np.sqrt(npeaks)
dhdt_u *= 2 / np.sqrt(npeaks)
# dhdt_u_smooth *= 2 / np.sqrt(10)


# <codecell>

draingrav = g/nu * h**2 * dhdx
draingrav_u = draingrav * np.sqrt(2*(h_u / h)**2 + (dhdx_u / dhdx)**2)

drainsnap = g*lcap**2/(3*nu) * h**3 * snap
drainsnap_u = drainsnap * np.sqrt(3*(h_u / h)**2 + (snap_u / snap)**2)


# <codecell>

SAVEPLOT = False or SAVEPLOT
utility.activate_saveplot(True or SAVEPLOT)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=utility.figsize('simple', 'simple'))

ax = axes[0]
ax.errorbar(tt, h, yerr=h_u, ms=5, ls='', marker='o', label='h', mfc='w', color='k')

ddddx = 2
ax.set_xlim(-ddddx, 50+ddddx)
ax.set_ylim(0, 6.5)
ax.set_ylabel(r'$h$ [$\mu$m]')

ax = axes[1]
ax.errorbar(tt, dhdx*1e4, yerr=dhdx_u*1e4, ms=5, ls='', marker='o', label='dhdx', mfc='w', color=grav_color)
ax.set_ylim(0, 1.5e-4*1e4)
ax.set_ylabel(r'$\partial_x h$ [\textpertenthousand]')


ax = axes[2]
ax.errorbar(tt, snap*1e9, yerr = snap_u*1e9, ms=5, ls='', mfc='w', marker='o', label='snap', color=snap_color)
ax.set_ylim(0, 1.5)
ax.set_ylabel(r'$\partial_{xxxx} h$ [mm$^{-3}$]')
ax.set_xlabel(r'Time $t$ [s]')



utility.tighten_graph(h_pad=1, pad=.25)
if SAVEPLOT:
    utility.save_graphe('drainstudy_variables')


# <codecell>

# drainevap = np.full_like(dhdt, 5/1000)
# drainevap_u = np.full_like(dhdt, 5/1000)

draintot = drainsnap + draingrav 
draintot_u = drainsnap_u + draingrav_u 

ylim = (0, .8)

fig, axes = plt.subplots(1, 1, sharex=True)

ax = axes

errbar_kw = {'ls':'', 'lw':1, 'mfc':'w', 'capsize':3, 'xerr':utility.step(tnew)*(1+blur_fn)}

ax.errorbar(tsamples, -dhdt, yerr = dhdt_u, marker='o', color='k', label='dhdt', **errbar_kw)
ax.errorbar(tsamples, draingrav, yerr=draingrav_u, marker='d', color='b', label='draingrav', **errbar_kw)
ax.errorbar(tsamples, drainsnap, yerr = drainsnap_u, marker='P', color='r', label='drainsnap', **errbar_kw)
ax.errorbar(tsamples, draintot, yerr = draintot_u, marker='s', color='m', label='draintot', **errbar_kw)

ax.set_ylim(ylim)
ax.legend()


# <codecell>

SAVEPLOT = False or SAVEPLOT
utility.activate_saveplot(True or SAVEPLOT)


# <codecell>

figw = utility.genfig.figw_aps['simple']

plt.figure(figsize=(figw, figw))
ax = plt.gca()

colors = plt.cm.cool(np.linspace(0, 1, len(tsamples)))[::-1]

for p in range(1, 50):
    plt.axhspan((2*p-1)*h0, (2*p)*h0, color='gray', alpha=.1)

for i_tsample, tsample in enumerate(tsamples):
    i_t = i_samples[i_tsample]

    hhh = np.mean(Hnew[i_t-1:i_t+2, :], axis=0)

    xxx = Xnew[i_t,:]/1000

    ax.plot(xxx, hhh, color=colors[i_tsample], label=rf'$t={str(int(np.rint(tt[i_tsample])))}$'+' s')

ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'Crest height $h$ [$\mu$m]')
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 6.5)
# ax.legend()

utility.tighten_graph()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize

cbaxes = inset_axes(ax, width="50%", height="5%", loc='lower right')
plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=50), cmap=plt.cm.cool_r), cax=cbaxes, ticks=[0, 25, 50], orientation='horizontal', 
             label='$t$ [s]')

cbaxes.xaxis.set_ticks_position('top')
cbaxes.xaxis.set_label_position('top')


if SAVEPLOT:
    utility.save_graphe('drainstudy_heightevolve')


# <codecell>

SAVEPLOT = False or SAVEPLOT
utility.activate_saveplot(True or SAVEPLOT)


# <codecell>

ylim = (0, 700)


figw = utility.figw_aps['simple']

fig, axes = plt.subplots(1, 1, figsize=(figw, figw), sharex=True)

ax = axes

errbar_invisible_kw = {'ls':'', 'lw':1, 'mfc':'w', 'marker':'', 'color':'k', 'capsize':3}

errbar_visible_kw = {'ls':'', 'lw':2, 'mfc':'w', 'capsize':5, 'capthick': 2, 'ms': 5, 'mew': 2, 'zorder':4}

barw = 1
a = .4

tt = tsamples - 50
tt_smooth = tsamples_smooth - 50

fact = 1000

ax.bar(tt - barw/2, drainsnap*fact, width=barw, color=snap_color, alpha=a, label=r'Capillarity')
ax.errorbar(tt - barw/2, drainsnap*fact, yerr = drainsnap_u*fact, alpha=a, **errbar_invisible_kw)

ax.bar(tt + barw/2, draingrav*fact, width=barw, color=grav_color, alpha=a, label=r'Gravity')
ax.errorbar(tt + barw/2, draingrav*fact, yerr=draingrav_u*fact, alpha=a, **errbar_invisible_kw)

ax.fill_between(tt_smooth, (-dhdt_smooth - dhdt_u_smooth)*fact, (-dhdt_smooth + dhdt_u_smooth)*fact, alpha=.2, color='k', label=r'-d$h/$d$t$')
ax.plot(tt_smooth, -dhdt_smooth*fact, ls='-', c='k', label=r'-d$h/$d$t$')
ax.errorbar(tt, draintot*fact, yerr = draintot_u*fact, marker='o', color='m', label='Total', **errbar_visible_kw)
# ax.errorbar(tsamples, -dhdt, yerr = dhdt_u, marker='o', color='k', label='dhdt', **errbar_visible_kw)

ax.set_xlim(-barw, 50+barw)
ax.set_xlabel('Time $t$ [s]')
ax.set_ylim(ylim)
ax.set_ylabel('Height decrease [nm/s]')
ax.legend()

utility.tighten_graph()
if SAVEPLOT:
    utility.save_graphe('drainstudy')


# <codecell>

figw = utility.figw_aps['inset']

fig, axes = plt.subplots(1, 1, figsize=(figw, figw), sharex=True)

ax = axes

errbar_visible_kw = {'ls':'', 'lw':1, 'mfc':'w', 'capsize':3, 'capthick': 1, 'ms': 4, 'mew': 1, 'zorder':4}

ax.fill_between(tt_smooth, (-dhdt_smooth - dhdt_u_smooth)*fact, (-dhdt_smooth + dhdt_u_smooth)*fact, alpha=.2, color='k', label='dhdt')
ax.plot(tt_smooth, (-dhdt_smooth)*fact, ls='-', c='k', label='dhdt')
ax.errorbar(tt, draintot*fact, yerr = draintot_u*fact, marker='o', color='m', label='draintot', **errbar_visible_kw)

ax.set_yscale('log')
ax.set_xlim(-barw, 50+barw)
ax.set_xticks([0, 25, 50])
ax.set_ylim(1e1, 1e3)

utility.tighten_graph()
if SAVEPLOT:
    utility.save_graphe('drainstudy_inset')


# <codecell>

T -= 50
t -= 50


# <codecell>

t


# <codecell>

SAVEPLOT = False or SAVEPLOT
utility.activate_saveplot(True or SAVEPLOT)


# <codecell>

vmin_l = np.percentile(sig_raw, 1) # vmin_Na
vmax_l = np.percentile(sig_raw, 99) # vmax_Na


# <codecell>

# for contourplots

from mpl_toolkits.axes_grid1 import make_axes_locatable

# figsize = utility.figsize('double')
fig, axes = plt.subplots(1, 2, figsize=(utility.genfig.figw['double'], utility.genfig.figw['double']/1.618), sharex=True, sharey=True)

imshow_kw = {'aspect': 'auto', 'origin': 'lower', 'extent': utility.correct_extent(x/1000, t, 'lower'), 'interpolation':'bilinear'}

ax = axes[0]
# ax.set_title('Luminosity (normalized)')
# im = ax.imshow(sig, cmap='bwr', vmin=-1, vmax=1, **imshow_kw)
im = ax.imshow(sig_raw[:, ::-1], cmap=cmap_Na, vmin=vmin_l, vmax=vmax_l, **imshow_kw)
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.set_ylabel('Time $t$ [s]')
ax.set_xlabel('Distance $x$ [mm]')
# ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
# ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel(r'Luminosity [0-255]')
# cax.set_ylabel(r'Normalized luminosity [0-1]')

ax = axes[1]
# ax.set_title('Crest height')
im = ax.imshow(H, cmap='inferno', vmin=0, vmax=6.25, **imshow_kw)
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
# ax.set_ylabel('Time $t$ [s]')
ax.set_xlabel('Distance $x$ [mm]')
ax.plot([], [], color='k', lw=1, ls='--', label=r'$h = p\,\lambda / 2$')
ax.plot([], [], color='w', lw=1, ls='--', label=r'$h = (p+1/2)\,\lambda/2$')
ax.legend()

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel(r'Crest height $h$ [$\mu$m]')
ax.set_ylim(0, 50)

utility.tighten_graph(w_pad = 2.5, pad=0.)
if SAVEPLOT:
    utility.save_graphe('drainstudy_2dmap')


# <codecell>

fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
ax = axes

# Plot the surface.
surf = ax.plot_surface(X, T, H, cmap='coolwarm',
                       linewidth=0, antialiased=False)

ax.set_ylabel('time [s]')
ax.set_xlabel('distance d (a bit ill-defined) [um]')
ax.set_zlabel('height h [um]')


# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
# plt.rcParams['text.usetex'] = True

SAVEPLOT = False
mm_per_in = .1 / 2.54 if SAVEPLOT else .1/2

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.signal import find_peaks, savgol_filter, hilbert

from tools import datareading, utility


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
imshow_kw_rawunits = {'extent': utility.correct_extent_spatio(np.arange(sig_raw.shape[1]), framenumbers, origin='lower'), **imshow_kw}
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
sig_blurred = gaussian_filter(sig_norm, sigma = (.5, 5))


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

# technique 1 : hilbert vertical, with horizontal reference  at height 0
ref_y = 0
hilV = hilbert(sig, axis=1)
phase_wrapped_hilV = np.angle(hilV)
reference_hilV = np.angle(hilbert(sig[ref_y, :]))
phase_wrapped_hilV = phase_wrapped_hilV + np.expand_dims(reference_hilV - phase_wrapped_hilV[ref_y, :], axis=0)

# technique 2 : hilbert horizontal, with vertical reference  at x = 0
ref_x = -1
hilH = hilbert(sig, axis=0)
phase_wrapped_hilH = np.angle(hilH)
reference_hilH = np.angle(hilbert(sig[:, ref_x]))
phase_wrapped_hilH = phase_wrapped_hilH + np.expand_dims(reference_hilH - phase_wrapped_hilH[:, ref_x], axis=1)


# <codecell>

from scipy.signal import hilbert2

# technique 3 : hilbert2 naive
# hil2D = hilbert2(sig)
# hil2D = hilbert2(sig[::-1,:])[::-1,:] # PARFOIS SIL FAUT INVERSER LE BAZARD ???
# hil2D = np.angle(hilbert2(sig[:,:]))
# phase_wrapped_hil2D = np.angle(hil2D)

# technique 4 : hilbert2 on symmetrix matrix
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

# for contourplots

fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=False, sharey=False)

ax = axes[0,0]
ax.set_title('Reference signal')
ax.imshow(sig, aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())

ax = axes[1,0]
ax.set_title('Reference signal flipped x')
ax.imshow(sig[:,::-1], aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())

ax = axes[2,0]
ax.set_title('Reference signal flipped y')
ax.imshow(sig[::-1, :], aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())

ax = axes[3,0]
ax.set_title('Reference signal flipped x and y')
ax.imshow(sig[::-1, ::-1], aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())


ax = axes[0,1]
ax.set_title('Hilbert 2D')

ax.imshow(np.angle(hilbert2(sig[:,:])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

ax = axes[1,1]
ax.set_title('Hilbert 2D flipped x')
ax.imshow(np.angle(hilbert2(sig[:,::-1])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

ax = axes[2,1]
ax.set_title('Hilbert 2D flipped y')
ax.imshow(np.angle(hilbert2(sig[::-1,:])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

ax = axes[3,1]
ax.set_title('Hilbert 2D flipped x and y')
ax.imshow(np.angle(hilbert2(sig[::-1,::-1])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

plt.tight_layout()


# <codecell>

# for contourplots

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

ax = axes[0,0]
ax.set_title('Reference signal')
ax.imshow(sig, aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())

ax = axes[1,0]
ax.set_title('4 signal')
ax.imshow(sig_x4, aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())

ax = axes[0,1]
ax.set_title('hil2D')
ax.imshow(np.angle(hilbert2(sig[:,:])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

ax = axes[1,1]
ax.set_title('hil2D - a sign')
ax.imshow(np.angle(hilbert2(sig_x4[:, :])), aspect='auto', origin='lower', interpolation='nearest',
          cmap='seismic', vmin=-np.pi, vmax=np.pi)

plt.tight_layout()


# <codecell>

# for contourplots
X = np.arange(sig_raw.shape[1])
Y = framenumbers.copy()
X, Y = np.meshgrid(X, Y)

fig, axes = plt.subplots(1, 4, figsize=(12, 8), sharex=True, sharey=True)

ax = axes[0]
ax.set_title('Reference signal')
ax.imshow(sig, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2), interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())
ax.contour(X, Y, sig, linewidths=1, colors='k', levels=(-np.inf, 0, np.inf), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='k', lw=1, ls='--', label='0-crossings of reference signal')
ax.legend()

ax = axes[1]
ax.set_title('Hilbert vertical (scipy hilbert)'+'\n'+'with horizontal reference')
ax.imshow(phase_wrapped_hilV, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1] + .5, framenumbers[0] - interframes / 2, framenumbers[-1] + interframes / 2), cmap='bwr', vmin=-np.pi, vmax=np.pi)
ax.contour(X, Y, sig, linewidths=1, colors='k', levels=(-np.inf, 0, np.inf), linestyles='--')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

ax = axes[2]
ax.set_title('Hilbert horizontal (scipy hilbert)'+'\n'+'with vertical reference')
ax.imshow(phase_wrapped_hilH, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1] + .5, framenumbers[0] - interframes / 2, framenumbers[-1] + interframes / 2), cmap='bwr', vmin=-np.pi, vmax=np.pi)
ax.contour(X, Y, sig, linewidths=1, colors='k', levels=(-np.inf, 0, np.inf), linestyles='--')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

ax = axes[3]
ax.set_title('Hilbert 2D (scipy hilbert2)')
ax.imshow(phase_wrapped_hil2D, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1] + .5, framenumbers[0] - interframes / 2, framenumbers[-1] + interframes / 2), cmap='bwr', vmin=-np.pi, vmax=np.pi)
ax.contour(X, Y, sig, linewidths=1, colors='k', levels=(-np.inf, 0, np.inf), linestyles='--')
ax.set_xlabel('distance d (a bit ill-defined) [px]')

plt.tight_layout()


# <codecell>

# phase_wrapped = -phase_wrapped_hil2D # sometimes you need to switch
phase_wrapped = phase_wrapped_hil2D

phase_unwrapped_HV = np.unwrap(np.unwrap(phase_wrapped,axis=1), axis=0)
phase_unwrapped_VH = np.unwrap(np.unwrap(phase_wrapped,axis=0), axis=1)

from skimage.restoration import unwrap_phase
phase_unwrapped_skimage = unwrap_phase(phase_wrapped)


# <codecell>

# for contourplots
X = np.arange(sig_raw.shape[1])
Y = framenumbers.copy()
X, Y = np.meshgrid(X, Y)

fig, axes = plt.subplots(1, 4, figsize=(15, 8), sharex=True, sharey=True)
ax = axes[0]
ax.set_title('Phase wrapped (hilbert)')
ax.imshow(phase_wrapped, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1] + .5, framenumbers[0] - interframes / 2, framenumbers[-1] + interframes / 2))
ax.contour(X, Y, phase_wrapped, linewidths=1, colors='w', levels=(-np.inf, 0, np.inf), linestyles='-')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='-', label=r'wrap angle = $0$')
ax.legend()

ax = axes[1]
ax.set_title('Phase unwrapped H then V')
ax.imshow(phase_unwrapped_HV, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2))
ax.contour(X, Y, phase_wrapped, linewidths=1, colors='w', levels=(-np.inf, 0, np.inf), linestyles='-')
ax.contour(X, Y, phase_unwrapped_HV, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped_HV.min()/np.pi)-1, int(phase_unwrapped_HV.max()/np.pi)+1, 1), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='-', label=r'wrap angle = $0$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $n\pi$')
ax.legend()

ax = axes[2]
ax.set_title('Phase unwrapped V then H')
ax.imshow(phase_unwrapped_VH, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2))
ax.contour(X, Y, phase_wrapped, linewidths=1, colors='w', levels=(-np.inf, 0, np.inf), linestyles='-')
ax.contour(X, Y, phase_unwrapped_VH, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped_VH.min()/np.pi)-1, int(phase_unwrapped_VH.max()/np.pi)+1, 1), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='-', label=r'wrap angle = $0$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $n\pi$')
ax.legend()

ax = axes[3]
ax.set_title('skimage image_unwrapped')
ax.imshow(phase_unwrapped_skimage, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2))
ax.contour(X, Y, phase_wrapped, linewidths=1, colors='w', levels=(-np.inf, 0, np.inf), linestyles='-')
ax.contour(X, Y, phase_unwrapped_skimage, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped_skimage.min()/np.pi)-1, int(phase_unwrapped_skimage.max()/np.pi)+1, 1), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='-', label=r'wrap angle = $0$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $n\pi$')
ax.legend()


# <codecell>

phase_unwrapped = phase_unwrapped_skimage


# <codecell>

# for contourplots
X = np.arange(sig_raw.shape[1])
Y = framenumbers.copy()
X, Y = np.meshgrid(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharex=True, sharey=True)

ax = axes[0]
ax.set_title('Reference signal')
ax.imshow(sig, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2), interpolation='nearest',
          cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max())
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
ax.legend()

ax = axes[1]
ax.set_title('Phase wrapped (hilbert)')
ax.imshow(phase_wrapped, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1] + .5, framenumbers[0] - interframes / 2, framenumbers[-1] + interframes / 2))
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
ax.legend()

ax = axes[2]
ax.set_title('Phase unwrapped')
ax.imshow(phase_unwrapped, aspect='auto', origin='lower', extent=(-.5, sig_raw.shape[1]+.5, framenumbers[0]-interframes/2, framenumbers[-1]+interframes/2))
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X, Y, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.set_ylabel('time [frames]')
ax.set_xlabel('distance d (a bit ill-defined) [px]')
ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
ax.legend()


# <codecell>

phase_real = phase_unwrapped.copy()
phase_real -= int(np.rint(phase_real[-1, -1] / np.pi)) * np.pi
print(phase_real.min())
phase_real += (18-3) * 2 * np.pi
# phase_real += (2) * 2 * np.pi

g = 9.81
nu = 1.

### DIMENSION
x = np.arange(sig_raw.shape[1]) * um_per_px
t = framenumbers.copy() / acquisition_frequency * interframes
X, T = np.meshgrid(x, t)
H = phase_real  / (2 * np.pi) * lambd /2

V = g/(2 * nu) * H**2

dHdX = (H[1:-1, 2:]-H[1:-1, :-2])/(X[1:-1, 2:]-X[1:-1, :-2])
dHdT = (H[2:, 1:-1]-H[:-2, 1:-1])/(T[2:, 1:-1]-T[:-2, 1:-1])
Hnew = H[1:-1, 1:-1]
Tnew = T[1:-1, 1:-1]
Xnew = X[1:-1, 1:-1]
Vnew = V[1:-1, 1:-1]



# <codecell>

# for contourplots
fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharex=True, sharey=True)

imshow_kw = {'aspect': 'auto', 'origin': 'lower', 'extent': utility.correct_extent_spatio(x/1000, t, 'lower'), 'interpolation':'bilinear'}

ax = axes[0]
ax.set_title('Reference signal')
ax.imshow(sig, cmap='seismic', vmin=-np.abs(sig).max(), vmax=np.abs(sig).max(), **imshow_kw)
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.set_ylabel('Time $t$ [s]')
ax.set_xlabel('Distance $x$ [mm]')
# ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
# ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
# ax.legend()

ax = axes[1]
ax.set_title('Phase wrapped (hilbert)')
ax.imshow(phase_wrapped, **imshow_kw)
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
# ax.set_ylabel('Time $t$ [s]')
ax.set_xlabel('Distance $x$ [mm]')
# ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
# ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
# ax.legend()

ax = axes[2]
ax.set_title('Phase unwrapped')
ax.imshow(phase_unwrapped, **imshow_kw)
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='w', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2-1, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
ax.contour(X/1000, T, phase_unwrapped, linewidths=1, colors='k', levels=np.pi*np.arange(int(phase_unwrapped.min()/2/np.pi-1)*2, int(phase_unwrapped.max()/2/np.pi+1)*2+1, 2), linestyles='--')
# ax.set_ylabel('Time $t$ [s]')
ax.set_xlabel('Distance $x$ [mm]')
ax.plot([], [], color='k', lw=1, ls='--', label=r'unwrap angle = $2n\pi$')
ax.plot([], [], color='w', lw=1, ls='--', label=r'unwrap angle = $(2n+1)\pi$')
ax.legend()

# utility.save_graphe('demo_hilbert2d_10Hzdecal')


# <codecell>

fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
ax = axes

# Plot the surface.
surf = ax.plot_surface(X, T, H, cmap='coolwarm',
                       linewidth=0, antialiased=False)

ax.set_ylabel('time [s]')
ax.set_xlabel('distance d (a bit ill-defined) [um]')
ax.set_zlabel('height h [um]')


# <codecell>

plt.figure()
ax = plt.gca()

tmin = t.min()
tsamples = np.arange(0, t.max() - t.min(), 10)
colors = plt.cm.cool(np.linspace(0, 1, len(tsamples)))[::-1]

for p in range(1, 50):
    plt.axhspan((2*p-1)*h0, (2*p)*h0, color='gray', alpha=.1)

for i_tsample, tsample in enumerate(tsamples):
    i_t = np.argmin((t-(tsample+t.min()))**2)
    ax.plot(Xnew[i_t,:]/1000, Hnew[i_t,:], color=colors[i_tsample], label=rf'$t={str(int(tsample))}$'+' s')

ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'Crest height $h$ [$\mu$m]')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.legend()


# <codecell>

i_x = X.shape[1]//2
# i_x = 1500

tt = Tnew[:, i_x]

dhdx = np.abs(np.mean(dHdX[:, i_x-100:i_x+100], axis=1))
dhdt = np.abs(np.mean(dHdT[:, i_x-100:i_x+100], axis=1))
h = np.abs(np.mean(Hnew[:, i_x-100:i_x+100], axis=1))

fig, axes = plt.subplots(2, 1)

ax = axes[0]
ax.plot(tt, 1000* dhdt, 
        marker='o', color='k', ms=2, label=r'$\partial_t h$')
ax.plot(tt, 1000* g/nu * (h)**2  *dhdx, 
        color='b', marker='o', ms=2, label=r'$gh^2\partial_x h/\nu$')
ax.fill_between(tt, 1000* g/nu * (h-10*h0)**2  *dhdx, 1000* g/nu * (h+10*h0)**2  *dhdx,
                color='b', alpha=.1, label=r'$+-10 h_0$')

ax.legend()
ax.set_xlabel(r'Time $t$ [s]')
ax.set_ylabel(r' [nm/s]')

ax = axes[1]
ax.plot(1000* g/nu * (h)**2  *dhdx, 1000* dhdt,
        color='g', marker='o', ms=2)
ax.plot([0, 50], [0, 50])
ax.set_xlabel(r'$gh^2\partial_x h/\nu$ [nm/s]')
ax.set_ylabel(r'$\partial_t h$ [nm/s]')



# <codecell>

T.shape


# <codecell>

fig, axes = plt.subplots(1, 1)

ax = axes
for i_t in range(50, T.shape[0], 100):
    ax.plot(1000* g/nu * np.mean(Hnew[i_t-10:i_t+10, :], axis=0)**2 * np.mean(dHdX[i_t-10:i_t+10, :], axis=0), 
            -1000* np.mean(dHdT[i_t-10:i_t+10, :], axis=0),
            marker='o', ls='', ms=2)
    # ax.plot(1000* g/nu * (Hnew[i_t])**2 * dHdX[i_t], -1000* dHdT[i_t],
    #         marker='o', ls='', ms=2)
# ax.plot(tt, 1000* g/nu * (h)**2  *dhdx,
#         color='b', marker='o', ms=2, label=r'$gh^2\partial_x h/\nu$')
# ax.fill_between(tt, 1000* g/nu * (h-10*h0)**2  *dhdx, 1000* g/nu * (h+10*h0)**2  *dhdx,
#                 color='b', alpha=.1, label=r'$+-10 h_0$')
# 
# ax.legend()
# ax.set_xlabel(r'Time $t$ [s]')
# ax.set_ylabel(r' [nm/s]')
# 
# ax = axes[1]
# ax.plot(1000* g/nu * (h)**2  *dhdx, 1000* dhdt,
#         color='g', marker='o', ms=2)
ax.plot([0, 50], [0, 50])
ax.set_xlabel(r'$gh^2\partial_x h/\nu$ [nm/s]')
ax.set_ylabel(r'$|\partial_t h|$ [nm/s]')



# <codecell>

fig, ax = plt.subplots()

x = Xnew[i_t][i_x]
dhdt = dHdT[i_t][i_x]
h = Hnew[i_t][i_x] # in um
dhdx = savgol_filter(dHdX[i_t], 501, 2)[i_x]
dhdx = dHdX[i_t][i_x]
dhdx = dHdX[i_t][i_x]

v = g/nu * (h)**2

ax.plot(x, v * dhdx, c='r', lw=2, label=r'$h^2g/\nu\cdot dh/dx$')
ax.plot(x, -dhdt, c='g', lw=2, label=r'$dh/dt$')
ax.legend()


# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




# <codecell>




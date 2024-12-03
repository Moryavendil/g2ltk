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
        framenumbers = np.concatenate((np.arange(2478, 2578, interframes), np.arange(4850, 4950, interframes)))
        roi = None, None, None, 260  #start_x, start_y, end_x, end_y


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
    x1, z1 = p1
    x2, z2 = p2
    dlength = int(np.hypot(x2 - x1, z2 - z1))+1
    x, z = np.linspace(x1, x2, dlength), np.linspace(z1, z2, dlength)

    d = np.hypot(x - x1, z - z1)

    # bourrin, minimal
    l = map_coordinates(frame, np.vstack((z, x)))

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

sig = sig_norm


# <codecell>

# interesting frame
nframeseek = 2480
nframeseek = 4900
# interest point at wich to compute the snap
d_interest = width//2
if nframeseek==2480:
    d_interest = 810
    # d_interest = 1930
    # d_interest = 340
if nframeseek==4900:
    d_interest = 670

xd = np.arange(sig.shape[1])

iframeseek = np.where(framenumbers == nframeseek)[0][0]

frame = frames[iframeseek]
lnorm = sig[iframeseek]
p1, p2 = extremapoints(nframeseek)
# These are in _pixel_ coordinates
x1, z1 = p1
x2, z2 = p2
dlength = int(np.hypot(x2 - x1, z2 - z1))+1
x, z = np.linspace(x1, x2, dlength), np.linspace(z1, z2, dlength)

d = np.hypot(x - x1, z - z1)


i_d_interest = np.argmin((d-d_interest)**2)
xp, zp = x[i_d_interest], z[i_d_interest]

# bourrin, minimal
l = map_coordinates(frame, np.vstack((z, x))).astype(float)
l -= l.mean()
l /= np.abs(l).max()


# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='auto', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=2, c='k')
ax.scatter(xp, zp, fc='r', ec='k', lw=2, s=50, zorder=4)

ax = axes[1]
ax.plot(d, l)
ax.plot(xd, lnorm)
ax.axvline(d_interest, c='b', ls='--')


# <codecell>

if nframeseek==4900:
    length_n = 200
if nframeseek==2480:
    length_n = 150
    
near_point = 50
length_n = 150


nearp = np.abs(d - d_interest) < near_point
slope = np.mean(utility.der1(x[nearp], z[nearp])[1])
a = np.arctan(slope)

xn1, zn1 = xp - np.cos(np.pi/2 + a) * length_n/2, zp - np.sin(np.pi/2 + a) * length_n/2
xn2, zn2 = xp + np.cos(np.pi/2 + a) * length_n/2, zp + np.sin(np.pi/2 + a) * length_n/2

x_n, z_n = np.linspace(xn1, xn2, length_n*2), np.linspace(zn1, zn2, length_n*2)

d_n = np.hypot(x_n - xn1, z_n - zn1) - length_n/2
l_n_centre = map_coordinates(frame, np.vstack((z_n, x_n))).astype(float)


# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='equal', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=1, c='k')
ax.plot(x[nearp], z[nearp], lw=3, c='k')
ax.scatter(xp, zp, fc='r', ec='k', lw=2, s=50, zorder=4)
ax.scatter([xn1, xn2], [zn1, zn2], fc='b', ec='k', lw=2, s=20, zorder=4)
ax.plot(x_n, z_n, lw=1, c='b')

ax = axes[1]
ax.plot(d_n, l_n_centre)
# ax.plot(xd, lnorm)
# ax.axvline(d_interest, c='b', ls='--')


# <codecell>

thalfspan = 5
len_tspan = int(2*thalfspan)+1

xn1t1, zn1t1 = xn1 - np.cos(a) * thalfspan, zn1 - np.sin(a) * thalfspan
xn1t2, zn1t2 = xn1 + np.cos(a) * thalfspan, zn1 + np.sin(a) * thalfspan
xn2t1, zn2t1 = xn2 - np.cos(a) * thalfspan, zn2 - np.sin(a) * thalfspan
xn2t2, zn2t2 = xn2 + np.cos(a) * thalfspan, zn2 + np.sin(a) * thalfspan

d_nt = np.zeros((len_tspan, length_n*2))
l_nt = np.zeros((len_tspan, length_n*2))

for i_tdist, t_dist in enumerate(np.linspace(-thalfspan, thalfspan, len_tspan, endpoint=True)):
    xnt1, znt1 = xn1 + np.cos(a) * t_dist, zn1 + np.sin(a) * t_dist
    xnt2, znt2 = xn2 + np.cos(a) * t_dist, zn2 + np.sin(a) * t_dist
    x_nt, z_nt = np.linspace(xnt1, xnt2, length_n*2), np.linspace(znt1, znt2, length_n*2)
    
    d_nt[i_tdist] = np.hypot(x_n - xn1, z_n - zn1) - length_n/2
    l_nt[i_tdist] = map_coordinates(frame, np.vstack((z_nt, x_nt))).astype(float)

d_n = np.mean(d_nt, axis=0)

l_n = np.mean(l_nt, axis=0)


# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='auto', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=1, c='k')
ax.plot(x[nearp], z[nearp], lw=3, c='k')
ax.scatter(xp, zp, fc='r', ec='w', lw=2, s=50, zorder=4)
ax.scatter([xn1, xn2], [zn1, zn2], fc='b', ec='w', lw=2, s=20, zorder=4)
ax.scatter([xn1t1, xn1t2, xn2t1, xn2t2], [zn1t1, zn1t2, zn2t1, zn2t2], fc='g', ec='w', lw=2, s=10, zorder=4)
ax.plot(x_n, z_n, lw=1, c='b')

ax = axes[1]
ax.plot(d_n, l_n_centre, color='gray')
for i_tdist, t_dist in enumerate(np.linspace(-thalfspan, thalfspan, len_tspan, endpoint=True)):
    ax.plot(d_nt[i_tdist], l_nt[i_tdist], color='gray', alpha=.1)
ax.plot(d_n, l_n, color='k', lw=2)
# ax.axvline(d_interest, c='b', ls='--')


# <codecell>

n_max = 25
n_min = -25

if nframeseek==4900:
    if d_interest == 670:
        n_max = 67
        n_min = -65
if nframeseek==2480:
    if d_interest == 810:
        n_max = 26.7
        n_min = -45.1
    if d_interest == 1930:
        n_max = 25
        n_min = -32
    if d_interest == 340:
        n_max = 30.5
        n_min = -42

valid = (d_n > n_min) & (d_n < n_max)

prominence = 1
distance = .8
mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], prominence=prominence, distance=distance)

# find the central peak
d_steps = np.concatenate((d_n[valid][maxs], d_n[valid][mins]))
d_steps.sort()

# find the central peak
dcentre_verycoarse = d_steps[np.argmax(d_steps[2:] - d_steps[:-2]) + 1]
d_steps_beforecentre = d_steps[np.argmax(d_steps[2:]-d_steps[:-2])]
d_steps_aftercentre = d_steps[np.argmax(d_steps[2:]-d_steps[:-2])+2]
dcentre_coarse = (d_steps_beforecentre + d_steps_aftercentre) / 2
# dcentre_coarse = (d_n[valid])[np.argmax(l_n[valid])]

# estimated brutally the phase shifts
p_steps = (2 * (d_steps <= dcentre_coarse).astype(int) - 1).cumsum() - 1
p_steps -= p_steps.min() # au minimum on a p = 0
phi_steps = p_steps * np.pi



from tools.fringeswork import instantaneous_phase, prepare_signal_for_hilbert

sig_norm = normalize_for_hilbert(l_n[valid], prominence=prominence, distance=distance)

d_phase, phase_wrapped = instantaneous_phase(sig_norm, x=d_n[valid], usesplines=True, symmetrize=True)

# Find the centre (precisely)
phase_zeros = utility.find_roots(d_phase, phase_wrapped)
dcentre_fine = phase_zeros[np.argmin((phase_zeros - dcentre_coarse) ** 2)]

# change the direction
phase_wrapped[d_phase > dcentre_fine] *= -1

phase = np.unwrap(phase_wrapped)
phase -= phase[np.argmin((d_phase - dcentre_fine)**2)]
phase += phi_steps[(np.argmin((d_steps - dcentre_fine)**2))]


# <codecell>

fig, axes = plt.subplots(3, 1, sharex=True, sharey=False)


l_mins_cs, l_maxs_cs = find_cminmax(l_n[valid], x=d_n[valid], prominence=prominence, distance=distance)

ax = axes[0]
ax.plot(d_n, l_n, color='k', alpha=.3)
ax.axvspan(n_min, n_max, color='gray', alpha=.1)
ax.plot(d_n[valid], l_n[valid], color='k', lw=2)
ax.scatter(d_n[valid][maxs], l_n[valid][maxs], s=50, lw=1, ec='k', fc='#FF000088', label='maxs')
ax.scatter(d_n[valid][mins], l_n[valid][mins], s=50, lw=1, ec='k', fc='#0000FF88', label='mins')
ax.plot(d_n[valid], l_maxs_cs(d_n[valid]), color='r', alpha=0.5)
ax.plot(d_n[valid], l_mins_cs(d_n[valid]), color='b', alpha=0.5)
ax.axvline(dcentre_coarse, c='g', ls='--')

ax = axes[1]
ax.plot(d_n[valid], sig_norm, color='k', alpha=.1)
ax.axvline(dcentre_coarse, c='g', ls='--')
ax.axvline(dcentre_fine, c='g', ls=':')
ax.plot(d_phase, phase_wrapped, color='k')
ax.scatter(d_n[valid][mins], np.full(len(d_n[valid][mins]), -np.pi), s=50, lw=1, ec='k', fc='#0000FF88')
ax.scatter(d_n[valid][mins], np.full(len(d_n[valid][mins]), np.pi), s=50, lw=1, ec='k', fc='#0000FF88')
ax.scatter(d_n[valid][maxs], np.full(len(d_n[valid][maxs]), 0), s=50, lw=1, ec='k', fc='#FF000088')

ax = axes[2]
ax.plot(d_phase, phase, color='k')
ax.scatter(d_steps, phi_steps, s=50, ec='k', fc='w', label='center of fringes')


# <codecell>

d_um = d_steps * um_per_px
h_um = phi_steps/(2*np.pi) * lambd / 2

x0_guess = dcentre_fine * um_per_px
h0_guess = h_um.max()
m_guess = 0.
R_guess = 10e3
S_guess = 0.
# p0 = [x0_guess, h0_guess, R_guess, S_guess]

def fitfn2(x, x0, h0, R):
    return h0 - (x-x0)**2 * 1/(2*R)
def fitfn12(x, x0, h0, m, R):
    return h0 + (x-x0) * m - (x-x0)**2 * 1/(2*R)
def fitfn24(x, x0, h0, R, S):
    return h0 - (x-x0)**2 * 1/(2*R) - (x-x0)**4 * 3*S/4
def polyfit_snap_0124(x, x0, h0, m, R, S):
    return h0 + (x-x0) * m - (x-x0)**2 * 1/(2*R) - (x-x0)**4 * 3*S/4

d_test = np.linspace(d_um.min(), d_um.max(), 1000)

popt12, _ = curve_fit(fitfn12, d_um, h_um, p0=[x0_guess, h0_guess, m_guess, R_guess])

popt124, _ = curve_fit(polyfit_snap_0124, d_um, h_um, p0=[popt12[0], popt12[1], popt12[2], popt12[3], S_guess])



# <codecell>




# <codecell>

x0, h0, m, R = popt12


x0, h0, m, R, S = popt124
L = np.sqrt(1  / np.abs(2*S*R))
print(f'L (1): {L} (S = {S})')

# h0, x0, R, S, _ = popt243
# L = np.sqrt(1  / np.abs(2*S*R))
# print(f'L (3): {L}')


# <markdowncell>

# ### CONCLUSION
# 
# Il faut utiliser le modèle 0+1+2 pour l'initial guess (courbure + asymétrie = parabole quelconque) et 0+1+2+4 pour trouver l'évolution de la courbure


# <codecell>

fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, squeeze=False)

ax = axes[0, 0]
# ax.plot(d_phase*um_per_px, phase/(2*np.pi) * lambd / 2, color='k')
ax.scatter(d_um, h_um, s=50, ec='k', fc='w', label='center of fringes')
ax.plot(d_test, fitfn12(d_test, *popt12), color='b', ls='--', label='Best parabola')
ax.plot(d_test, polyfit_snap_0124(d_test, *popt124), color='r', ls='-', label='Best parabola with snap')
# ax.plot(d_test, fitfn243(d_test, *popt243), color='b', ls='-')
ax.legend()
ax.set_ylabel('Film height (- unknown offset) $h$ [um]')
ax.set_xlabel('Transverse distance $z$ [um]')


# <codecell>

S


# <codecell>




# <codecell>




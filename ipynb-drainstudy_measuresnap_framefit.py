# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

import tools.fringeswork
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
in_per_mm = .14/2

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.signal import find_peaks, savgol_filter, hilbert

from tools import datareading, utility, fringeswork


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

nframeseekfort10 = [2478, 2727, 2976, 3225, 3475, 3724, 3973, 4222, 4471, 4720, 4969]

if dataset == 'Nalight_cleanplate_20240708':
    if acquisition=='10Hz_decal':
        framenumbers = np.arange(2478, datareading.get_number_of_available_frames(acquisition_path), interframes)
        framenumbers = np.sort(np.concatenate((np.arange(2478, 2578, interframes), np.arange(4850, 4950, interframes), nframeseekfort10)))
        
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

from matplotlib.colors import LinearSegmentedColormap
cmap_Na = LinearSegmentedColormap.from_list("cmap_Na", ['black', "xkcd:bright yellow"])

vmin = np.percentile(frames, 1)
vmax = np.percentile(frames, 99)


# <codecell>

def extremapoints(n_frame):
    p1 = [0, 68]
    p2 = [width-1, 180]

    if dataset=='Nalight_cleanplate_20240708':
        if acquisition=='10Hz_decal':
            frames1 = [2477, 2497, 2566, 3084, 5011]
            heights1 = [76, 68, 68, 68, 55]
            p1[1] = np.interp(n_frame, frames1, heights1)

            frames2 = [2477, 5011]
            heights2 = [183, 178]
            p2[1] = np.interp(n_frame, frames2, heights2)
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
# nframeseek = 2480
# nframeseek = 4900
nframeseek = nframeseekfort10[0]
# interest point at wich to compute the snap

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

d_freq = utility.estimatesignalfrequency(lnorm, x=xd)
T = 1/d_freq
i_d_interests_potential, _ = find_peaks(np.abs(lnorm), width=T/10)
d_interests_potential = xd[np.sort(i_d_interests_potential)]
d_interests = d_interests_potential[(d_interests_potential -xd.min() > T/4) & (xd.max() - d_interests_potential > T/4)]

n_samplepoints = len(d_interests)

i_d_interests = [np.argmin((d-d_interest)**2) for d_interest in d_interests]
xps, zps = x[i_d_interests], z[i_d_interests]

# bourrin, minimal
l = map_coordinates(frame, np.vstack((z, x))).astype(float)
l -= l.mean()
l /= np.abs(l).max()


colors = plt.cm.hsv(np.linspace(0, 1, n_samplepoints, endpoint=False))
colors = plt.cm.rainbow(np.linspace(0, 1, n_samplepoints, endpoint=False))



# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='auto', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=2, c='k')
ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)

ax = axes[1]
ax.plot(d, l, color='gray', label='Raw luminosity (from image)')
ax.plot(xd, lnorm, color='k', label='Normalized, smoothed luminosity')
for d_interest in d_interests_potential:
    ax.axvline(d_interest, c='r', ls='--', alpha = 1 if d_interest in d_interests else 0.2)
ax.plot([], [], c='r', ls='--', alpha = 1, label='Maxima at which the snap is measured')
ax.plot([], [], c='r', ls='--', alpha = .2, label='Maxima discarded (too close to the edge)')
ax.legend()


# <codecell>

near_point = 50
length_n = 200
if nframeseek==4900:
    length_n = 200
if nframeseek==2480:
    length_n = 150
    
nearps = [np.abs(d - d_interest) < near_point for d_interest in d_interests]
slope = [np.mean(utility.der1(x[nearp], z[nearp])[1]) for nearp in nearps]
a = np.arctan(np.array(slope))

xn1s, zn1s = xps - np.cos(np.pi / 2 + a) * length_n / 2, zps - np.sin(np.pi / 2 + a) * length_n / 2
xn2s, zn2s = xps + np.cos(np.pi / 2 + a) * length_n / 2, zps + np.sin(np.pi / 2 + a) * length_n / 2

x_ns = [np.linspace(xn1s[i], xn2s[i], length_n*2) for i in range(n_samplepoints)]
z_ns = [np.linspace(zn1s[i], zn2s[i], length_n*2) for i in range(n_samplepoints)]

d_ns = [np.hypot(x_ns[i] - xn1s[i], z_ns[i] - zn1s[i]) - length_n/2 for i in range(n_samplepoints)]
l_n_centres = [map_coordinates(frame, np.vstack((z_ns[i], x_ns[i]))).astype(float) for i in range(n_samplepoints)]


# <codecell>

fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='equal', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=1, c='k', alpha=.8, ls=':')
for i_interest, d_interest in enumerate(d_interests):
    nearp = nearps[i_interest]
    # ax.plot(x[nearp], z[nearp], lw=3, c='k')

    x_n, z_n = x_ns[i_interest], z_ns[i_interest]
    color = colors[i_interest]
    ax.plot(x_n, z_n, lw=1, color=color)
# ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)
ax.scatter(xn1s, zn1s, fc=colors, ec='k', lw=1, s=20, zorder=4)
ax.scatter(xn2s, zn2s, fc=colors, ec='k', lw=1, s=20, zorder=4)

ax.set_xlim(0, frame.shape[1])
ax.set_ylim(0, frame.shape[0])
# ax.set_xlabel(r'$x$ [px]')
ax.set_ylabel(r'$z$ [px]')

ax = axes[1]
ax.plot(d, l, color='gray', label='Raw luminosity (from image)')
ax.plot(xd, lnorm, color='k', label='Normalized, smoothed luminosity')
for d_interest in d_interests_potential:
    alpha = 1 if d_interest in d_interests else 0.
    color = 'r' if d_interest not in d_interests else colors[np.where(d_interests == d_interest)[0]]
    ax.axvline(d_interest, c=color, ls='--', alpha=alpha)
ax.plot([], [], c='r', ls='--', alpha = 1, label='Maxima at which the snap is measured')
# ax.plot([], [], c='r', ls='--', alpha = .3, label='Maxima discarded (too close to the edge)')

ax.set_xlim(0, frame.shape[1])
ax.set_xlabel(r'$x$ [px]')
ax.set_ylabel(r'Luminosity [normalized]')
ax.legend()

plt.tight_layout()


# <codecell>

thalfspan = 5
len_tspan = int(2*thalfspan)+1

xn1t1s, zn1t1s = xn1s - np.cos(a) * thalfspan, zn1s - np.sin(a) * thalfspan
xn1t2s, zn1t2s = xn1s + np.cos(a) * thalfspan, zn1s + np.sin(a) * thalfspan
xn2t1s, zn2t1s = xn2s - np.cos(a) * thalfspan, zn2s - np.sin(a) * thalfspan
xn2t2s, zn2t2s = xn2s + np.cos(a) * thalfspan, zn2s + np.sin(a) * thalfspan

d_nts = np.zeros((n_samplepoints, len_tspan, length_n*2))
l_nts = np.zeros((n_samplepoints, len_tspan, length_n*2))

for i_interest, d_interest in enumerate(d_interests):
    for i_tdist, t_dist in enumerate(np.linspace(-thalfspan, thalfspan, len_tspan, endpoint=True)):
        x_n, z_n = x_ns[i_interest], z_ns[i_interest]
        xn1, xn2 = xn1s[i_interest], xn2s[i_interest]
        zn1, zn2 = zn1s[i_interest], zn2s[i_interest]
        
        xnt1, znt1 = xn1 + np.cos(a[i_interest]) * t_dist, zn1 + np.sin(a[i_interest]) * t_dist
        xnt2, znt2 = xn2 + np.cos(a[i_interest]) * t_dist, zn2 + np.sin(a[i_interest]) * t_dist
        
        x_nt, z_nt = np.linspace(xnt1, xnt2, length_n*2), np.linspace(znt1, znt2, length_n*2)

        d_nts[i_interest][i_tdist] = np.hypot(x_n - xn1, z_n - zn1) - length_n/2
        l_nts[i_interest][i_tdist] = map_coordinates(frame, np.vstack((z_nt, x_nt))).astype(float)

d_ns = np.mean(d_nts, axis=1)

l_ns = np.mean(l_nts, axis=1)


# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

ax = axes[0]
ax.imshow(frame, aspect='auto', origin='lower', interpolation='nearest')
ax.plot(x, z, lw=1, c='k')
for i_interest, d_interest in enumerate(d_interests):
    nearp = nearps[i_interest]
    ax.plot(x[nearp], z[nearp], lw=3, c='k')
ax.scatter(xps, zps, fc='r', ec='w', lw=2, s=50, zorder=4)

for i_interest, d_interest in enumerate(d_interests):
    for i_tdist, t_dist in enumerate(np.linspace(-thalfspan, thalfspan, len_tspan, endpoint=True)):
        xn1, xn2 = xn1s[i_interest], xn2s[i_interest]
        zn1, zn2 = zn1s[i_interest], zn2s[i_interest]
        x_n, z_n = x_ns[i_interest], z_ns[i_interest]
        
        xnt1, znt1 = xn1 + np.cos(a[i_interest]) * t_dist, zn1 + np.sin(a[i_interest]) * t_dist
        xnt2, znt2 = xn2 + np.cos(a[i_interest]) * t_dist, zn2 + np.sin(a[i_interest]) * t_dist
        
        x_nt, z_nt = np.linspace(xnt1, xnt2, length_n*2), np.linspace(znt1, znt2, length_n*2)

        color = colors[i_interest]
        ax.plot(x_n, z_n, lw=1, color=color)
ax.scatter(xn1s, zn1s, fc='b', ec='k', lw=2, s=20, zorder=4)
ax.scatter(xn2s, zn2s, fc='b', ec='k', lw=2, s=20, zorder=4)

ax = axes[1]
for i_interest, d_interest in enumerate(d_interests):
    color = colors[i_interest]
    ax.plot(d_ns[i_interest], l_n_centres[i_interest], color=color, alpha=.3)
    ax.plot(d_ns[i_interest], l_ns[i_interest], color=color, lw=2)



# <codecell>

maxfev = 10000

def decide_nminmax(nframe, d:float=0):
    nframes = [2478, 2480, 2727, 3225, 4900]
    n_maxs = [28, 30, 65, 90, 90]
    # n_mins = [-35, -35, -45, -56, -65]
    n_mins = [-40, -40, -45, -56, -65]
    n_max = np.interp(nframe, nframes, n_maxs)
    n_min = np.interp(nframe, nframes, n_mins)
    return n_min, n_max

def peakfinding_parameters_forframe(nframe, d:float=0):
    peak_finding_params = {'prominence': 3, 'distance': 1}
    if nframe==2478:
        peak_finding_params['prominence'] = 1
        if d == 845:
            peak_finding_params['forcedmins'] = [23]
            peak_finding_params['forcedmaxs'] = [24]
    if nframe==2480:
        peak_finding_params['prominence'] = 2

    return peak_finding_params

number_of_points_on_each_side = 8


# <codecell>

Ztots = []
Ptots = []
Zs = []
Ps = []

z_ums = []
h_ums = []

for i_interest, d_interest in enumerate(d_interests):
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)

    valid = (d_n > n_min) & (d_n < n_max)
    mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], **peak_finding_params)

    # find the peaks
    d_steps_tot = np.concatenate((d_n[valid][maxs], d_n[valid][mins]))
    d_steps_tot.sort()
    
    # find the central peak
    i_centralpeak = np.argmin(np.abs(d_steps_tot))
    dcentre = d_steps_tot[i_centralpeak]

    Ztots.append(d_steps_tot)
    
    i_steps_below = max(0, i_centralpeak-number_of_points_on_each_side)
    i_steps_above = min(len(d_steps_tot), i_centralpeak + number_of_points_on_each_side + 1)

    d_steps_taken = d_steps_tot[i_steps_below:i_steps_above]
    if len(d_steps_taken) < number_of_points_on_each_side*2+1:
        utility.log_warn("Oopsy daisy l'intervalle de mesure de snap est trop petit")

    # estimated brutally the phase shifts
    p_steps_taken = (2 * (d_steps_taken <= dcentre).astype(int) - 1).cumsum() - 1
    p_steps_taken -= p_steps_taken.min() # au minimum on a p = 0
    phi_steps_taken = p_steps_taken * np.pi

    Zs.append(d_steps_taken)
    Ps.append(phi_steps_taken)

    p_steps_tot = (2 * (d_steps_tot <= dcentre).astype(int) - 1).cumsum() - 1
    p_steps_tot -= p_steps_tot[i_centralpeak]
    p_steps_tot += p_steps_taken.max()
    Ptots.append(p_steps_tot * np.pi)
    
    ### Do the hilbert thingy
    norm = fringeswork.normalize_for_hilbert(l_n[valid], x=d_n[valid], **peak_finding_params)
    z_hilbert, phase_wrap = fringeswork.instantaneous_phase(norm, x=d_n[valid], usesplines=True)

    phase_wrap_zeros = utility.find_roots(z_hilbert, phase_wrap)
    zcentre_hilbert = phase_wrap_zeros[np.argmin((phase_wrap_zeros - dcentre) ** 2)]
    phase_wrap[z_hilbert > zcentre_hilbert] *= -1
    phase = np.unwrap(phase_wrap)
    phase -= phase[np.argmin((z_hilbert - dcentre)**2)]
    phase += p_steps_taken.max() * np.pi
    
    dmin, dmax = d_steps_tot[i_steps_below], d_steps_tot[i_steps_above-1]
    crit = (z_hilbert >= dmin) & (z_hilbert <= dmax+.5)

    z_ums.append(z_hilbert[crit] * um_per_px)
    h_ums.append(phase[crit] / (2 * np.pi) * lambd / 2)
    
Z_ums = [np.array(Zs[i_interest]) * um_per_px for i_interest in range(n_samplepoints)]
H_ums = [np.array(Ps[i_interest]) / (2 * np.pi) * lambd / 2 for i_interest in range(n_samplepoints)]
H_um_tots = [np.array(Ptots[i_interest]) / (2 * np.pi) * lambd / 2 for i_interest in range(n_samplepoints)]



# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

i_interest = 0
d_interest = d_interests[i_interest]

ax = axes[0]
if True:
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]
    color = colors[i_interest]

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)

    valid = (d_n > n_min) & (d_n < n_max)
    mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], **peak_finding_params)

    cmin, cmax = fringeswork.find_cminmax(l_n[valid], x=d_n[valid], **peak_finding_params)

    ax.plot(d_n, l_n, color=color, alpha=.3)
    ax.plot(d_n[valid], l_n[valid], color=color, lw=2)
    ax.plot(d_n, cmax(d_n), color='r', lw=1, alpha=.3)
    ax.plot(d_n, cmin(d_n), color='b', lw=1, alpha=.3)
    ax.scatter(d_n[valid][maxs], l_n[valid][maxs], s=50, lw=1, fc=color, ec='r', alpha=.8, label='maxs')
    ax.scatter(d_n[valid][mins], l_n[valid][mins], s=50, lw=1, fc=color, ec='b', alpha=.8, label='mins')

ax.set_ylim(np.min(mins)*0.85, np.max(maxs)*1.15)

ax = axes[1]
if True:
    color = colors[i_interest]
    ax.scatter(Z_ums[i_interest], H_ums[i_interest], s=50, fc=color, ec='k')
    ax.plot(z_ums[i_interest], h_ums[i_interest], c='k', lw=.5)



# <codecell>

fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)

ax = axes[0]
ax.axvspan(n_min, n_max, color='gray', alpha=.1)
for i_interest, d_interest in enumerate(d_interests):
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]
    color = colors[i_interest]

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)
        
    valid = (d_n > n_min) & (d_n < n_max)
    mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], **peak_finding_params)
    
    ax.plot(d_n, l_n, color=color, alpha=.3)
    ax.plot(d_n[valid], l_n[valid], color=color, lw=2)
    ax.scatter(d_n[valid][maxs], l_n[valid][maxs], s=50, lw=1, fc=color, ec='r', alpha=.8, label='maxs')
    ax.scatter(d_n[valid][mins], l_n[valid][mins], s=50, lw=1, fc=color, ec='b', alpha=.8, label='mins')

ax = axes[1]
for i_interest, d_interest in enumerate(d_interests):
    color = colors[i_interest] * np.array([1, 1, 1, .3])
    ax.scatter(Zs[i_interest], Ps[i_interest], s=50, fc=color, ec='k')
ax.set_xlabel(r'$\phi$ [rad]')
ax.set_xlabel(r'z [px]')


# <codecell>

def find_snap(z, h, maxfev=None):
    x0_guess = z[np.argmax(h)]
    h0_guess = h.max()
    m_guess = 0.
    if nframeseek == 2478:
        m_guess = 1e-2
    R_guess = 1e4
    S_guess = 0.
    p0 = [x0_guess, h0_guess, m_guess, R_guess]
    limmin = [Z_um.min(), H_um.max() / 2, -np.inf, 0, -np.inf]
    limmax = [z.max(), h.max()*2, np.inf, np.inf, np.inf]
    utility.log_debug(f'p0: {p0}')

    popt12, _ = curve_fit(polyfit_curb_012, z, h, p0=p0, bounds=[limmin[:-1], limmax[:-1]], maxfev=maxfev)
    utility.log_debug(f'popt12: {popt12}')

    popt124, _ = curve_fit(polyfit_snap_0124, z, h, p0=[popt12[0], popt12[1], popt12[2], popt12[3], S_guess], bounds=[limmin, limmax], maxfev=maxfev)
    utility.log_debug(f'popt124: {popt124}')
    S = popt124[4]
    return S


# <codecell>


def polyfit_curb_012(x, x0, h0, m, R):
    return h0 + (x-x0) * m - (x-x0)**2 * 1/(2*R)
def polyfit_snap_0124(x, x0, h0, m, R, Srel):
    return h0 + (x-x0) * m - (x-x0)**2 * 1/(2*R) - (x-x0)**4 * Srel /24
d_test = np.linspace(min([np.min(z_um) for z_um in Z_ums]), max([np.max(z_um) for z_um in Z_ums]), 1000)

from tools import set_verbose
set_verbose('info')


popt_curb = []
popt_snap = []

for i_interest, d_interest in enumerate(d_interests):
    Z_um, H_um = Z_ums[i_interest], H_ums[i_interest]
    # Z_um, H_um = z_ums[i_interest], h_ums[i_interest]

    x0_guess = Z_um[np.argmax(H_um)]
    h0_guess = H_um.max()
    m_guess = 0.
    if nframeseek == 2478:
        m_guess = 1e-2
    R_guess = 1e4
    S_guess = 0.
    p0 = [x0_guess, h0_guess, m_guess, R_guess]
    limmin = [Z_um.min(), H_um.max() / 2, -np.inf, 0, -np.inf]
    limmax = [Z_um.max(), H_um.max() * 2, np.inf, np.inf, np.inf]
    utility.log_debug(f'p0: {p0}')
    
    popt12, _ = curve_fit(polyfit_curb_012, Z_um, H_um, p0=p0, bounds=[limmin[:-1], limmax[:-1]])
    utility.log_debug(f'popt12: {popt12}')
    popt_curb.append(popt12)
    
    popt124, _ = curve_fit(polyfit_snap_0124, Z_um, H_um,
                           p0=[popt12[0], popt12[1], popt12[2], popt12[3], S_guess], bounds=[limmin, limmax],
                           maxfev=maxfev)
    utility.log_debug(f'popt124: {popt124}')
    S = popt124[4]
    utility.log_info(f'S: {S}')
    
    popt_snap.append(popt124)



# <codecell>

fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

i_interest = 0
d_interest = d_interests[i_interest]

ax = axes[0]
if True:
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]
    color = colors[i_interest]

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)

    valid = (d_n > n_min) & (d_n < n_max)
    mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], **peak_finding_params)

    ax.plot(d_n, l_n, color=color, alpha=.3)
    ax.plot(d_n[valid], l_n[valid], color=color, lw=2)
    ax.scatter(d_n[valid][maxs], l_n[valid][maxs], s=50, lw=1, fc=color, ec='r', alpha=.8, label='maxs')
    ax.scatter(d_n[valid][mins], l_n[valid][mins], s=50, lw=1, fc=color, ec='b', alpha=.8, label='mins')

ax = axes[1]
if True:
    color = colors[i_interest]
    ax.scatter(Z_ums[i_interest], H_ums[i_interest], s=50, fc=color, ec='k')
    ax.plot(z_ums[i_interest], h_ums[i_interest], c='k', lw=.5)
    ax.plot(d_test, polyfit_curb_012(d_test, *popt_curb[i_interest]), color=color, ls='--', alpha=.3, label='Best parabola')
    ax.plot(d_test, polyfit_snap_0124(d_test, *popt_snap[i_interest]), color=color, ls='-', label='Best parabola with snap')



# <codecell>

fig, axes = plt.subplots(1, 1, sharex=True, sharey=False)

ax = axes
for i_interest in range(7):
    color = colors[i_interest]
    ax.scatter(Z_ums[i_interest], H_ums[i_interest], s=50, fc=color, ec='k')
    ax.plot(d_test, polyfit_curb_012(d_test, *popt_curb[i_interest]), color=color, ls='--', label='Best parabola')
    ax.plot(d_test, polyfit_snap_0124(d_test, *popt_snap[i_interest]), color=color, ls='-', label='Best parabola with snap')



# <codecell>

snaps = np.array([find_snap(Z_ums[i_interest], H_ums[i_interest], maxfev=maxfev) for i_interest in range(n_samplepoints)])

snaps *= 1e9 # conversion from um-3 to mm-3

crit = np.full(len(snaps), True)
if nframeseek < 2600:
    crit = snaps > 0

snapval = np.mean(snaps[crit]) 
snapincert = np.std(snaps[crit])

utility.log_info(f'snap = {snapval} +- {snapincert}')


# <codecell>

popt_curb


# <codecell>

popt_snap


# <codecell>


def activate_saveplot():
    global in_per_mm
    plt.rcParams['text.usetex'] = True
    in_per_mm = .1 / 2.54
    figwidth = 86*in_per_mm
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'legend.fontsize': 10, 'legend.handlelength': 2,
                         'axes.labelsize': 10, 'axes.titlesize': 10,
                         'figure.labelsize': 10,
                         'savefig.bbox': 'tight', 'savefig.pad_inches': 0., 'savefig.transparent': True,
                         # # tight layout
                         # 'figure.subplot.hspace': 0., 'figure.subplot.wspace': 0.,
                         # # 'figure.subplot.hspace': 0.2, 'figure.subplot.wspace': 0.2,
                         # 'figure.subplot.left': 0, 'figure.subplot.right': 1.,
                         # 'figure.subplot.top': 1., 'figure.subplot.bottom': 0.,
                         # # constrained layout
                         # 'figure.constrained_layout.h_pad': 0., 
                         # 'figure.constrained_layout.w_pad': 0.,
                         })

def deactivate_saveplot():
    global in_per_mm, SAVEPLOT
    plt.rcParams['text.usetex'] = False
    in_per_mm = .14/2
    figwidth = 86*in_per_mm * 2
    figheight = figwidth / 1.618 # golden ratio
    plt.rcParams["figure.figsize"] = (figwidth, figheight)



# <codecell>

deactivate_saveplot()


# <codecell>

fig, axes = plt.subplots(3, 1, figsize=(86*in_per_mm*2, 86*in_per_mm*1.5), sharex=False, sharey=False)

kw_imshow_Na = {'aspect': 'equal', 'origin':'lower', 'cmap':cmap_Na, 'vmin':vmin, 'vmax':vmax}

ax = axes[0]
ax.imshow(frame, interpolation='bilinear', extent=(0, frame.shape[1]*mm_per_px, 0, frame.shape[0]*mm_per_px), **kw_imshow_Na)
ax.plot(x*mm_per_px, z*mm_per_px, lw=1, c='k', ls=':')
for i_interest, d_interest in enumerate(d_interests):
    nearp = nearps[i_interest]
    # ax.plot(x[nearp], z[nearp], lw=3, c='k')

    x_n, z_n = x_ns[i_interest]*mm_per_px, z_ns[i_interest]*mm_per_px
    color = colors[i_interest]
    ax.plot(x_n, z_n, lw=1, color=color)
# ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)
ax.scatter(xn1s*mm_per_px, zn1s*mm_per_px, fc=colors, ec='k', lw=1, s=20, zorder=4)
ax.scatter(xn2s*mm_per_px, zn2s*mm_per_px, fc=colors, ec='k', lw=1, s=20, zorder=4)

ax.set_xlim(0, frame.shape[1]*mm_per_px)
ax.set_ylim(0, frame.shape[0]*mm_per_px)
ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$z$ [mm]')

ax = axes[1]
for i_interest, d_interest in enumerate(d_interests):
    color = colors[i_interest]
    color_faded = color * np.array([1, 1, 1, .3])
    # ax.scatter(Zs[i_interest] + d_interest, h_ums[i_interest], s=50, fc=color, ec='k')
    ax.scatter(Ztots[i_interest] + d_interest, H_um_tots[i_interest], s=50, fc=color, ec='k', alpha=.1)
    ax.scatter(Zs[i_interest] + d_interest, H_ums[i_interest], s=50, fc=color_faded, ec='k')

    d_test = np.linspace(min([np.min(z_um) for z_um in Z_ums]) * 1.6, max([np.max(z_um) for z_um in Z_ums]) * 1.7, 1000)


    ax.plot(d_test / um_per_px + d_interest, polyfit_snap_0124(d_test, *popt_snap[i_interest]), 
            color=color, ls='-', label='Best parabola with snap')

ticks_relative_um = np.array([-500, 0, 500])
ticks_relative_labels = ['0' if tick_relative==0 else str(round(tick_relative, 1)) for tick_relative in ticks_relative_um]
ticks = np.concatenate([d_interests + tick_relative/um_per_px for tick_relative in ticks_relative_um])
ticklabels = np.concatenate([[ticks_relative_label]*n_samplepoints for ticks_relative_label in ticks_relative_labels])
points_on_bottom = np.arange(0, n_samplepoints, 2)
points_on_top = np.arange(1, n_samplepoints, 2)
on_bottom = np.concatenate([points_on_bottom + i*n_samplepoints for i in range(len(ticks_relative_um))])
on_top = np.concatenate([points_on_top + i*n_samplepoints for i in range(len(ticks_relative_um))])

ax.set_xlim(0, frame.shape[1])
ax.set_xticks(ticks[on_bottom])
ax.set_xticklabels(ticklabels[on_bottom])
for i_interest in points_on_bottom:
    color = colors[i_interest]
    ax.get_xticklabels()[i_interest//2 + 0*len(points_on_bottom)].set_color(color)
    ax.get_xticklabels()[i_interest//2 + 1*len(points_on_bottom)].set_color(color)
    ax.get_xticklabels()[i_interest//2 + 2*len(points_on_bottom)].set_color(color)
    
axtop = ax.twiny()
axtop.set_xlim(0, frame.shape[1])
axtop.set_xticks(ticks[on_top])
axtop.set_xticklabels(ticklabels[on_top])
for i_interest in points_on_top:
    color = colors[i_interest]
    axtop.get_xticklabels()[i_interest//2 + 0*len(points_on_top)].set_color(color)
    axtop.get_xticklabels()[i_interest//2 + 1*len(points_on_top)].set_color(color)
    axtop.get_xticklabels()[i_interest//2 + 2*len(points_on_top)].set_color(color)

ax.set_ylim(-1, max([h_um_tot.max() for h_um_tot in H_um_tots]) * 1.25)
if nframeseek == 2478:
    ax.set_ylim(-1, 1.5)
ax.set_xlabel(r'Local distance in the normal direction $\tilde{z}$ [$\mu$m]')
ax.set_ylabel(r'Height $h$ [$\mu$m]')

ax = axes[2]
ax.scatter(d_interests, snaps, s=50, fc=colors, ec='k')
ax.axhline(snapval, c='k', ls='--')
ax.axhspan(snapval-snapincert, snapval+snapincert, alpha=0.1, color='k')

ax.set_xlim(0, frame.shape[1])
# ax.set_xticks(d_interests)
# ax.set_xticklabels(np.arange(n_samplepoints)+1)
# for i_interest in range(n_samplepoints):
#     color = colors[i_interest]
#     ax.get_xticklabels()[i_interest].set_color(color)
ax.set_xticks([])
ax.set_ylim(min(0, snaps.min()*1.15), max(0, snaps.max()*1.15))
# ax.set_xlabel(r'Measurement point')
ax.set_ylabel(r'snap $\partial_{zzzz} h$ [mm$^{-3}$]')

plt.tight_layout()

if SAVEPLOT:
    utility.save_graphe(f'snapmeasure_frame{nframeseek}')


# <codecell>

utility.activate_saveplot()
SAVEPLOT = False
in_per_mm = 1/25.4


# <codecell>

i_interest = 0
if nframeseek==2478:
    i_interest = 0
d_interest = d_interests[i_interest]

# fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=False, sharey=False)
fig = plt.figure(figsize=(86*in_per_mm*2, 86*in_per_mm*.9))

ax = fig.add_subplot(121)
ax.imshow(frame, interpolation='nearest', extent=(0, frame.shape[1]*mm_per_px, 0, frame.shape[0]*mm_per_px), **kw_imshow_Na)
ax.plot(x*mm_per_px, z*mm_per_px, lw=1, c='k', ls=':')
if True:
    nearp = nearps[i_interest]
    # ax.plot(x[nearp], z[nearp], lw=3, c='k')

    x_n, z_n = x_ns[i_interest]*mm_per_px, z_ns[i_interest]*mm_per_px
    color = colors[i_interest]
    ax.plot(x_n, z_n, lw=1, color=color)
# ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)
ax.scatter(xn1s*mm_per_px, zn1s*mm_per_px, fc=colors, ec='k', lw=1, s=20, zorder=4)
ax.scatter(xn2s*mm_per_px, zn2s*mm_per_px, fc=colors, ec='k', lw=1, s=20, zorder=4)

ax.set_xlim((d_interest - frame.shape[0]/2)*mm_per_px, (d_interest + frame.shape[0]/2)*mm_per_px)
ax.set_ylim(0, frame.shape[0]*mm_per_px)
ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$z$ [mm]')

ax = fig.add_subplot(222)
# ax.axvspan(n_min, n_max, color='gray', alpha=.1)
if True:
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]
    color = colors[i_interest]
    color_faded = color * np.array([1, 1, 1, .3])

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)

    valid = (d_n > n_min) & (d_n < n_max)
    mins, maxs = findminmaxs(l_n[valid], x=d_n[valid], **peak_finding_params)

    # find the peaks
    d_steps_tot = np.concatenate((d_n[valid][maxs], d_n[valid][mins]))
    d_steps_tot.sort()

    # find the central peak
    i_centralpeak = np.argmin(np.abs(d_steps_tot))
    dcentre = d_steps_tot[i_centralpeak]
    
    i_steps_below = max(0, i_centralpeak-number_of_points_on_each_side)
    i_steps_above = min(len(d_steps_tot), i_centralpeak + number_of_points_on_each_side + 1)

    d_steps_taken_min = d_steps_tot[i_steps_below]
    d_steps_taken_max = d_steps_tot[i_steps_above]

    maxtaken = (d_n[valid][maxs] >= d_steps_taken_min) & (d_n[valid][maxs] < d_steps_taken_max)
    mintaken = (d_n[valid][mins] >= d_steps_taken_min) & (d_n[valid][mins] < d_steps_taken_max)

    # ax.plot(d_n, l_n, color=color, alpha=.3)
    ax.plot(d_n*um_per_px, l_n, color=color, lw=1)
    ax.scatter(d_n[valid][maxs]*um_per_px, l_n[valid][maxs], s=30, lw=1, fc=color, ec='k', alpha=.2)
    ax.scatter(d_n[valid][maxs][maxtaken]*um_per_px, l_n[valid][maxs][maxtaken], s=30, lw=1, fc=color_faded, ec='k')
    ax.scatter(d_n[valid][mins]*um_per_px, l_n[valid][mins], s=30, lw=1, fc=color, ec='b', alpha=.2)
    ax.scatter(d_n[valid][mins][mintaken]*um_per_px, l_n[valid][mins][mintaken], s=30, lw=1, fc=color_faded, ec='k')
ax.set_ylabel('Luminosity [0-255]')
ax.set_xlim(-200, 200)
ax.set_ylim(100, 160)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

ax = fig.add_subplot(224)
if True:
    color = colors[i_interest]
    color_faded = color * np.array([1, 1, 1, .3])
    # ax.scatter(Zs[i_interest] + d_interest, h_ums[i_interest], s=50, fc=color, ec='k')
    ax.scatter(Ztots[i_interest]*um_per_px, H_um_tots[i_interest], s=30, fc=color, ec='k', alpha=.2)
    ax.scatter(Zs[i_interest]*um_per_px , H_ums[i_interest], s=30, fc=color_faded, ec='k', label='Extrema over which the fit was made')

    d_test = np.linspace(min([np.min(z_um) for z_um in Z_ums]) * 1.6, max([np.max(z_um) for z_um in Z_ums]) * 1.7, 1000)


    ax.plot(d_test, polyfit_snap_0124(d_test, *popt_snap[i_interest]),
            color=color, ls='-', label='Best fitting parabola')
    ax.plot(d_test, polyfit_curb_012(d_test, *popt_curb[i_interest]),
            color=color, ls='--', alpha=.3, label='Best fitting parabola + snap')
ax.legend()
ax.set_xlim(-200, 200)
ax.set_ylim(-1, 1.25)
ax.set_xlabel(r'Local distance in the normal direction $\tilde{z}$ [$\mu$m]')
ax.set_ylabel(r'Height $h$ [$\mu$m]')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
# ax.set_ylim(-1, max([h_um_tot.max() for h_um_tot in H_um_tots]) * 1.15)
# ax.set_xlabel(r'Local distance in the normal direction $\tilde{z}$ [px]')
# ticks_relative = [-50, 0, 50]
# ticks = np.concatenate([d_interests + tick_relative for tick_relative in ticks_relative])
# ticklabels = np.concatenate([[tick_relative]*n_samplepoints for tick_relative in ticks_relative])
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticklabels)
# ax.set_ylabel(r'Height $h$ (- offset) [um]')

plt.tight_layout(pad=0., w_pad=0.1, h_pad=0.)
SAVEPLOT =True
if SAVEPLOT:
    utility.save_graphe(f'snapmeasure_frame{nframeseek}_point{i_interest}')


# <codecell>




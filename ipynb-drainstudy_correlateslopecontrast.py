# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from tools.utility import activate_saveplot
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

nframeseekfort10 = [2490, 2739, 2989, 3238, 3487, 3736, 3985, 4234, 4483, 4732, 4982]

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

            frames2 = [2477, 2490, 5011]
            heights2 = [183, 179, 178]
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

from tools.fringeswork import findminmaxs, find_cminmax, normalize_for_hilbert, findminmaxs_subpixel, findminmax_indexes

for i_frame_ref, n_frame in enumerate(framenumbers):
    frame = frames[i_frame_ref]

    p1, p2 = extremapoints(n_frame)
    # These are in _pixel_ coordinates
    x1, z1 = p1
    x2, z2 = p2
    dlength = int(np.hypot(x2 - x1, z2 - z1))+1
    x_crest, z_crest = np.linspace(x1, x2, dlength), np.linspace(z1, z2, dlength)

    d = np.hypot(x_crest - x1, z_crest - z1)

    # bourrin, minimal
    l = map_coordinates(frame, np.vstack((z_crest, x_crest)))

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

# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
# imshow_kw = {'aspect':'auto', 'origin':'lower', 'interpolation':'nearest'}
# imshow_kw_rawunits = {'extent': utility.correct_extent_spatio(np.arange(sig_raw.shape[1]), framenumbers, origin='lower'), **imshow_kw}
# ax = axes[0]
# ax.set_title('Signal raw')
# ax.imshow(sig_raw, **imshow_kw_rawunits)
# ax.set_ylabel('time [frames]')
# ax.set_xlabel('distance d (a bit ill-defined) [px]')
# 
# ax = axes[1]
# ax.set_title('Signal smoothed')
# ax.imshow(sig_raw, **imshow_kw_rawunits)
# # ax.set_ylabel('time [frames]')
# ax.set_xlabel('distance d (a bit ill-defined) [px]')
# 
# ax = axes[2]
# ax.set_title('Signal smoothed, normalized')
# ax.imshow(sig_norm, **imshow_kw_rawunits)
# # ax.set_ylabel('time [frames]')
# ax.set_xlabel('distance d (a bit ill-defined) [px]')


# <codecell>

sig = sig_norm


# <codecell>

# interesting frame
nframeseek = nframeseekfort10[5]

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
x_crest, z_crest = np.linspace(x1, x2, dlength), np.linspace(z1, z2, dlength)

d = np.hypot(x_crest - x1, z_crest - z1)

d_freq = utility.estimatesignalfrequency(lnorm, x=xd)
T = 1/d_freq
i_d_interests_potential, _ = find_peaks(np.abs(lnorm), width=T/10)
d_interests_potential = xd[np.sort(i_d_interests_potential)]
d_interests = d_interests_potential[(d_interests_potential -xd.min() > T/4) & (xd.max() - d_interests_potential > T/4)]

n_samplepoints = len(d_interests)

i_d_interests = [np.argmin((d-d_interest)**2) for d_interest in d_interests]
xps, zps = x_crest[i_d_interests], z_crest[i_d_interests]

# bourrin, minimal
l = map_coordinates(frame, np.vstack((z_crest, x_crest))).astype(float)
l -= l.mean()
l /= np.abs(l).max()


colors = plt.cm.hsv(np.linspace(0, 1, n_samplepoints, endpoint=False))
colors = plt.cm.rainbow(np.linspace(0, 1, n_samplepoints, endpoint=True))
colors = plt.cm.gnuplot(np.linspace(0, 1, n_samplepoints, endpoint=False))



# <codecell>

# fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)
# 
# ax = axes[0]
# ax.imshow(frame, aspect='auto', origin='lower', interpolation='nearest')
# ax.plot(x_crest, z_crest, lw=2, c='k')
# ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)
# 
# ax = axes[1]
# ax.plot(d, l, color='gray', label='Raw luminosity (from image)')
# ax.plot(xd, lnorm, color='k', label='Normalized, smoothed luminosity')
# for d_interest in d_interests_potential:
#     ax.axvline(d_interest, c='r', ls='--', alpha = 1 if d_interest in d_interests else 0.2)
# ax.plot([], [], c='r', ls='--', alpha = 1, label='Maxima at which the snap is measured')
# ax.plot([], [], c='r', ls='--', alpha = .2, label='Maxima discarded (too close to the edge)')
# ax.legend()


# <codecell>

near_point = 50
length_n = 200
if nframeseek==4900:
    length_n = 200
if nframeseek==2480:
    length_n = 150
    
nearps = [np.abs(d - d_interest) < near_point for d_interest in d_interests]
slope = [np.mean(utility.der1(x_crest[nearp], z_crest[nearp])[1]) for nearp in nearps]
a = np.arctan(np.array(slope))

xn1s, zn1s = xps - np.cos(np.pi / 2 + a) * length_n / 2, zps - np.sin(np.pi / 2 + a) * length_n / 2
xn2s, zn2s = xps + np.cos(np.pi / 2 + a) * length_n / 2, zps + np.sin(np.pi / 2 + a) * length_n / 2

x_ns = [np.linspace(xn1s[i], xn2s[i], length_n*2) for i in range(n_samplepoints)]
z_ns = [np.linspace(zn1s[i], zn2s[i], length_n*2) for i in range(n_samplepoints)]

d_ns = [np.hypot(x_ns[i] - xn1s[i], z_ns[i] - zn1s[i]) - length_n/2 for i in range(n_samplepoints)]
l_n_centres = [map_coordinates(frame, np.vstack((z_ns[i], x_ns[i]))).astype(float) for i in range(n_samplepoints)]


# <codecell>

# fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# 
# ax = axes[0]
# ax.imshow(frame, aspect='equal', origin='lower', interpolation='nearest')
# ax.plot(x_crest, z_crest, lw=1, c='k', alpha=.8, ls=':')
# for i_interest, d_interest in enumerate(d_interests):
#     nearp = nearps[i_interest]
#     # ax.plot(x[nearp], z[nearp], lw=3, c='k')
# 
#     x_n, z_n = x_ns[i_interest], z_ns[i_interest]
#     color = colors[i_interest]
#     ax.plot(x_n, z_n, lw=1, color=color)
# # ax.scatter(xps, zps, fc=colors, ec='k', lw=2, s=50, zorder=4)
# ax.scatter(xn1s, zn1s, fc=colors, ec='k', lw=1, s=20, zorder=4)
# ax.scatter(xn2s, zn2s, fc=colors, ec='k', lw=1, s=20, zorder=4)
# 
# ax.set_xlim(0, frame.shape[1])
# ax.set_ylim(0, frame.shape[0])
# # ax.set_xlabel(r'$x$ [px]')
# ax.set_ylabel(r'$z$ [px]')
# 
# ax = axes[1]
# ax.plot(d, l, color='gray', label='Raw luminosity (from image)')
# ax.plot(xd, lnorm, color='k', label='Normalized, smoothed luminosity')
# for d_interest in d_interests_potential:
#     alpha = 1 if d_interest in d_interests else 0.
#     color = 'r' if d_interest not in d_interests else colors[np.where(d_interests == d_interest)[0]]
#     ax.axvline(d_interest, c=color, ls='--', alpha=alpha)
# ax.plot([], [], c='r', ls='--', alpha = 1, label='Maxima at which the snap is measured')
# # ax.plot([], [], c='r', ls='--', alpha = .3, label='Maxima discarded (too close to the edge)')
# 
# ax.set_xlim(0, frame.shape[1])
# ax.set_xlabel(r'$x$ [px]')
# ax.set_ylabel(r'Luminosity [normalized]')
# ax.legend()
# 
# plt.tight_layout()


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
ax.plot(x_crest, z_crest, lw=1, c='k')
for i_interest, d_interest in enumerate(d_interests):
    nearp = nearps[i_interest]
    ax.plot(x_crest[nearp], z_crest[nearp], lw=3, c='k')
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
    if nframe == 4234 and d == 1862:
        n_min, n_max = -48, 89
    if nframe == 2490 and d > 1700:
        n_min, n_max = -31, 31
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
    if nframe==2490:
        peak_finding_params['prominence'] = 2
        if d==1304:
            peak_finding_params['prominence'] = 3
            

    return peak_finding_params

number_of_points_on_each_side = 12


# <codecell>

i_interest = 5

d_interest = d_interests[i_interest]


# <codecell>

if True:
    d_n = d_ns[i_interest]
    l_n = l_ns[i_interest]

    n_min, n_max = decide_nminmax(nframeseek, d=d_interest)
    peak_finding_params = peakfinding_parameters_forframe(nframeseek, d=d_interest)

    valid = (d_n > n_min) & (d_n < n_max)
    
    d, l = d_n[valid], l_n[valid]
    
    dmins, dmaxs = findminmaxs_subpixel(l, x=d, **peak_finding_params)


    minc, maxc = fringeswork.find_cminmax(l, x=d, **peak_finding_params)

    amp = (maxc(d_n[valid]) - minc(d_n[valid]))/2
    off = (maxc(d_n[valid]) + minc(d_n[valid]))/2

    # find the peaks
    d_steps_tot = np.concatenate((dmins, dmaxs))
    l_steps_tot = np.concatenate((np.interp(dmins, d_n[valid], l_n[valid]), np.interp(dmaxs, d_n[valid], l_n[valid])))
    d_steps_tot_i = d_steps_tot.argsort()
    l_steps_tot = l_steps_tot[d_steps_tot_i]
    d_steps_tot = d_steps_tot[d_steps_tot_i]
    
    # find the central peak
    i_centralpeak = np.argmin(np.abs(d_steps_tot))
    dcentre = d_steps_tot[i_centralpeak]

    # # select steps near centre
    # i_steps_below = max(0, i_centralpeak-number_of_points_on_each_side)
    # i_steps_above = min(len(d_steps_tot), i_centralpeak + number_of_points_on_each_side + 1)
    # 
    # d_steps_taken = d_steps_tot[i_steps_below:i_steps_above]
    # if len(d_steps_taken) < number_of_points_on_each_side*2+1:
    #     utility.log_warn("Oopsy daisy l'intervalle de mesure de snap est trop petit")

    # # estimated brutally the phase shifts
    # p_steps_taken = (2 * (d_steps_taken <= dcentre).astype(int) - 1).cumsum() - 1
    # p_steps_taken -= p_steps_taken.min() # au minimum on a p = 0
    # phi_steps_taken = p_steps_taken * np.pi

    # p_steps_tot = (2 * (d_steps_tot <= dcentre).astype(int) - 1).cumsum() - 1
    # p_steps_tot -= p_steps_tot[i_centralpeak]




# <codecell>

if True:

    ### Do the hilbert thingy
    norm = fringeswork.normalize_for_hilbert(l_n[valid], x=d_n[valid], **peak_finding_params)
    z_hilbert, phase_wrap = fringeswork.instantaneous_phase(norm, x=d_n[valid], usesplines=True)

    phase_wrap_zeros = utility.find_roots(z_hilbert, phase_wrap)
    zcentre_hilbert = phase_wrap_zeros[np.argmin((phase_wrap_zeros - dcentre) ** 2)]
    phase_wrap[z_hilbert > zcentre_hilbert] *= -1
    phase = np.unwrap(phase_wrap)
    phase -= phase[np.argmin((z_hilbert - dcentre)**2)]

    # diemsnions
    z_um_tot = z_hilbert* um_per_px
    h_um_tot = phase / (2 * np.pi) * lambd / 2

    # # select steps near centre
    # dmin, dmax = d_steps_tot[i_steps_below], d_steps_tot[i_steps_above-1]
    # crit = (z_hilbert >= dmin) & (z_hilbert <= dmax+.5)
    # z_um = z_um_tot[crit]
    # h_um = h_um_tot[crit] 



# <codecell>


# coarse
deltad = d_steps_tot[1:]-d_steps_tot[:-1]
d_coarse = (d_steps_tot[1:] + d_steps_tot[:-1])/2
l_coarse = np.abs(l_steps_tot[1:] - l_steps_tot[:-1])
offset_coarse = (maxc(d_coarse) + minc(d_coarse))/2
deltaz = deltad*um_per_px

slope_coarse = lambd/(4*deltaz)
contrast_coarse = l_coarse / offset_coarse

# smoothed
h_um_smoothed = savgol_filter(h_um_tot, 51, 2)
z_sd, dhdz = utility.der2(z_um_tot, h_um_smoothed)

z_slope_fine, slope_fine = utility.der2(z_um_tot, h_um_smoothed)
slope_fine = np.abs(slope_fine)
l_fine = maxc(z_slope_fine / um_per_px) - minc(z_slope_fine / um_per_px)
offset_fine = (maxc(z_slope_fine / um_per_px) + minc(z_slope_fine / um_per_px)) / 2
contrast_fine = l_fine / offset_fine


# <codecell>

SAVEPLOT = False
utility.activate_saveplot(activate = SAVEPLOT)


# <codecell>

### LOOKATME
figw = utility.figw['simple']
fig, axes = plt.subplots(2, 1, figsize=(figw, figw), sharex=True, sharey=False)

color = 'k'

ax = axes[0]
if True:
    ax.plot(d_n*um_per_px, l_n, color='k', lw=1, alpha=.8)
    ax.plot(d*um_per_px, l, color='k', lw=1.5, label='Signal')
    ax.plot(d*um_per_px, off, color='b', alpha=.6, ls=':', lw=1, label='Offset')
    ax.plot(d*um_per_px, off-amp, color='b', alpha=.6, ls='--', lw=1, label = 'Enveloppe')
    ax.plot(d*um_per_px, off+amp, color='b', alpha=.6, ls='--', lw=1)
    ax.scatter(d_steps_tot*um_per_px, l_steps_tot, s=30, lw=1, fc='#00000000', ec='r', label='Extrema')

ax.legend()
ax.set_ylim(100, 160)
# ax.set_yticks(np.arange(-500, 500+1, 250))
ax.set_ylabel(r'Luminosity [0-255]')
# ax.set_xlim(-325, 500)
# ax.set_xlabel(r'$z$ [$\mu$m]')

ax = axes[1]
if True:
    ax.plot(z_um_tot , h_um_smoothed-h_um_smoothed.min(), c='k')
    
    ax.set_ylim(0, 3)
    ax.set_ylabel(r'Height $h$ [$\mu$m]')
    
    axd = ax.twinx()

    axd.plot(z_sd, dhdz*100, c='g')
    sgn = np.sign(np.interp(d_coarse * um_per_px, z_sd, dhdz))
    axd.scatter(d_coarse * um_per_px, slope_coarse*100 * sgn, fc='#00000000', ec='g')
    
    axd.axhline(0, color='g', alpha=.3, ls=(0, (1, 3)))
    
    axd.set_ylim(-2, 2)
    axd.set_ylabel('Slope d$h/$d$z$ [\%]')
    axd.spines['right'].set_color('g')
    axd.yaxis.label.set_color('g')
    axd.tick_params(axis='y', colors='g')

ax.set_xlim(-325, 500)
# ax.set_xticks(np.arange(-500, 500+1, 250))
ax.set_xlabel(r'$z$ [$\mu$m]')

utility.tighten_graph()
if SAVEPLOT:
    utility.save_graphe('luminosityslope')


# <codecell>

SAVEPLOT = True
utility.activate_saveplot(activate = SAVEPLOT)


# <codecell>

### LOOKATME
figw = utility.figw['simple']
fig, axes = plt.subplots(1, 1, figsize=(figw, figw/1.618), sharex=False, sharey=False)

# coarse
deltad = d_steps_tot[1:]-d_steps_tot[:-1]
l_coarse = np.abs(l_steps_tot[1:] - l_steps_tot[:-1])
offset_coarse = (maxc((d_steps_tot[1:] + d_steps_tot[:-1])/2) + minc((d_steps_tot[1:] + d_steps_tot[:-1])/2))/2
deltaz = deltad*um_per_px

slope_coarse = lambd/(4*deltaz)
contrast_coarse = l_coarse / offset_coarse

# smoothed
z_slope_fine, slope_fine = dhdz = utility.der2(z_um_tot, h_um_smoothed)
slope_fine = np.abs(slope_fine)
l_fine = maxc(z_slope_fine / um_per_px) - minc(z_slope_fine / um_per_px)
offset_fine = (maxc(z_slope_fine / um_per_px) + minc(z_slope_fine / um_per_px)) / 2
contrast_fine = l_fine / offset_fine

ax = axes
if True:
    ax.plot(slope_fine*1e2,contrast_fine, color='k', label='Continuous')
    ax.scatter(slope_coarse*1e2, contrast_coarse, label=r'Coarse', fc='#00000000', ec='r')

ax.set_xlim(0, 1.6)
ax.set_xlabel(r'Slope [abs. val., \%]')
ax.set_ylim(0, .35)
ax.set_ylabel(r'Constrast [0-1]')
ax.legend()

utility.tighten_graph()
if SAVEPLOT:
    utility.save_graphe('contrastdependance')


# <codecell>

SAVEPLOT = False
utility.activate_saveplot(activate = SAVEPLOT)


# <codecell>

### LOOKATME
figw = utility.figw['inset']
fig, axes = plt.subplots(1, 1, figsize=(figw, figw/1.618), sharex=False, sharey=False)

# coarse
deltad = d_steps_tot[1:]-d_steps_tot[:-1]
l_coarse = np.abs(l_steps_tot[1:] - l_steps_tot[:-1])
offset_coarse = (maxc((d_steps_tot[1:] + d_steps_tot[:-1])/2) + minc((d_steps_tot[1:] + d_steps_tot[:-1])/2))/2
deltaz = deltad*um_per_px

slope_coarse = lambd/(4*deltaz)
contrast_coarse = l_coarse / offset_coarse

# smoothed
z_slope_fine, slope_fine = dhdz = utility.der2(z_um_tot, h_um_smoothed)
slope_fine = np.abs(slope_fine)
l_fine = maxc(z_slope_fine / um_per_px) - minc(z_slope_fine / um_per_px)
offset_fine = (maxc(z_slope_fine / um_per_px) + minc(z_slope_fine / um_per_px)) / 2
contrast_fine = l_fine / offset_fine

ax = axes
if True:
    ax.plot(slope_fine*1e2,contrast_fine, color='k', label='Continuous')
    ax.scatter(slope_coarse*1e2, contrast_coarse, label=r'Coarse', fc='#00000000', ec='r', s=10)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')
# ax.legend()
ax.set_xlim(.1, 2)
ax.set_ylim(4e-2, 4e-1)

utility.tighten_graph()
if SAVEPLOT:
    utility.save_graphe('contrastdependance_log')


# <codecell>

SAVEPLOT = False
utility.activate_saveplot(activate = SAVEPLOT)


# <codecell>

### LOOKATME

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False, sharey=False)

# coarse
deltad = d_steps_tot[1:]-d_steps_tot[:-1]
l_coarse = np.abs(l_steps_tot[1:] - l_steps_tot[:-1])
offset_coarse = (maxc((d_steps_tot[1:] + d_steps_tot[:-1])/2) + minc((d_steps_tot[1:] + d_steps_tot[:-1])/2))/2
deltaz = deltad*um_per_px

slope_coarse = lambd/(4*deltaz)
contrast_coarse = l_coarse / offset_coarse

# smoothed
z_slope_fine, slope_fine = dhdz = utility.der2(z_um_tot, h_um_smoothed)
slope_fine = np.abs(slope_fine)
l_fine = maxc(z_slope_fine / um_per_px) - minc(z_slope_fine / um_per_px)
offset_fine = (maxc(z_slope_fine / um_per_px) + minc(z_slope_fine / um_per_px)) / 2
contrast_fine = l_fine / offset_fine

ax = axes[0]
if True:
    ax.plot(slope_fine,contrast_fine, color='k', label='approx')
    ax.scatter(slope_coarse, contrast_coarse, label=r'extrema', color='r')

ax.set_xlim(0, 0.016)
ax.set_xlabel('slope')
ax.set_ylim(0, .35)
ax.set_ylabel('constrast')
ax.legend()

ax = axes[1]
if True:
    ax.plot(slope_fine,contrast_fine, color='k', label='approx')
    ax.scatter(slope_coarse, contrast_coarse, label=r'extrema', color='r')

ax.set_xscale('log')
ax.set_xlabel('slope')
ax.set_yscale('log')
ax.set_ylabel('constrast')
ax.set_aspect('equal')
ax.legend()
# ax.legend()



# <codecell>




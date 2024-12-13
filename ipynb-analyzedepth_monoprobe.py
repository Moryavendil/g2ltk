# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

mm_per_in = .1 / 2.54

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
from scipy.signal import hilbert

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
acquisition = '1Hz_start'
acquisition_path = os.path.join(dataset_path, acquisition)
datareading.is_this_a_video(acquisition_path)


# <codecell>

# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        framenumbers = np.arange(1100, 1600)
    elif acquisition=='1Hz_strong':
        framenumbers = np.arange(1600-500, 1600)
    elif acquisition=='2400mHz_start_vrai':
        framenumbers = np.arange(4769-500, 4769)
    elif acquisition=='10Hz_decal':
        framenumbers = np.arange(1500, 1800)


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
lambda_Na_void = 0.589
gamma = 14
bsur2 = 600/2

lambd = lambda_Na_void/refractive_index
h0 = lambd * np.pi / (4 * np.pi)


# <codecell>

from matplotlib.colors import LinearSegmentedColormap

colors = ['black', "xkcd:bright yellow"]
cmap_Na = LinearSegmentedColormap.from_list("mycmap", colors)

vmin = np.percentile(frames, 1)
vmax = np.percentile(frames, 99)


# <codecell>

n_frame_ref = 1345

x_probe = 1400
probespan = 5

interesting_probes = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        # n_frame_ref = 1345
        if n_frame_ref == 1345:
            interesting_probes = [100, 605, 1000, 1400, 1800]
        elif n_frame_ref == 1367:
            interesting_probes = [175, 630, 1030, 1400, 1800]
    if acquisition=='1Hz_strong':
        if n_frame_ref == 1566:
            interesting_probes = [160, 430, 660, 850, 1060, 1260, 1440, 1620, 1820, 2000]
    if acquisition=='2400mHz_start_vrai':
        x_probe = 1024
    if acquisition=='10Hz_decal':
        probespan = 5
        interesting_probes = [480, 1270, 1950]


# <codecell>

i_frame_ref = n_frame_ref - framenumbers[0]

DISP_OFFSET = 0
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        DISP_OFFSET = 820-533

z = np.arange(height)
frame_ref = frames[i_frame_ref]#[::-1, :][DISP_OFFSET:]


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(frame_ref, origin='lower', aspect='auto')
for x_probe_interest in interesting_probes:
    ax.axvline(x_probe_interest, color='k', ls='--', alpha=0.3 if x_probe_interest!= x_probe else 1.)
    ax.axvspan(x_probe_interest - probespan, x_probe_interest + probespan, color='k', alpha=.01)
ax.axvline(x_probe, color='k', ls='--', alpha=1.)
ax.axvspan(x_probe - probespan, x_probe + probespan, color='k', alpha=.1)
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')


# <codecell>

ymax, ymin = None, None

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        ymax, ymin = 820, 820-533
    if acquisition=='10Hz_decal':
        if n_frame_ref == 1673:
            ymax, ymin = 540, None
        
img = frame_ref[ymin:ymax,:]


# <markdowncell>

# ##  # F# i# l# m#  # h# e# i# g# h# t


# <codecell>

print(f'xprobe: {x_probe}', end='\r')

# Luminosity signal
z = np.arange(height)
l_centre = frame_ref[:, x_probe]
l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)

# Interference zone (depends on eveything)
z_interf_min = None
z_interf_max = None
required_prominence = None
required_distance = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        if n_frame_ref == 1345:
            if x_probe==100:
                z_interf_min = 383
                z_interf_max = 604
            elif x_probe==605:
                z_interf_min = 399
                z_interf_max = 611
            elif x_probe==1000:
                z_interf_min = 413
                z_interf_max = 611
            elif x_probe==1400:
                z_interf_min = 425
                z_interf_max = 610
            elif x_probe==1800:
                z_interf_min = 442
                z_interf_max = 611
        elif n_frame_ref == 1367:
            if x_probe==175:
                z_interf_min = 521
                z_interf_max = 739
            elif x_probe==630:
                z_interf_min = 536
                z_interf_max = 741
            elif x_probe==1030:
                z_interf_min = 548
                z_interf_max = 741
            elif x_probe==1400:
                z_interf_min = 557
                z_interf_max = 741
            elif x_probe==1800:
                z_interf_min = 574
                z_interf_max = 742
            z_interf_max = 800
        required_prominence = 5
        required_distance = 2
    if acquisition=='1Hz_strong':
        if n_frame_ref==1566:
            z_interf_max = 1020
            if x_probe==160:
                z_interf_min = 223
        required_prominence = 3
        required_distance = 2
    if acquisition=='10Hz_decal':
        if n_frame_ref == 1673:
            if x_probe==480:
                z_interf_min = 116
                z_interf_max = 309
            if x_probe==1270:
                z_interf_min = 162
                z_interf_max = 336
            if x_probe==1950:
                z_interf_min = 207
                z_interf_max = 347
        required_prominence = 3
        required_distance = 2


# <codecell>


zone_interf = (z >= z_interf_min) * (z <= z_interf_max)
z_interf = z[zone_interf]
l_interf = l_span[zone_interf]

centre_ref = fringeswork.find_uneven_bridge_centre(z, l_span)

# Find the peaks (grossier)

forcedmins=None
forcedmaxs=None
find_peaks_kwargs = {'prominence': required_prominence, 'distance': required_distance,}

mins, maxs = fringeswork.findminmaxs(l_interf, x=z_interf, forcedmins=forcedmins, forcedmaxs=forcedmaxs, **find_peaks_kwargs)

z_maxs = z_interf[maxs]
z_mins = z_interf[mins]

# find the central peak
z_steps = np.concatenate((z_maxs, z_mins))
z_steps.sort()
zcentre_verycoarse = z_steps[np.argmax(z_steps[2:] - z_steps[:-2]) + 1]
z_steps_beforecentre = z_steps[np.argmax(z_steps[2:]-z_steps[:-2])]
z_steps_aftercentre = z_steps[np.argmax(z_steps[2:]-z_steps[:-2])+2]
zcentre_coarse = (z_steps_beforecentre + z_steps_aftercentre) / 2



# <codecell>

fig, axes = plt.subplots(2,1)
ax = axes[0]
ax.plot(z, l_centre, color='k', alpha=.3, label=fr'At centre $x = {x_probe}$')
ax.plot(z, l_span, color='k', lw=2, label=fr'On and $x = {x_probe} \pm {probespan}$')
ax.axvspan(z_interf_min, z_interf_max, color='r', alpha=.1, label='interesting zone')
ax.axvline(centre_ref, color='m', lw=2, label='Riv centre')
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity [0-255]')
ax.legend()

ax = axes[1]
ax.plot(z_interf, l_interf, color='k', lw=2)
for z_max in z_maxs:
    ax.axvline(z_max, color='r', ls='-', alpha=.3)
for z_min in z_mins:
    ax.axvline(z_min, color='b', ls='-', alpha=.3)

ax.scatter(z_interf[maxs], l_interf[maxs], s=50, color='r', label='maxs')
ax.scatter(z_interf[mins], l_interf[mins], s=50, color='b', label='mins')

ax.axvline(zcentre_coarse, color='k', ls=':', alpha=.5, label='centre')
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity [0-255]')
ax.legend()


# <codecell>

# estimated brutally the phase shifts
p_steps = (2 * (z_steps <= zcentre_verycoarse).astype(int) - 1).cumsum() - 1
p_steps -= p_steps.min() # au minimum on a p = 0
phi_steps = p_steps * np.pi

from tools import set_verbose
set_verbose('trace')

### HILBERT PHASE ESTIMATION FOR FANCY BITCHES

l_mins_cs, l_maxs_cs = fringeswork.find_cminmax(l_interf, x=z_interf, forcedmins=forcedmins, forcedmaxs=forcedmaxs, **find_peaks_kwargs)

# the midline
l_offset = (l_maxs_cs(z_interf) + l_mins_cs(z_interf))/2

l_interf_clean = fringeswork.normalize_for_hilbert(l_interf, x=z_interf, forcedmins=forcedmins, forcedmaxs=forcedmaxs, **find_peaks_kwargs)

usesplines = True

z_hilbert, l_hilbert_smoothed = fringeswork.prepare_signal_for_hilbert(l_interf_clean, x=z_interf, usesplines=usesplines)

z_hilbert, amplitude_envelope, instantaneous_phase_wrapped = fringeswork.hilbert_transform(l_interf_clean, x=z_interf, 
                                                                                           usesplines=usesplines, symmetrize=True)

# Find the centre (precisely)
instantaneous_phase_wrapped_zeros = utility.find_roots(z_hilbert, instantaneous_phase_wrapped)
zcentre_better = instantaneous_phase_wrapped_zeros[np.argmin((instantaneous_phase_wrapped_zeros - zcentre_coarse) ** 2)]

# change the direction
instantaneous_phase_wrapped[z_hilbert > zcentre_better] *= -1

# unwrap the angle
phi_hilbert = np.unwrap(instantaneous_phase_wrapped)
phi_hilbert -=phi_hilbert.min()

# if dataset=='Nalight_cleanplate_20240708' and acquisition=='1Hz_start' and n_frame_ref == 1345 and x_probe == 1400:
#     phi_hilbert[z_interf > 429] += 2*np.pi



# <codecell>

fig, axes = plt.subplots(3,1, figsize = (12, 10), sharex=True)

ax = axes[0]
ax.plot(z_interf, l_interf, color='k', lw=2, label='Signal (raw)')

ax.scatter(z_interf[maxs], l_interf[maxs], s=50, color='r', label='maxs')
ax.plot(z_interf, l_maxs_cs(z_interf), color='r', alpha=0.5)
ax.scatter(z_interf[mins], l_interf[mins], s=50, color='b', label='mins')
ax.plot(z_interf, l_mins_cs(z_interf), color='b', alpha=0.5)

ax.plot(z_interf, l_offset, color='k', alpha=0.5, label='Midline (estimated)')

ax.axvline(zcentre_better, color='k', ls=':', alpha=.5, label='Phase extrema (estimated)')
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity [0-255]')
ax.legend()

ax = axes[1]
ax.plot(z_interf, l_interf_clean, color='k', lw=0, marker='o', label='Signal (shifted)')
ax.plot(z_hilbert, l_hilbert_smoothed, color='k', lw=2, label='Signal (shifted, smoothed)')
ax.plot(z_hilbert, amplitude_envelope, color='r', lw=2, label='Amplitude (Hilbert)')
ax.plot(z_hilbert, -amplitude_envelope, color='r', lw=2)
# ax.plot(z_hilbert, l_offset-l_offset, color='k', alpha=0.5)

ax.axvline(zcentre_better, color='k', ls=':', alpha=.5)
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity - midline')
ax.legend()

ax = axes[2]
ax.plot(z_hilbert, instantaneous_phase_wrapped, color='k', alpha=.3)
ax.plot(z_hilbert, phi_hilbert, color='k', lw=2)

ax.scatter(z_steps, phi_steps, s=30, ec='k', fc='w', label='center of fringes')

ax.scatter(z_interf[maxs], np.zeros(len(z_interf[maxs])), s=30, ec='w', fc='k', label='maxs-mins')
ax.scatter(z_interf[mins], np.full(len(z_interf[mins]), np.pi), s=30, ec='w', fc='k')
ax.scatter(z_interf[mins], np.full(len(z_interf[mins]), -np.pi), s=30, ec='w', fc='k')

ax.axvline(zcentre_better, color='k', ls=':', alpha=.5, label='Phase extrema (estimated)')
ax.set_xlabel('z [px]')
ax.set_ylabel(r'Phase unwrapped $\Phi = 2\pi (2h)/\lambda$ [rad]')
pticks = np.arange(-2, (int(phi_steps.max() / (2 * np.pi)) + 2) * 2 + 1, 2)
ax.set_ylim(pticks.min()*np.pi, pticks.max()*np.pi)
ax.set_yticks(pticks*np.pi)
ax.set_yticklabels([fr'${p}\pi$' for p in pticks])
ax.legend()


# <codecell>

z_um = z_hilbert * um_per_px
zcentre_um = zcentre_better * um_per_px
h_um = phi_hilbert / (2 * np.pi) * lambd / 2


# <codecell>

number_of_points_on_each_side = 8

zmi = z_steps[z_steps < zcentre_better][-number_of_points_on_each_side]
zma = z_steps[z_steps > zcentre_better][number_of_points_on_each_side+1]

utility.log_debug(f'{zmi, zma}')
zone_for_snap_estimation = [zmi, zma]
# zone_for_snap_estimation = [450, 600]
# zone_for_snap_estimation = [425, 600]
# zone_for_snap_estimation = [450 - 25, 600 + 25]
# zone_for_snap_estimation = [450 + 25, 600 - 25]
# zone_for_snap_estimation = [450 + 50, 600 - 50]
in_zone = (z_hilbert > zone_for_snap_estimation[0]) & (z_hilbert < zone_for_snap_estimation[1])

z_snapestim = z_um[in_zone] - zcentre_um
h_snapestim = h_um[in_zone]


poly4 = np.polyfit(z_snapestim, h_snapestim, 4)

h_polyfit = np.poly1d(poly4)(z_snapestim)

S = -poly4[0]*4*3*2

utility.log_info(f'xprobe: {x_probe} px')
utility.log_info(f'S = {S*1e9} mm-3')

hmax = h_um.max()

zmax = np.max(np.abs(z_um - zcentre_um))

utility.log_info(f'hmax: {hmax/1000} mm | zmax: {zmax/1000} mm')

utility.log_info(str(4/3 * hmax / (zmax**4) * 1e9)+' mm-3')



# <codecell>




# <codecell>

fig, axes = plt.subplots(3,1, figsize = (12, 10), sharex=True)

ax = axes[0]
ax.plot(z_snapestim, h_snapestim, color='k', lw=1, label=r'$h(z)$')
ax.plot(z_snapestim, h_polyfit, color='r', ls='--', lw=1, label=r'$h(z)$')

# ax.scatter(z_interf[maxs], l_interf[maxs], s=50, color='r', label='maxs')
# ax.plot(z_interf, l_maxs_cs(z_interf), color='r', alpha=0.5)
# ax.scatter(z_interf[mins], l_interf[mins], s=50, color='b', label='mins')
# ax.plot(z_interf, l_mins_cs(z_interf), color='b', alpha=0.5)
# 
# ax.plot(z_interf, l_offset, color='k', alpha=0.5, label='Midline (estimated)')
# 
# ax.axvline(zcentre_better, color='k', ls=':', alpha=.5, label='Phase extrema (estimated)')
# ax.set_xlabel('z [px]')
# ax.set_ylabel('luminosity [0-255]')
# ax.legend()
# 
# ax = axes[1]
# ax.plot(z_interf, l_interf_clean, color='k', lw=0, marker='o', label='Signal (shifted)')
# ax.plot(z_hilbert, l_hilbert_smoothed, color='k', lw=2, label='Signal (shifted, smoothed)')
# ax.plot(z_hilbert, amplitude_envelope, color='r', lw=2, label='Amplitude (Hilbert)')
# ax.plot(z_hilbert, -amplitude_envelope, color='r', lw=2)
# # ax.plot(z_hilbert, l_offset-l_offset, color='k', alpha=0.5)
# 
# ax.axvline(zcentre_better, color='k', ls=':', alpha=.5)
# ax.set_xlabel('z [px]')
# ax.set_ylabel('luminosity - midline')
# ax.legend()
# 
# ax = axes[2]
# ax.plot(z_hilbert, instantaneous_phase_wrapped, color='k', alpha=.3)
# ax.plot(z_hilbert, phi_hilbert, color='k', lw=2)
# 
# ax.scatter(z_steps, phi_steps, s=30, ec='k', fc='w', label='center of fringes')
# 
# ax.scatter(z_interf[maxs], np.zeros(len(z_interf[maxs])), s=30, ec='w', fc='k', label='maxs-mins')
# ax.scatter(z_interf[mins], np.full(len(z_interf[maxs]), np.pi), s=30, ec='w', fc='k')
# ax.scatter(z_interf[mins], np.full(len(z_interf[maxs]), -np.pi), s=30, ec='w', fc='k')
# 
# ax.axvline(zcentre_better, color='k', ls=':', alpha=.5, label='Phase extrema (estimated)')
# ax.set_xlabel('z [px]')
# ax.set_ylabel(r'Phase unwrapped $\Phi = 2\pi (2h)/\lambda$ [rad]')
# pticks = np.arange(-2, (int(phi_steps.max() / (2 * np.pi)) + 2) * 2 + 1, 2)
# ax.set_ylim(pticks.min()*np.pi, pticks.max()*np.pi)
# ax.set_yticks(pticks*np.pi)
# ax.set_yticklabels([fr'${p}\pi$' for p in pticks])
# ax.legend()


# <codecell>




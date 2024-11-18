# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif', 'font.size': 20,
                     'figure.titlesize' : 20,
                     'axes.labelsize': 20,'axes.titlesize': 20,
                     'legend.fontsize': 20, 'legend.handlelength': 2})
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

n_frame_ref = 1673

x_probe = 400
probespan = 5

interesting_probes = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
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
    ax.axvline(x_probe_interest, color='k', ls='--', alpha=0.3)
    ax.axvspan(x_probe_interest - probespan, x_probe_interest + probespan, color='k', alpha=.01)
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


# <codecell>

fig, axes = plt.subplots(2, 2, figsize = (130 * mm_per_in, 130 * mm_per_in))
gs = axes[0,0].get_gridspec()
for ax in axes[:, 0]:
    ax.remove()
ax = fig.add_subplot(gs[:, 0])
ax.imshow(img[::-1, :].T, extent=[-mm_per_px/2, (img.shape[0]+1/2)*mm_per_px, (img.shape[1]+1/2)*mm_per_px, -mm_per_px/2], origin='upper', aspect='equal',
          cmap=cmap_Na, vmin=vmin, vmax=vmax)
for x_probe_interest in interesting_probes:
    ax.axhspan((x_probe_interest-probespan)*mm_per_px, (x_probe_interest+probespan)*mm_per_px, color='k', ls='', alpha=0.5)
    # ax.axhline(x_probe_interest*mm_per_px, color='k', ls='-', alpha=0.5)

ax.set_ylim()
ax.set_ylabel(r'$x$ [mm]')
ax.set_xticks(np.arange(4))
ax.set_xlabel(r'$z$ [mm]')
plt.tight_layout()


# <codecell>

def singlepeak_gauss(z_, position, width):
    # modelisation = gaussienne
    return utility.gaussian_unnormalized(z_, position, width)
    # # modelisation = doubletanch
    # return (np.tanh(z_-position-width/2) - np.tanh(z_-position+width/2) + 1)/2

def signal_bridge(z_, bckgnd_light, bridge_centre, depth, peak_width, peak_spacing):
    return bckgnd_light - depth*(singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) + singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width))

def signal_uneven_bridge(z_, bckgnd_light, bridge_centre, depth_1, depth_2, peak_width, peak_spacing):
    return bckgnd_light - depth_1*singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) - depth_2*singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width)

def find_uneven_bridge_centre(z_, l_, peak_width_0=30, peak_depth_0=90, peak_spacing_0 = 60, peak_spacing_max = 100, peak_spacing_min = 40):
    # brutal estimates

    l_ = savgol_filter(l_, peak_width_0, 2)

    peak1_z = z_[np.argmin(l_)]

    zone_findpeak2 = (z_ < peak1_z + peak_spacing_max + 5) * (z_ > peak1_z - peak_spacing_max + 5)
    z_peak2 = z[zone_findpeak2]
    l_peak2 = l_[zone_findpeak2]
    l_peak2 += peak_depth_0 * singlepeak_gauss(z_peak2, peak1_z, peak_width_0)

    peak2_z = z_peak2[np.argmin(l_peak2)]

    # minz, maxz = -np.inf, np.inf
    minz, maxz = min(peak1_z, peak2_z) - peak_width_0, max(peak1_z, peak2_z) + peak_width_0

    zone_fit = (z_ < maxz) * (z_ > minz)
    zfit = z_[zone_fit]
    lfit = l_[zone_fit]


    bckgnd_light = lfit.max()
    depth = lfit.max() - lfit.min()
    # p0 = (bckgnd_light, (peak1_z+peak2_z)/2, depth, peak_width_0, peak_min_spacing_0)
    # popt_bipeak, pcov = curve_fit(signal_bridge, zfit, lfit, p0=p0, sigma=None)
    p0 = (bckgnd_light, (peak1_z+peak2_z)/2, depth, depth, peak_width_0, peak_spacing_0)
    bounds = ([0, zfit.min(), 0, 0, 0, peak_spacing_min], [255, zfit.max(), 255, 255, z_.max(), peak_spacing_max])
    popt_bipeak, pcov = curve_fit(signal_uneven_bridge, zfit, lfit, p0=p0, sigma=None, bounds=bounds)

    centre = popt_bipeak[1]
    return centre



# <markdowncell>

# ##  # F# i# l# m#  # h# e# i# g# h# t


# <codecell>

data = {}

for x_probe in interesting_probes:
    data[x_probe] = {}

for x_probe in interesting_probes:
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
                    z_interf_min = 412
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
                    z_interf_min = 125
                    z_interf_max = 309
                if x_probe==1270:
                    z_interf_min = 162
                    z_interf_max = 336
                if x_probe==1950:
                    z_interf_min = 207
                    z_interf_max = 347
            required_prominence = 3
            required_distance = 2



    zone_interf = (z >= z_interf_min) * (z <= z_interf_max)
    z_interf = z[zone_interf]
    l_interf = l_span[zone_interf]
    
    
    # Find the peaks (grossier)
    prominence = required_prominence
    distance = required_distance
    maxs = find_peaks(l_interf, prominence = prominence, distance=distance)[0]
    mins = find_peaks(255 - l_interf, prominence = prominence, distance=distance)[0]
    # add the min and max
    if maxs.min() < mins.min(): # if the first peak is a max, the first point is a min
        mins = np.concatenate(([0], mins))
    else:
        maxs = np.concatenate(([0], maxs))
    if maxs.max() > mins.max(): # if the last peak is a max, the last point is a min
        mins = np.concatenate((mins, [len(l_interf)-1]))
    else:
        maxs = np.concatenate((maxs, [len(l_interf)-1]))
    z_maxs = z_interf[maxs]
    z_mins = z_interf[mins]
    # if dataset=='Nalight_cleanplate_20240708' and acquisition=='1Hz_start' and n_frame_ref == 1345 and x_probe == 1800:
    #     z_mins = np.concatenate(([444], z_mins))
    
    # find the central peak
    z_steps = np.concatenate((z_maxs, z_mins))
    z_steps.sort()
    zcentre_verycoarse = z_steps[np.argmax(z_steps[2:] - z_steps[:-2]) + 1]
    z_steps_beforecentre = z_steps[np.argmax(z_steps[2:]-z_steps[:-2])]
    z_steps_aftercentre = z_steps[np.argmax(z_steps[2:]-z_steps[:-2])+2]
    zcentre_coarse = (z_steps_beforecentre + z_steps_aftercentre) / 2


    # estimated brutally the phase shifts
    p_steps = (2 * (z_steps <= zcentre_verycoarse).astype(int) - 1).cumsum() - 1
    p_steps -= p_steps.min() # au minimum on a p = 0
    phi_steps = p_steps * np.pi
    
    ### HILBERT PHASE ESTIMATION FOR FANCY BITCHES
    
    # find the max and min lines, to estimated the 0-phase line
    l_maxs_cs = np.poly1d(np.polyfit(z_interf[maxs], l_interf[maxs], 1))
    if len(maxs) > 5:
        l_maxs_cs = make_smoothing_spline(z_interf[maxs], l_interf[maxs], lam=None)
    l_mins_cs = np.poly1d(np.polyfit(z_interf[mins], l_interf[mins], 1))
    if len(maxs) > 5:
        l_mins_cs = make_smoothing_spline(z_interf[mins], l_interf[mins], lam=None)
    l_offset = (l_maxs_cs(z_interf) + l_mins_cs(z_interf))/2
    # substract the midline
    l_interf_clean = l_interf - l_offset
    
    # we want to ain a bit in resolution especially on the edges where the extrema can be very close
    z_hilbert = np.linspace(z_interf.min(), z_interf.max(), len(z_interf) + (len(z_interf)-1)*2, endpoint=True)
    l_hilbert_raw = np.interp(z_hilbert, z_interf, l_interf_clean)
    
    # we smooth everything a bit to ease the phase reconstruction process via Hilbert transform
    lhibert_spline = make_smoothing_spline(z_interf, l_interf_clean, lam=None)
    l_hilbert_smoothed = lhibert_spline(z_hilbert)
    
    ## Now we use the hilbert transform to do some **magic**
    
    # analytic_signal = hilbert(l_hilbert_smoothed) # this is brutal (but sometimes works well, just not on film detection. it avoid the phase bad definition if at the end we are not on a min / max
    analytic_signal = hilbert(np.concatenate((l_hilbert_smoothed, l_hilbert_smoothed[::-1])))[:len(l_hilbert_smoothed)] # We use a small symmetrization trick here because the hilbert thinks everything in it has to be periodic smh
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase_wrapped = np.angle(analytic_signal)
    
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

    # Scale the height
    h = lambd * phi_hilbert / (4 * np.pi)
    h_steps = lambd * phi_steps / (4*np.pi)
    
    centre_ref = find_uneven_bridge_centre(z, l_span)
    data[x_probe]['centre_ref_px'] = centre_ref
    
    # Scale the width
    Z = (z_hilbert-centre_ref) * um_per_px
    Z_steps = (z_steps-centre_ref) * um_per_px
    Zfilm_centre = (zcentre_better-centre_ref) * um_per_px


    data[x_probe]['Zfilm_centre_px'] = zcentre_better-centre_ref
    data[x_probe]['Zfilm_px'] = z_hilbert-centre_ref
    data[x_probe]['Zfilm_steps_px'] = z_steps-centre_ref
    
    data[x_probe]['Zfilm_centre'] = Zfilm_centre
    data[x_probe]['Zfilm'] = Z
    data[x_probe]['Hfilm'] = h
    data[x_probe]['Zfilm_steps'] = Z_steps
    data[x_probe]['Hfilm_steps'] = h_steps



# <codecell>

probecolors = plt.cm.rainbow(np.linspace(0, 1, len(interesting_probes)))


# <codecell>

fig, axes = plt.subplots(2,1)

ax = axes[0]
ax2 = ax.twinx()
for i_probe, x_probe in enumerate(interesting_probes):
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe}')
    ax.axvline(datapts['centre_ref_px'], color=color, ls='--')

    z = np.arange(height)
    l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)
    ax.plot(z, l_span, c=color, alpha=.3, label=f'xprobe = {x_probe}')

    ax2.plot(datapts['Zfilm_px'] + datapts['centre_ref_px'], datapts['Hfilm'], c=color)
    
#     ax.plot(datapts['Zfilm'], datapts['Hfilm'], c=color)
#     ax.scatter(datapts['Zfilm_steps'], datapts['Hfilm_steps'], s=50, ec=color, fc='w')
# 
# for i in range(0, int(ax.get_ylim()[1]/h0)+1):
#     plt.axhline(i * h0, c='k', alpha=.1)

ax.legend()
ax.set_xlabel(r'$z$ [um]')
# ax.set_ylabel(r'Height $h$ [um]')

ax = axes[1]
ax2 = ax.twinx()
for i_probe, x_probe in enumerate(interesting_probes):
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe}')
    ax.axvline(datapts['centre_ref_px'] - datapts['centre_ref_px'], color=color, ls='--')

    z = np.arange(height)
    l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)
    ax.plot(z - datapts['centre_ref_px'], l_span, c=color, alpha=.3, label=f'xprobe = {x_probe}')

    ax2.plot(datapts['Zfilm_px'], datapts['Hfilm'], c=color)
#     ax.plot(datapts['Zfilm'], datapts['Hfilm'], c=color)
#     ax.scatter(datapts['Zfilm_steps'], datapts['Hfilm_steps'], s=50, ec=color, fc='w')
# 
# for i in range(0, int(ax.get_ylim()[1]/h0)+1):
#     plt.axhline(i * h0, c='k', alpha=.1)

ax.legend()
ax.set_xlabel(r'$z$ [um]')
# ax.set_ylabel(r'Height $h$ [um]')


# <codecell>

fig, ax = plt.subplots(1,1)

for i_probe, x_probe in enumerate(interesting_probes):
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe}')
    ax.plot(datapts['Zfilm'], datapts['Hfilm'], c=color)
    ax.scatter(datapts['Zfilm_steps'], datapts['Hfilm_steps'], s=50, ec=color, fc='w')

for i in range(0, int(ax.get_ylim()[1]/h0)+1):
    plt.axhline(i * h0, c='k', alpha=.1)

ax.legend()
ax.set_xlabel(r'$z$ [um]')
ax.set_ylabel(r'Height $h$ [um]')
# ax.set_aspect('equal')


# <codecell>

fig, ax = plt.subplots(1,1)

for i_probe, x_probe in enumerate(interesting_probes):
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe}')
    ax.plot(datapts['Zfilm'], datapts['Hfilm'], c=color)
    ax.scatter(datapts['Zfilm_steps'], datapts['Hfilm_steps'], s=50, ec=color, fc='w')

for i in range(0, int(ax.get_ylim()[1]/h0)+1):
    plt.axhline(i * h0, c='k', alpha=.1)

ax.legend()
ax.set_xlabel(r'$z$ [um]')
ax.set_ylabel(r'Height $h$ [um]')
# ax.set_aspect('equal')


# <codecell>

fig, axes = plt.subplots(2, 2, figsize = (130 * mm_per_in, 130 * mm_per_in))
gs = axes[0,0].get_gridspec()
for ax in axes[:, 0]:
    ax.remove()
ax = fig.add_subplot(gs[:, 0])
ax.imshow(img[::-1, :].T, extent=[-mm_per_px/2, (img.shape[0]+1/2)*mm_per_px, (img.shape[1]+1/2)*mm_per_px, -mm_per_px/2], origin='upper', aspect='equal',
          cmap=cmap_Na, vmin=vmin, vmax=vmax)
for x_probe_interest in interesting_probes:
    ax.axhspan((x_probe_interest-probespan)*mm_per_px, (x_probe_interest+probespan)*mm_per_px, color='k', ls='', alpha=0.5)
    # ax.axhline(x_probe_interest*mm_per_px, color='k', ls='-', alpha=0.5)

# ax.set_ylim()
ax.set_ylabel(r'$x$ [mm]')
ax.set_xticks(np.arange(4))
ax.set_xlabel(r'$z$ [mm]')

ax = axes[0, 1]
for i_probe, x_probe in enumerate(interesting_probes):
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe}')
    ax.plot(-datapts['Zfilm'], datapts['Hfilm'], c=color, lw=1)
    ax.scatter(-datapts['Zfilm_steps'], datapts['Hfilm_steps'], s=20, ec=color, fc='w')

ax.legend()
ax.set_xlabel(r'$z - z_c$ [um]')
ax.set_ylabel(r'Film height $h$ [um]')

plt.tight_layout()


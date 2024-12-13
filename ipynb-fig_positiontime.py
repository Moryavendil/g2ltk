# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from tools.utility import activate_saveplot
%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
from scipy.signal import hilbert

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
acquisition = '1Hz_start'
acquisition_path = os.path.join(dataset_path, acquisition)


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
lambda_Na_void = 0.589 # in um
gamma = 14
bsur2 = 600/2 # in mm

lambd = lambda_Na_void/refractive_index # in um
h0 = lambd * np.pi / (2 * np.pi)/2 # h0 = lambd/4 in um

from matplotlib.colors import LinearSegmentedColormap
cmap_Na = LinearSegmentedColormap.from_list("cmap_Na", ['black', "xkcd:bright yellow"])

vmin_Na = np.percentile(frames, 1)
vmax_Na = np.percentile(frames, 99)


# <codecell>

n_frame_ref = 1345
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

probecolors = plt.cm.rainbow(np.linspace(0, 1, len(interesting_probes)))[::-1]

img = frame_ref[ymin:ymax,:]

i_probe = 2
x_probe = interesting_probes[i_probe]


# <codecell>

from tools import fringeswork

# Luminosity signal
z = np.arange(img.shape[0])
l_centre = img[:, x_probe]
l_span = img[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)

centre_ref_px = fringeswork.find_uneven_bridge_centre(z[::-1], l_span)


# <codecell>

fig, axes = plt.subplots(2, 2, figsize = (utility.genfig.figw['wide'], utility.genfig.figw['wide']/1.618))
gs = axes[0,1].get_gridspec()
for ax in axes[:, 1]:
    ax.remove()
ax = fig.add_subplot(gs[:, 1])

pcolor = 'lightgreen'

ax = axes[0,0]
ax.imshow(img[::-1, :].T, extent=[-mm_per_px/2, (img.shape[0]+1/2)*mm_per_px, (img.shape[1]+1/2)*mm_per_px, -mm_per_px/2], origin='upper', aspect='auto',
          cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na)

ax.axhspan((x_probe-probespan)*mm_per_px, (x_probe+probespan)*mm_per_px, color=pcolor, ls='', alpha=0.75)
    # ax.axhline(x_probe_interest*mm_per_px, color='k', ls='-', alpha=0.5)
# ax.plot([centre_ref_px*mm_per_px]*2, [(x_probe-probespan)*mm_per_px, (x_probe+probespan)*mm_per_px], color=probecolors[i_probe], lw=2, ls='--')
ax.scatter([centre_ref_px*mm_per_px], [x_probe*mm_per_px], fc='r', ec='k', lw=1, ls='-')


# ax.set_ylim(1.2, 0) # i probe 0
ax.set_ylim(x_probe*mm_per_px+2, x_probe*mm_per_px-2) # i probe 2
ax.set_ylabel(r'$x$ [mm]')
ax.set_xticks(np.arange(4))
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')

ax = axes[1,0]
ax.plot(z[::-1] * mm_per_px, l_span, lw=1, ls='-', color=pcolor)
ax.axvline((centre_ref_px+4)*mm_per_px, color='k', lw=2, ls='--')
ax.axvline((centre_ref_px-4)*mm_per_px, color='k', lw=2, ls='--')
ax.axvline(centre_ref_px*mm_per_px, color='r', lw=2, ls='--')

ax.set_ylim(120, 230)
ax.set_ylabel(r'Luminosity [0-255]')
ax.set_xticks(np.arange(4))
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')

utility.tighten_graph()


# <codecell>

# def singlepeak_gauss(z_, position, width):
#     # modelisation = gaussienne
#     return utility.gaussian_unnormalized(z_, position, width)
#     # # modelisation = doubletanch
#     # return (np.tanh(z_-position-width/2) - np.tanh(z_-position+width/2) + 1)/2
# 
# def signal_bridge(z_, bckgnd_light, bridge_centre, depth, peak_width, peak_spacing):
#     return bckgnd_light - depth*(singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) + singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width))
# 
# def signal_uneven_bridge(z_, bckgnd_light, bridge_centre, depth_1, depth_2, peak_width, peak_spacing):
#     return bckgnd_light - depth_1*singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) - depth_2*singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width)
# 
# def find_uneven_bridge_centre(z_, l_, peak_width_0=30, peak_depth_0=90, peak_spacing_0 = 60, peak_spacing_max = 100, peak_spacing_min = 40):
#     # brutal estimates
# 
#     l_ = savgol_filter(l_, peak_width_0, 2)
# 
#     peak1_z = z_[np.argmin(l_)]
# 
#     zone_findpeak2 = (z_ < peak1_z + peak_spacing_max + 5) * (z_ > peak1_z - peak_spacing_max + 5)
#     z_peak2 = z[zone_findpeak2]
#     l_peak2 = l_[zone_findpeak2]
#     l_peak2 += peak_depth_0 * singlepeak_gauss(z_peak2, peak1_z, peak_width_0)
# 
#     peak2_z = z_peak2[np.argmin(l_peak2)]
# 
#     # minz, maxz = -np.inf, np.inf
#     minz, maxz = min(peak1_z, peak2_z) - peak_width_0, max(peak1_z, peak2_z) + peak_width_0
# 
#     zone_fit = (z_ < maxz) * (z_ > minz)
#     zfit = z_[zone_fit]
#     lfit = l_[zone_fit]
# 
# 
#     bckgnd_light = lfit.max()
#     depth = lfit.max() - lfit.min()
#     # p0 = (bckgnd_light, (peak1_z+peak2_z)/2, depth, peak_width_0, peak_min_spacing_0)
#     # popt_bipeak, pcov = curve_fit(signal_bridge, zfit, lfit, p0=p0, sigma=None)
#     p0 = (bckgnd_light, (peak1_z+peak2_z)/2, depth, depth, peak_width_0, peak_spacing_0)
#     bounds = ([0, zfit.min(), 0, 0, 0, peak_spacing_min], [255, zfit.max(), 255, 255, z_.max(), peak_spacing_max])
#     popt_bipeak, pcov = curve_fit(signal_uneven_bridge, zfit, lfit, p0=p0, sigma=None, bounds=bounds)
# 
#     centre = popt_bipeak[1]
#     return centre



# <markdowncell>

# ##  # R# i# v# u# l# e# t#  # s# p# e# e# d


# <codecell>

smooth_length  =None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        smooth_length = 9
        if n_frame_ref == 1345:
            n_frame_start = 1319 - 5 - 20
            n_frame_stop = 1345 + 5 + 25 + 20 + 5
        elif n_frame_ref == 1367:
            n_frame_start = 1346 - 5
            n_frame_stop = 1367 + 5
    if acquisition=='10Hz_decal':
        smooth_length = None
        period_fn = 5
        n_frame_start = n_frame_ref - period_fn*10
        n_frame_stop = n_frame_ref + period_fn*10

i_frame_start = n_frame_start - framenumbers[0]
i_frame_stop = n_frame_stop - framenumbers[0]


# <codecell>

from scipy.optimize import curve_fit

if True:
    print(f'{i_probe+1}/{len(interesting_probes)} (x_probe = {x_probe})')
    slice = frames[i_frame_start:i_frame_stop, ymin:ymax, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)
    
    t_frames = np.arange(n_frame_start, n_frame_stop)
    
    i_frame_start = n_frame_start - framenumbers[0]
    i_frame_stop = n_frame_stop - framenumbers[0]
    
    pos_raw = np.array([fringeswork.find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])
    
    # Raw values
    t_vel = t_frames[1:-1]
    vel_raw = (pos_raw[2:] - pos_raw[:-2]) / 2
    
    # smoothing
    pos_smoothed = savgol_filter(pos_raw, smooth_length, 2) if smooth_length is not None else pos_raw
    vel_smoothed = (pos_smoothed[2:] - pos_smoothed[:-2]) / (2) if smooth_length is not None else vel_raw
    
    # cubic interpolation on smoothed things
    pos_cs = CubicSpline(t_frames, pos_smoothed)
    # vel_cs = pos_cs.derivative()
    # 
    # ### FIT SINUS
    # 
    # p0 = [pos_raw.max()-pos_raw.min(), 1/5, 0, 275, 1e-4]
    # 
    # popt, _ = curve_fit(sinfn, t_frames, pos_raw, p0=p0)
    # 
    # 
    # tim = np.linspace(t_frames.min(), t_frames.max(), len(t_frames) + 100*(len(t_frames)-1), endpoint=True)
    # pos, vel = pos_cs(tim), vel_cs(tim)
    # # pos, vel = sinfn(tim, *popt), dersinfn(tim, *popt)
    # spd = np.abs(vel)
    # 
    # 
    # time_interest = n_frame_ref
    # time_min = n_frame_ref - period_fn/2
    # zone_period_interp = (tim <= time_interest) * (tim >= time_min)
    # zone_period_vel = (t_vel <= time_interest) * (t_vel >= time_min)
    # 
    # 
    # zmaxvel = pos[zone_period_interp][np.argmax(np.abs(vel[zone_period_interp]))]
    # 
    # zmaxvel_px = zmaxvel
    # if data[x_probe].get('centre_ref_px', None) is None:
    #     z = np.arange(height)
    #     l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)
    #     centre_ref_px = fringeswork.find_uneven_bridge_centre(z, l_span)

t = (t_frames - n_frame_ref) / acquisition_frequency


# <codecell>

fig, axes = plt.subplots(2, 2, figsize = (utility.genfig.figw['wide'], utility.genfig.figw['wide']/1.618))
gs = axes[0,1].get_gridspec()
for ax in axes[:, 1]:
    ax.remove()
ax = fig.add_subplot(gs[:, 1])

ax.imshow(slice[:, ::-1], aspect='auto', cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na, extent=utility.correct_extent(z*mm_per_px, t))
ax.plot((z.max()-pos_smoothed)*mm_per_px, t, color=probecolors[i_probe], lw=2, ls='-')
ax.set_ylim(-1, 1)
ax.set_ylabel(r'$t$ [s]')
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

utility.tighten_graph()


# <codecell>

utility.activate_saveplot()


# <codecell>

fig, axes = plt.subplots(2, 2, figsize = (utility.genfig.figw['wide'], utility.genfig.figw['wide']/1.618))
gs = axes[0,1].get_gridspec()
for ax in axes[:, 1]:
    ax.remove()
ax = fig.add_subplot(gs[:, 1])
ax.imshow(slice[:, ::-1], aspect='auto', cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na, extent=utility.correct_extent(z*mm_per_px, t))
ax.axhspan(-1/acquisition_frequency/2/2, 1/acquisition_frequency/2/2, color=pcolor, ls='', alpha=0.75)
ax.plot((z.max()-pos_smoothed)*mm_per_px, t, color='r', lw=2, ls='-')
ax.scatter((z.max()-pos_raw)*mm_per_px, t, fc='r', ec='k', lw=1, ls='-', zorder=4)
ax.set_ylim(1, -.245)
ax.set_ylabel(r'$t$ [s]')
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


pcolor = 'lightgreen'

ax = axes[0,0]
ax.imshow(img[::-1, :].T, extent=[-mm_per_px/2, (img.shape[0]+1/2)*mm_per_px, (img.shape[1]+1/2)*mm_per_px, -mm_per_px/2], origin='upper', aspect='auto',
          cmap=cmap_Na, vmin=vmin_Na, vmax=vmax_Na)

ax.axhspan((x_probe-probespan)*mm_per_px, (x_probe+probespan)*mm_per_px, color=pcolor, ls='', alpha=0.75)
# ax.axhline(x_probe_interest*mm_per_px, color='k', ls='-', alpha=0.5)
# ax.plot([centre_ref_px*mm_per_px]*2, [(x_probe-probespan)*mm_per_px, (x_probe+probespan)*mm_per_px], color=probecolors[i_probe], lw=2, ls='--')
ax.scatter([centre_ref_px*mm_per_px], [x_probe*mm_per_px], fc='r', ec='k', lw=1, ls='-')


ax.set_ylim(x_probe*mm_per_px+2, x_probe*mm_per_px-2) # iegneral
# ax.set_ylim(1.2, 0) # i probe 0
ax.set_ylim(4, 7) # i probe 2
ax.set_ylabel(r'$x$ [mm]')
ax.set_xticks(np.arange(4))
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')

ax = axes[1,0]
ax.plot(z[::-1] * mm_per_px, l_span, lw=1, ls='-', color=pcolor)
ax.axvline((centre_ref_px+4)*mm_per_px, color='k', lw=2, ls='--')
ax.axvline((centre_ref_px-4)*mm_per_px, color='k', lw=2, ls='--')
ax.axvline(centre_ref_px*mm_per_px, color='r', lw=2, ls='--')

ax.set_ylim(120, 230)
ax.set_ylabel(r'Luminosity [0-255]')
ax.set_xticks(np.arange(4))
ax.set_xlim(0, 3)
ax.set_xlabel(r'$z$ [mm]')

utility.tighten_graph(w_pad=.5)


# <codecell>

# utility.save_graphe('profile_and_speed')


# <codecell>




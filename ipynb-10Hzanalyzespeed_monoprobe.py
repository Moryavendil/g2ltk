# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['pgf.texsystem'] = 'pdflatex'
# plt.rcParams.update({'font.family': 'serif', 'font.size': 20,
#                      'figure.titlesize' : 20,
#                      'axes.labelsize': 20,'axes.titlesize': 20,
#                      'legend.fontsize': 20, 'legend.handlelength': 2})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline

from tools import datareading, utility


# <codecell>

# Dataset selection
dataset = 'Nalight_cleanplate_20240708'
dataset_path = '../' + dataset
print('Available acquisitions:', datareading.find_available_videos(dataset_path))


# <codecell>

# Acquisition selection
acquisition = '10Hz_decal'
# acquisition = '1Hz_strong'
# acquisition = '2400mHz_start_vrai'
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


# <codecell>

n_frame_ref = 1600
# n_frame_ref = 1609
# n_frame_ref = 1611
n_frame_ref = 1660

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
        x_probe = 400
        probespan = 11


# <codecell>

i_frame_ref = n_frame_ref - framenumbers[0]

z = np.arange(height)
frame_ref = frames[i_frame_ref]


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(frame_ref, origin='lower', aspect='auto')
ax.axvline(x_probe, color='k')
ax.axvspan(x_probe - probespan, x_probe + probespan, color='k', alpha=.1)
if interesting_probes is not None:
    for x_probe_interest in interesting_probes:
        ax.axvline(x_probe_interest, color='k', ls='--', alpha=0.3)
        ax.axvspan(x_probe_interest - probespan, x_probe_interest + probespan, color='k', alpha=.01)
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')


# <markdowncell>

# Measure speed
# ===


# <markdowncell>

# ## Measure rivulet position


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

local_only = False
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_strong':
        local_only = True
    elif acquisition=='2400mHz_start_vrai':
        local_only = True


# <codecell>

z = np.arange(height)
l_centre = frame_ref[:, x_probe]
l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)

l_ = l_span
z_ = z

# brutal estimates
peak_width_0 = 30
peak_depth_0 = 90
peak_spacing_0 = 60
peak_spacing_min = 40
peak_spacing_max = 100

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


# <codecell>

fig, axes = plt.subplots(2,1)
fig.suptitle('Rivulet finding on the reference frame')
ax = axes[0]
ax.plot(z, l_centre, color='k', ls='--', alpha=.3, label=fr'At centre $x = {x_probe}$')
ax.plot(z, l_span, color='k', alpha=.3, label=fr'Mean $x = {x_probe} \pm {probespan}$')
ax.plot(z_, l_, color='k', lw=2, label=fr'minus main peak (to find smaller one)')

ax.plot(z_peak2, l_peak2, color='b', lw=2, label=fr'filtered')

ax.axvline(peak1_z, c='g', ls='--', alpha=.5, label='2 highest peak')
ax.axvline(peak2_z, c='g', ls='--', alpha=.5)
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity [0-255]')
ax.legend()

ax = axes[1]
ax.plot(z_, l_, color='k', lw=1, alpha=.3, label=fr'filtered')
ax.plot(zfit, lfit, color='k', lw=2, label=fr'Fitted signal')

ax.plot(zfit, signal_uneven_bridge(zfit, *p0), c='r', ls='--', alpha=.8, label='initial estimate')
ax.plot(zfit, signal_uneven_bridge(zfit, *popt_bipeak), c='r', label='best fit')

ax.axvline(centre, c='m', ls='--', label='center')
ax.axvline(find_uneven_bridge_centre(z, l_span), c='m', ls=':', label='center')
# 
ax.set_xlabel('z [px]')
ax.set_ylabel('luminosity [0-255]')
ax.legend()


# <markdowncell>

# ## Measure speed for N cycle
# 
# And make position-speed correspondance


# <markdowncell>

# ## ##  # S# h# o# r# t#  # t# i# m# e# s


# <codecell>

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='10Hz_decal':
        n_frame_start = 1550
        n_frame_stop = 1750

t_frames = np.arange(n_frame_start, n_frame_stop)

i_frame_start = n_frame_start - framenumbers[0]
i_frame_stop = n_frame_stop - framenumbers[0]
slice = frames[i_frame_start:i_frame_stop, :, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)

pos_raw = np.array([find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(slice, origin='lower', aspect='auto', extent=[-0.5, height+0.5, t_frames.min()-0.5, t_frames.max()+0.5])
ax.plot(pos_raw, t_frames, c='w', ls='-', lw=1, marker='o', mfc='k', mec='w')

ax.set_xlabel('z [px]')
ax.set_ylabel('time [frames]')


# <codecell>

# Raw values
t_vel = t_frames[1:-1]
vel_raw = (pos_raw[2:] - pos_raw[:-2]) / (2)

# smoothing
smooth_length = None
pos_smoothed = savgol_filter(pos_raw, 9, 2) if smooth_length is not None else pos_raw
vel_smoothed = (pos_smoothed[2:] - pos_smoothed[:-2]) / (2) if smooth_length is not None else vel_raw

# cubic interpolation on smoothed things
ttest = np.linspace(t_frames.min(), t_frames.max(), 1000)
pos_cs = CubicSpline(t_frames, pos_smoothed)
vel_cs = CubicSpline(t_vel, vel_smoothed)

t_roots_vel = utility.find_roots(ttest, vel_cs(ttest))


# <codecell>

fig, axes = plt.subplots(2,1, sharex=True)

ax = axes[0]
ax.scatter(t_frames, pos_raw, color='r', s = 100, alpha=.5, label='Position')
ax.scatter(t_frames, pos_smoothed, color='k', s = 20, label=fr'Position (smoothed)')
ax.plot(ttest, pos_cs(ttest), c='k', label=fr'Position (smoothed, interpolated)')

for troot in t_roots_vel:
    ax.axvline(troot, c='k', alpha=.3, ls='--')
    ax.axhline(pos_cs(troot), c='k', alpha=.3, ls='--')

ax.set_ylabel(r'Position of centre $z$ [px]')
ax.set_xlabel('t [frames]')
ax.legend()

ax = axes[1]
ax.scatter(t_vel, vel_raw, color='r', s = 100, alpha=.5, label=fr'Velocity')
ax.scatter(t_vel, vel_smoothed, color='k', s = 20, label=fr'Velocity (smoothed)')
ax.plot(ttest, vel_cs(ttest), c='k', label=fr'Velocity (smoothed, interpolated)')

ax.axhline(0, c='k', alpha=.3, ls='--')
for troot in t_roots_vel:
    ax.axvline(troot, c='k', alpha=.3, ls='--')

ax.set_ylim(-np.abs(vel_raw).max()*1.15, np.abs(vel_raw).max()*1.15)
ax.set_xlabel('t [frames]')
ax.set_ylabel('Speed_0padded [px/frames]')
ax.legend()


# <codecell>

if len(t_roots_vel) != 2:
    print('AAAAAAAAAAAAH ERROR AAAAAAAAAAAAAAAAAAAH IL Y A PLUS OU MOINS DE 2 ZEROS')
tmin, tmax = t_roots_vel
zext1, zext2 = pos_cs([tmin, tmax])

pos_spd = pos_smoothed[1:-1]
pos_for_spdinterp = np.concatenate(([zext1], pos_spd[(t_vel > tmin) * (t_vel < tmax)], [zext2]))
vel_raw_for_spdinterp = np.concatenate(([0], vel_raw[(t_vel > tmin) * (t_vel < tmax)], [0]))
vel_smoothed_for_spdinterp = np.concatenate(([0], vel_smoothed[(t_vel > tmin) * (t_vel < tmax)], [0]))

index = pos_for_spdinterp.argsort()
pos_for_spdinterp = pos_for_spdinterp[index]
vel_raw_for_spdinterp = vel_raw_for_spdinterp[index]
vel_smoothed_for_spdinterp = vel_smoothed_for_spdinterp[index]


# <codecell>

ztest = np.linspace(pos_for_spdinterp.min()-10, pos_for_spdinterp.max()+10, 2000)

def Speed_cs_px(z):
    spd_cs = CubicSpline(pos_for_spdinterp, vel_smoothed_for_spdinterp)
    spd_val = np.abs(spd_cs(z))
    spd_val[z < pos_for_spdinterp.min()] *= 0
    spd_val[z > pos_for_spdinterp.max()] *= 0
    return spd_val

def Speed_mss_px(z):
    spd_mss = make_smoothing_spline(pos_for_spdinterp, vel_smoothed_for_spdinterp, lam=1)
    spd_val = np.abs(spd_mss(z))
    spd_val[z < pos_for_spdinterp.min()] *= 0
    spd_val[z > pos_for_spdinterp.max()] *= 0
    return spd_val

Zcentre = pos_for_spdinterp * um_per_px
Vcentre = np.abs(vel_smoothed_for_spdinterp) * um_per_px * acquisition_frequency

Ztest = np.linspace(Zcentre.min()-10*um_per_px, Zcentre.max()+10*um_per_px, 2000)

def Speed_cs(Z):
    spd_cs = CubicSpline(Zcentre, Vcentre)
    spd_val = np.abs(spd_cs(Z))
    spd_val[Z < Zcentre.min()] *= 0
    spd_val[Z > Zcentre.max()] *= 0
    return spd_val

def Speed_mss(Z):
    spd_mss = make_smoothing_spline(Zcentre, Vcentre, lam=100)
    spd_val = np.abs(spd_mss(Z))
    spd_val[Z < Zcentre.min()] *= 0
    spd_val[Z > Zcentre.max()] *= 0
    return spd_val


# <codecell>

fig, axes = plt.subplots(2,1)
ax = axes[0]
ax.scatter(pos_for_spdinterp, np.abs(vel_raw_for_spdinterp), color='r', s = 100, alpha=.5, label=fr'Speed_0padded')
ax.scatter(pos_for_spdinterp, np.abs(vel_smoothed_for_spdinterp), color='k', s = 20, label=fr'Speed_0padded (smoothed)')
ax.plot(ztest, Speed_mss_px(ztest), c='k', label=fr'Speed_0padded (smoothed, interpolated)')

ax.set_xlabel('Position of centre [px]')
ax.set_ylabel('Speed_0padded [px/frames]')
ax.legend()

ax = axes[1]
ax.scatter(Zcentre, Vcentre, color='k', s = 20, label=fr'Speed_0padded (smoothed)')
ax.plot(Ztest, Speed_mss(Ztest), c='k', label=fr'Speed_0padded (smoothed, interpolated)')

ax.set_xlabel('Position of centre [um]')
ax.set_ylabel('Speed_0padded [um/s]')
ax.legend()


# <markdowncell>

# ## Long time


# <codecell>

method='classic'
threshold_robustmethod = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        if n_frame_ref == 1345:
            n_frame_start = 1319 - 5
            n_frame_stop = 1345 + 5
        elif n_frame_ref == 1367:
            n_frame_start = 1346 - 5
            n_frame_stop = 1367 + 5
    if acquisition=='1Hz_strong':
        if n_frame_ref == 1566:
            n_frame_start = 1544 - 125
            n_frame_stop = 1566 + 25
    if acquisition=='2400mHz_start_vrai':
        n_frame_start = framenumbers[100]
        n_frame_stop = framenumbers[-100]
        method='robust'
        threshold_robustmethod = -30
    if acquisition=='10Hz_decal':
        n_frame_start = 1550
        n_frame_stop = 1750


t_frames = np.arange(n_frame_start, n_frame_stop)

i_frame_start = n_frame_start - framenumbers[0]
i_frame_stop = n_frame_stop - framenumbers[0]
slice = frames[i_frame_start:i_frame_stop, :, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)

if method=='classic':
    pos_raw = np.array([find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])
elif method=='robust':
    slice_medianremoved = slice-np.median(slice, axis=0, keepdims=True)
    big_values_removed = slice_medianremoved.flatten() > threshold_robustmethod
    slice_onlysmall = slice_medianremoved.copy().flatten()
    slice_onlysmall[big_values_removed] *=0
    slice_onlysmall = -np.reshape(slice_onlysmall, slice.shape)
    pos_raw = np.array([np.average(z, weights=slice_onlysmall[i]) for i in range(len(slice))])


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(slice, origin='lower', aspect='auto', extent=[-0.5, height+0.5, t_frames.min()-0.5, t_frames.max()+0.5])
ax.plot(pos_raw, t_frames, color='w', lw=4)
ax.plot(pos_raw, t_frames, color='k', lw=2, marker='o', mfc='k', mec='w')

ax.set_xlabel('z [px]')
ax.set_ylabel('time [frames]')


# <codecell>

print(f'Method used: {method}')
if method == 'robust':
    fig, axes = plt.subplots(2,1)
    ax = axes[0]
    ax.hist(slice_medianremoved.flatten(), bins=100, alpha=.5, label='slice (time median removed)')
    ax.axvline(threshold_robustmethod, ls='--', color='k', label='threshold')
    ax.legend()

    ax = axes[1]
    ax.imshow(slice_onlysmall, origin='lower', aspect='auto', extent=[-0.5, height+0.5, t_frames.min()-0.5, t_frames.max()+0.5])
    ax.plot(pos_raw, t_frames, color='w', lw=4)
    ax.plot(pos_raw, t_frames, color='k', lw=2, marker='o', mfc='k', mec='w')

    ax.set_xlabel('z [px]')
    ax.set_ylabel('time [frames]')



# <codecell>

# Raw values
t_vel = t_frames[1:-1]
vel_raw = (pos_raw[2:] - pos_raw[:-2]) / (2)

# smoothing
smooth_length = None
pos_smoothed = savgol_filter(pos_raw, 9, 2) if smooth_length is not None else pos_raw
vel_smoothed = (pos_smoothed[2:] - pos_smoothed[:-2]) / (2) if smooth_length is not None else vel_raw

# cubic interpolation on smoothed things
ttest = np.linspace(t_frames.min(), t_frames.max(), 10000)
pos_cs = CubicSpline(t_frames, pos_smoothed)
vel_cs = CubicSpline(t_vel, vel_smoothed)

# t_roots_vel = utility.find_roots(ttest, vel_cs(ttest))


# <codecell>

fig, axes = plt.subplots(2,1, sharex=True)

ax = axes[0]
ax.scatter(t_frames, pos_raw, color='r', s = 100, alpha=.5, label='Position')
ax.scatter(t_frames, pos_smoothed, color='k', s = 20, label=fr'Position (smoothed)')
ax.plot(ttest, pos_cs(ttest), c='k', label=fr'Position (smoothed, interpolated)')

# for troot in t_roots_vel:
#     ax.axvline(troot, c='k', alpha=.3, ls='--')
#     ax.axhline(pos_cs(troot), c='k', alpha=.3, ls='--')

ax.set_ylabel(r'Position of centre $z$ [px]')
ax.set_xlabel('t [frames]')
ax.legend()

ax = axes[1]
ax.scatter(t_vel, vel_raw, color='r', s = 100, alpha=.5, label=fr'Velocity')
ax.scatter(t_vel, vel_smoothed, color='k', s = 20, label=fr'Velocity (smoothed)')
ax.plot(ttest, vel_cs(ttest), c='k', label=fr'Velocity (smoothed, interpolated)')
ax.plot(ttest, pos_cs.derivative()(ttest), c='r', label=fr'Velocity from position (smoothed, interpolated)')

ax.axhline(0, c='k', alpha=.3, ls='--')
# for troot in t_roots_vel:
#     ax.axvline(troot, c='k', alpha=.3, ls='--')

ax.set_ylim(-np.abs(vel_raw).max()*1.15, np.abs(vel_raw).max()*1.15)
ax.set_xlabel('t [frames]')
ax.set_ylabel('Speed_0padded [px/frames]')
ax.legend()


# <codecell>

### FIT SINUS
def sinfn(x, A, f, p, y0, B):
    return y0 + B*x + A*np.sin(2*np.pi*f*x + p)
def dersinfn(x, A, f, p, y0, B):
    return B + 2*np.pi*f*A*np.cos(2*np.pi*f*x + p)

p0 = [pos_raw.max()-pos_raw.min(), 1/5, 0, 275, 1e-4]

popt, _ = curve_fit(sinfn, t_frames, pos_raw, p0=p0)



# <codecell>

fig, axes = plt.subplots(1,1, sharex=True, squeeze=False)

ax = axes[0, 0]
ax.scatter(t_frames, pos_raw, color='r', s = 100, alpha=.5, label='Position')
ax.scatter(t_frames, pos_smoothed, color='k', s = 20, label=fr'Position (smoothed)')
ax.plot(ttest, sinfn(ttest, *popt), c='k', label=fr'Position (ssinusoidal fit)')

ax.set_ylabel(r'Position of centre $z$ [px]')
ax.set_xlabel('t [frames]')
ax.legend()


# <codecell>

fig, axes = plt.subplots()
ax = axes
ax.scatter(pos_raw[1:-1], vel_raw, color='r', s = 100, alpha=.5, label=fr'Speed_0padded')
ax.scatter(pos_smoothed[1:-1], vel_smoothed, color='k', s = 20, label=fr'Speed_0padded (smoothed)')
ax.plot(pos_cs(ttest), vel_cs(ttest), c='k', alpha=.8, label=fr'Speed_0padded (smoothed, interpolated)', lw=.5)
ax.plot(sinfn(ttest, *popt), dersinfn(ttest, *popt), c='k', lw=2, label=fr'Speedsinus fit')

ax.set_ylim(-max(np.abs(vel_raw).max(), dersinfn(ttest, *popt).max())*1.15, max(np.abs(vel_raw).max(), dersinfn(ttest, *popt).max())*1.15)
ax.set_xlabel('Position of centre $z_c$ [px]')
ax.set_ylabel('Speed_0padded [px/frames]')
ax.legend()


# <codecell>




# <codecell>




# <codecell>




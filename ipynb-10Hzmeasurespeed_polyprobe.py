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

# ##  # R# i# v# u# l# e# t#  # s# p# e# e# d


# <codecell>

smooth_length  =None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        smooth_length = 9
        if n_frame_ref == 1345:
            n_frame_start = 1319 - 5
            n_frame_stop = 1345 + 5
        elif n_frame_ref == 1367:
            n_frame_start = 1346 - 5
            n_frame_stop = 1367 + 5
    if acquisition=='10Hz_decal':
        smooth_length = None
        period_fn = 5
        n_frame_start = n_frame_ref - period_fn*10
        n_frame_stop = n_frame_ref + period_fn*10


### FIT SINUS
def sinfn(x, A, f, p, y0, B):
    return y0 + B*x + A*np.sin(2*np.pi*f*x + p)
def dersinfn(x, A, f, p, y0, B):
        return B + 2*np.pi*f*A*np.cos(2*np.pi*f*x + p)
    
i_frame_start = n_frame_start - framenumbers[0]
i_frame_stop = n_frame_stop - framenumbers[0]


# <codecell>

data = {}

for x_probe in interesting_probes:
    data[x_probe] = {}

for i_probe, x_probe in enumerate(interesting_probes):
    print(f'{i_probe+1}/{len(interesting_probes)} (x_probe = {x_probe})')
    slice = frames[i_frame_start:i_frame_stop, :, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)
    
    
    t_frames = np.arange(n_frame_start, n_frame_stop)
    
    i_frame_start = n_frame_start - framenumbers[0]
    i_frame_stop = n_frame_stop - framenumbers[0]
    slice = frames[i_frame_start:i_frame_stop, :, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)
    
    pos_raw = np.array([find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])
    
    # Raw values
    t_vel = t_frames[1:-1]
    vel_raw = (pos_raw[2:] - pos_raw[:-2]) / 2
    
    # smoothing
    smooth_length = None
    pos_smoothed = savgol_filter(pos_raw, 9, 2) if smooth_length is not None else pos_raw
    vel_smoothed = (pos_smoothed[2:] - pos_smoothed[:-2]) / (2) if smooth_length is not None else vel_raw
    
    # cubic interpolation on smoothed things
    pos_cs = CubicSpline(t_frames, pos_smoothed)
    vel_cs = pos_cs.derivative()
    
    ### FIT SINUS
    
    p0 = [pos_raw.max()-pos_raw.min(), 1/5, 0, 275, 1e-4]
    
    popt, _ = curve_fit(sinfn, t_frames, pos_raw, p0=p0)
    
    
    tim = np.linspace(t_frames.min(), t_frames.max(), len(t_frames) + 100*(len(t_frames)-1), endpoint=True)
    pos, vel = pos_cs(tim), vel_cs(tim)
    # pos, vel = sinfn(tim, *popt), dersinfn(tim, *popt)
    spd = np.abs(vel)
    
    
    time_interest = n_frame_ref
    time_min = n_frame_ref - period_fn/2
    zone_period_interp = (tim <= time_interest) * (tim >= time_min)
    zone_period_vel = (t_vel <= time_interest) * (t_vel >= time_min)


    zmaxvel = pos[zone_period_interp][np.argmax(np.abs(vel[zone_period_interp]))]
    
    data[x_probe]['zmaxvel_px'] = zmaxvel
    if data[x_probe].get('centre_ref_px', None) is None:
        z = np.arange(height)
        l_span = frame_ref[:, x_probe - probespan:x_probe + probespan + 1].mean(axis=1)
        data[x_probe]['centre_ref_px'] = find_uneven_bridge_centre(z, l_span)
    
    centre_ref_px = data[x_probe]['centre_ref_px']
    
    Zcentre = (pos[zone_period_interp] - centre_ref_px) * um_per_px
    Vcentre = np.abs(vel[zone_period_interp]) * um_per_px * acquisition_frequency
    
    # Ztest = np.linspace(Zcentre.min()-10*um_per_px, Zcentre.max()+10*um_per_px, 500)

    data[x_probe]['Zcentre_mes'] = (pos_raw[1:-1][zone_period_vel] - centre_ref_px) * um_per_px
    data[x_probe]['Vcentre_mes'] = vel_raw[zone_period_vel] * um_per_px * acquisition_frequency
    data[x_probe]['Zcentre_interp'] = (pos[zone_period_interp] - centre_ref_px) * um_per_px
    data[x_probe]['Vcentre_interp'] = np.abs(vel[zone_period_interp]) * um_per_px * acquisition_frequency



# <codecell>

probecolors = plt.cm.rainbow(np.linspace(0, 1, len(interesting_probes)))


# <codecell>

def Speed_0padded(Z, Zcentre_interp, Vcentre_interp):
    spd_val = np.abs(np.interp(Z, Zcentre_interp,Vcentre_interp ))
    spd_val[Z < Zcentre_interp.min()] *= 0
    spd_val[Z > Zcentre_interp.max()] *= 0
    return spd_val



# <codecell>

fig, ax = plt.subplots(1,1)

for i_probe, x_probe in enumerate(interesting_probes):
    print(f'{i_probe+1}/{len(interesting_probes)} (x_probe = {x_probe})')
    datapts = data[x_probe]

    color = probecolors[i_probe]
    ax.plot([], [], c=color, label=f'xprobe = {x_probe} px')

    Ztest = np.linspace(datapts['Zcentre_interp'].min()-50, datapts['Zcentre_interp'].max()+50, 1000)

    ax.scatter(datapts['Zcentre_mes'], datapts['Vcentre_mes'], ec=color, fc='w', s = 50)
    ax.plot(Ztest, Speed_0padded(Ztest, datapts['Zcentre_interp'], datapts['Vcentre_interp']), c=color)


ax.legend()
ax.set_xlabel(r'$z$ [um]')
ax.set_ylabel(r'Speed [um/s]')
ax.set_title(f'{acquisition} ({dataset}) - frame {n_frame_ref}')


# <codecell>




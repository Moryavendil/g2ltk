# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook
import os
import numpy as np
import matplotlib.pyplot as plt

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
dataset = 'Nalight_cleanplate_20240708'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


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
h0 = lambd * np.pi / (2 * np.pi)/2 # in um


# <codecell>

# Acquisition selection
# acquisition = '2400mHz_stop'
acquisition = '1Hz_start'
acquisition_path = os.path.join(dataset_path, acquisition)


# <codecell>

# Parameters definition
framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
roi = None, None, None, None  #start_x, start_y, end_x, end_y

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        framenumbers = np.arange(700, datareading.get_number_of_available_frames(acquisition_path))
        roi = None, 370, None, 800  #start_x, start_y, end_x, end_y
    if acquisition=='2400mHz_stop':
        framenumbers = np.arange(960)
        # roi = None, 370, None, 800  #start_x, start_y, end_x, end_y


# <codecell>

# Data fetching
datareading.describe_acquisition(dataset, acquisition, framenumbers = framenumbers, subregion=roi)
frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape

acquisition_frequency = datareading.get_acquisition_frequency(acquisition_path, unit="Hz")
t = datareading.get_times(acquisition_path, framenumbers=framenumbers, unit='s')


# <codecell>

n_frame_ref = 594

p1 = (0, 135)
p2 = (width-1, 170)

interesting_probes = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        p1 = (0, 135)
        p2 = (width-1, 170)
    if acquisition=='2400mHz_stop':
        p1 = (0, 430)
        p2 = (width-1, 470)

probespan = 5

x_probe = 1024


# <codecell>

i_frame_ref = n_frame_ref - framenumbers[0]

x1, y1 = p1
x2, y2 = p2

z = np.arange(height)
frame_ref = frames[i_frame_ref]


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(frame_ref, origin='lower', aspect='auto')
ax.axvline(x_probe, color='k')
ax.axvspan(x_probe - probespan, x_probe + probespan, color='k', alpha=.1)
ax.plot([x1, x2], [y1, y2], 'ko-', lw=2, mfc='w')
ax.set_xlabel('x [px]')
ax.set_ylabel('z [px]')


# <markdowncell>

# Measure speed
# ===


# <codecell>

from tools import fringeswork


# <codecell>

local_only = False
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_strong':
        
        local_only = True
    if acquisition=='2400mHz_stop':
        local_only = True


# <codecell>

if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        n_frame_start = framenumbers[0]
        n_frame_stop = framenumbers[-1]+1
    if acquisition=='2400mHz_stop':
        n_frame_start = framenumbers[0]
        n_frame_stop = framenumbers[-1]+1

t_frames = np.arange(n_frame_start, n_frame_stop)

i_frame_start = n_frame_start - framenumbers[0]
i_frame_stop = n_frame_stop - framenumbers[0]
slice = frames[i_frame_start:i_frame_stop, :, x_probe - probespan:x_probe + probespan + 1].mean(axis=2)

# pos_raw = np.array([find_symmetric_bridge_centre(z, slice[i_t], local_only=local_only) for i_t in range(len(slice))])
# pos_raw = np.array([find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])
pos_raw = fringeswork.find_riv_pos_raw(slice, z=z, problem_threshold=150)


# <codecell>

plt.figure()
ax = plt.gca()
ax.imshow(slice, origin='lower', aspect='auto', extent=[-0.5, height+0.5, t_frames.min()-0.5, t_frames.max()+0.5])
ax.plot(pos_raw, t_frames, c='w', ls='-', lw=1, marker='o', mfc='k', mec='w')

ax.set_xlabel('z [px]')
ax.set_ylabel('time [frames]')


# <codecell>

vel_raw = (pos_raw[2:] - pos_raw[:-2]) / (2)
t_vel = t_frames[1:-1]

pos_smoothed = savgol_filter(pos_raw, 9, 2)
vel_smoothed = (pos_smoothed[2:] - pos_smoothed[:-2]) / (2)

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

ax.set_xlabel('t [frames]')
ax.set_ylabel('Speed_0padded [px/frames]')
ax.legend()


# <codecell>

threshold_pos_min = None
threshold_pos_max = None
if dataset=='Nalight_cleanplate_20240708':
    if acquisition=='1Hz_start':
        threshold_pos_min = 275
        threshold_pos_max = 125
    if acquisition=='2400mHz_stop':
        threshold_pos_min = 550

possible_criterion_up = pos_smoothed > (threshold_pos_min or np.inf)
possible_criterion_down = pos_smoothed < (threshold_pos_max or -np.inf)

frames_glob_up = frames[possible_criterion_up]
tfr_glob_up = t_frames[possible_criterion_up]
pos_glob_up = pos_smoothed[possible_criterion_up]

chunkseparators_up = [i for i in np.arange(len(tfr_glob_up) - 1) if tfr_glob_up[i + 1] - tfr_glob_up[i] > 1]
chunkseparators_up = [(tfr_glob_up[chunkseparator] + tfr_glob_up[chunkseparator + 1]) / 2 for chunkseparator in chunkseparators_up]
# chunks = [possible_criterion * (t_frames <= chunkseparators[0])] + [possible_criterion * (t_frames >= chunkseparators[i_chksep]) * (t_frames <= chunkseparators[i_chksep+1]) for i_chksep in range(len(chunkseparators)-1)] + [possible_criterion * (t_frames >= chunkseparators[-1])]
chunks_up = [(tfr_glob_up <= chunkseparators_up[0])] + [(tfr_glob_up >= chunkseparators_up[i_chksep]) * (tfr_glob_up <= chunkseparators_up[i_chksep + 1]) for i_chksep in range(len(chunkseparators_up) - 1)] + [(tfr_glob_up >= chunkseparators_up[-1])]


frames_glob_down = frames[possible_criterion_down]
tfr_glob_down = t_frames[possible_criterion_down]
pos_glob_down = pos_smoothed[possible_criterion_down]

chunkseparators_down = [i for i in np.arange(len(tfr_glob_down) - 1) if tfr_glob_down[i + 1] - tfr_glob_down[i] > 1]
chunkseparators_down = [(tfr_glob_down[chunkseparator] + tfr_glob_down[chunkseparator + 1]) / 2 for chunkseparator in chunkseparators_down]
# chunks = [possible_criterion * (t_frames <= chunkseparators[0])] + [possible_criterion * (t_frames >= chunkseparators[i_chksep]) * (t_frames <= chunkseparators[i_chksep+1]) for i_chksep in range(len(chunkseparators)-1)] + [possible_criterion * (t_frames >= chunkseparators[-1])]
chunks_down = [(tfr_glob_down <= chunkseparators_down[0])] + [(tfr_glob_down >= chunkseparators_down[i_chksep]) * (tfr_glob_down <= chunkseparators_down[i_chksep + 1]) for i_chksep in range(len(chunkseparators_down) - 1)] + [(tfr_glob_down >= chunkseparators_down[-1])]


# <codecell>

fig, axes = plt.subplots(1,1, sharex=True)

ax = axes
ax.scatter(tfr_glob_up, pos_glob_up, color='k', s = 10, label=fr'Chunks')
for chunkseparator in chunkseparators_up:
    ax.axvline(chunkseparator, c='k', ls='--', alpha=0.3)
for chunkseparator in chunkseparators_down:
    ax.axvline(chunkseparator, c='k', ls=':', alpha=0.3)
for chunk in chunks_up:
    ax.plot(tfr_glob_up[chunk], pos_glob_up[chunk], '-o')
for chunk in chunks_down:
    ax.plot(tfr_glob_down[chunk], pos_glob_down[chunk], '-o')
ax.set_ylabel(r'Position of centre $z$ [px]')
ax.set_xlabel('t [frames]')
ax.legend()


# <codecell>

from scipy.signal import correlate, correlation_lags
from scipy.ndimage import map_coordinates
from scipy.signal import butter, filtfilt
from scipy.stats import linregress


# <codecell>


from tools.fringeswork import findminmaxs, find_cminmax, normalize_for_hilbert


# <codecell>

shiftspeeds_up = []
shiftspeeds_weights_up = []
shiftspeeds_avg_up = np.zeros(len(chunks_up), dtype=float)
shiftspeeds_std_up = np.zeros(len(chunks_up), dtype=float)
movespeeds_avg_up = np.zeros(len(chunks_up), dtype=float)
movespeeds_std_up = np.zeros(len(chunks_up), dtype=float)
slopes_up = np.zeros(len(chunks_up), dtype=float)


p1 = (0, 135)
p2 = (width - 1, 170)
x1, y1 = p1
x2, y2 = p2
dlength = int(np.hypot(x2 - x1, y2 - y1)) + 1
x, y = np.linspace(x1, x2, dlength), np.linspace(y1, y2, dlength)
d = np.hypot(x - x1, y - y1)
fn = lambda x, a:a*x
probespan = 9

for i_chunk, chunk in enumerate(chunks_up):
    print(f'Chunk {i_chunk+1}/{len(chunks_up)}\r', end='')
    chunk_length = len(frames_glob_up[chunk])
    
    l = np.zeros((probespan, chunk_length, len(d)), dtype=float)
    # obtain the data
    for i_t in range(chunk_length):
        for i_probe in np.arange(-probespan//2, probespan//2 + 1):
            l[i_probe][i_t] = map_coordinates(frames_glob_up[chunk][i_t], np.vstack((y+i_probe, x)))
    
    # average
    l = l.mean(axis=0, keepdims=False)
    
    # filter
    for i_t in range(chunk_length):
        # savgol filter has better behaviour near the edges.
        # it doesnt matter since we trim them but anyway
        l[i_t] = savgol_filter(l[i_t], 201, 2)
        # b, a = butter(4, 1/200, 'low')
        # l[i_t] = filtfilt(b, a, l[i_t])
    
    # we remove the mean component to better remove edges
    l -= np.mean(l, axis=1, keepdims=True)

    ### LOOK AT THE 0-PHASE SHIFTS
    # normalize
    l_normalized = np.empty_like(l)
    
    prominence = 3
    
    for i_t in range(chunk_length):
        l_normalized[i_t] = normalize_for_hilbert(l[i_t], prominence=prominence)

    # find the gradient dhdx
    i_gradmes = chunk_length//2
    l_gradmes = l_normalized[i_gradmes]

    analytic_signal = hilbert(np.concatenate((l_gradmes, l_gradmes[::-1])))[:len(l_gradmes)] # We use a small symmetrization trick here because the hilbert thinks everything in it has to be periodic smh
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase_wrapped = np.angle(analytic_signal)
    phase_unwrapped = np.unwrap(instantaneous_phase_wrapped)

    lbd = 1/utility.estimatesignalfrequency(l_gradmes, x=d)
    awayfromedges = (d > d.min() + lbd/8) * (d < d.max() - lbd/8)

    d_forfit = d[awayfromedges] * um_per_px
    h_forfit = phase_unwrapped[awayfromedges] / (2 * np.pi) * lambd /2

    result = linregress(d_forfit, h_forfit)
    dhdx = result.slope

    # find the zeros
    zeros = []
    for i_t in range(chunk_length):
        zeros.append(utility.find_roots(d, l_normalized[i_t]))

    lbd = 1/utility.estimatesignalfrequency(l_normalized[0], x=d)
    try:
        zeros = np.array(zeros)
    except:
        utility.log_debug("AAAAAAH CERTAINS SIGNAUX ONT DES ZEROS QUE D'AUTRE N'ONT PAS")
        zeros = [zero[(zero > d.min() + lbd/8) * (zero < d.max() - lbd/8)] for zero in zeros]
    
        try:
            # print([len(zero) for zero in zeros])
            zeros = np.array(zeros)
        except:
            utility.log_error("AAAAAAH CERTAINS SIGNAUX ONT DES ZEROS QUE D'AUTRE N'ONT PAS")
    # removce the zeros too close to the edges
    if zeros[:, 0].mean() < d.min() + lbd/8:
        zeros = zeros[:, 1:]
    if zeros[:, -1].mean() > d.max() - lbd/8:
        zeros = zeros[:, :-1]
    
    # intercept the zeros
    n_z = zeros.shape[1]
    slopes, intercetps = np.empty(n_z, dtype=float), np.empty(n_z, dtype=float)
    for i_z in range(n_z):
        slopes[i_z], intercetps[i_z] = np.polyfit(np.arange(chunk_length), zeros[:, i_z], 1)
    
    slope_mean = np.mean(slopes)
    slope_spread = np.std(slopes)

    ### LOOK AT THE CORRELATION
    # windowing to make edges match the 0-padding
    l_windowed = l * np.expand_dims( utility.get_window('tukey', len(d)), 0)
    
    lags = np.zeros((chunk_length * (chunk_length - 1))// 2, dtype=float)
    shifts = np.zeros((chunk_length * (chunk_length - 1)) // 2, dtype=float)
    
    pxlags = correlation_lags(l_windowed.shape[1], l_windowed.shape[1], mode='full')
    
    i_0 = 0
    for lag in np.arange(1, chunk_length):
        for i_t in range(chunk_length - lag):
            lags[i_0 + i_t] = lag
            shifts[i_0 + i_t] = utility.find_global_peak(pxlags, correlate(l_windowed[i_t], l_windowed[i_t + lag], mode='full'), peak_category='max')
        i_0 += chunk_length - lag
    
    
    weights = lags / (chunk_length-1)
    
    weights = np.array([lag / (chunk_length-1) for lag in lags])
    # weights[weights < 1/2] *= 0
    weights[weights > 0] = weights[weights > 0] - weights[weights > 0].min()
    weights[weights > 0] = weights[weights > 0] / weights[weights > 0].max()

    # linear regression
    popt, pcov = curve_fit(fn, lags, shifts, p0=[1], sigma=[1/weight if weight > 0 else np.inf for weight in weights])
    shiftspeed_reg = popt[0]
    # the longest lag
    shiftspeed_long = shifts[lags==(chunk_length-1)][0] / (chunk_length-1)
    
    shiftspeed = shifts / lags
    shiftspeed_weights = weights
    
    # just a good ol'mean
    shiftspeed_avg = np.average(shiftspeed, weights=shiftspeed_weights)
    
    shiftspeed_std = np.sqrt(np.average((shiftspeed - shiftspeed_avg) ** 2, weights=shiftspeed_weights))

    shiftspeeds_up.append(shiftspeed)

    movespeeds_avg_up[i_chunk] = -slope_mean
    movespeeds_std_up[i_chunk] = slope_spread
    shiftspeeds_up.append(shiftspeed)
    shiftspeeds_weights_up.append(shiftspeed_weights)
    shiftspeeds_avg_up[i_chunk] = shiftspeed_avg
    shiftspeeds_std_up[i_chunk] = shiftspeed_std
    slopes_up[i_chunk] = dhdx


# <codecell>

shiftspeeds_down = []
shiftspeeds_weights_down = []
shiftspeeds_avg_down = np.zeros(len(chunks_down), dtype=float)
shiftspeeds_std_down = np.zeros(len(chunks_down), dtype=float)
movespeeds_avg_down = np.zeros(len(chunks_down), dtype=float)
movespeeds_std_down = np.zeros(len(chunks_down), dtype=float)
slopes_down = np.zeros(len(chunks_down), dtype=float)

# # up
# p1 = (0, 135)
# p2 = (width - 1, 170)
# up
p1 = (0, 240)
p2 = (width - 1, 280)
x1, y1 = p1
x2, y2 = p2
dlength = int(np.hypot(x2 - x1, y2 - y1)) + 1
x, y = np.linspace(x1, x2, dlength), np.linspace(y1, y2, dlength)
d = np.hypot(x - x1, y - y1)
fn = lambda x, a:a*x
probespan = 9

for i_chunk, chunk in enumerate(chunks_down):
    print(f'Chunk {i_chunk+1}/{len(chunks_down)}\r', end='')
    chunk_length = len(frames_glob_down[chunk])

    l = np.zeros((probespan, chunk_length, len(d)), dtype=float)
    # obtain the data
    for i_t in range(chunk_length):
        for i_probe in np.arange(-probespan//2, probespan//2 + 1):
            l[i_probe][i_t] = map_coordinates(frames_glob_down[chunk][i_t], np.vstack((y+i_probe, x)))

    # average
    l = l.mean(axis=0, keepdims=False)

    # filter
    for i_t in range(chunk_length):
        # savgol filter has better behaviour near the edges.
        # it doesnt matter since we trim them but anyway
        l[i_t] = savgol_filter(l[i_t], 201, 2)
        # b, a = butter(4, 1/200, 'low')
        # l[i_t] = filtfilt(b, a, l[i_t])

    # we remove the mean component to better remove edges
    l -= np.mean(l, axis=1, keepdims=True)

    ### LOOK AT THE 0-PHASE SHIFTS
    # normalize
    l_normalized = np.empty_like(l)

    prominence = 3

    for i_t in range(chunk_length):
        l_normalized[i_t] = normalize_for_hilbert(l[i_t], prominence=prominence)

    # find the gradient dhdx
    i_gradmes = chunk_length//2
    l_gradmes = l_normalized[i_gradmes]
    
    analytic_signal = hilbert(np.concatenate((l_gradmes, l_gradmes[::-1])))[:len(l_gradmes)] # We use a small symmetrization trick here because the hilbert thinks everything in it has to be periodic smh
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase_wrapped = np.angle(analytic_signal)
    phase_unwrapped = np.unwrap(instantaneous_phase_wrapped)
    
    lbd = 1/utility.estimatesignalfrequency(l_gradmes, x=d)
    awayfromedges = (d > d.min() + lbd/8) * (d < d.max() - lbd/8)
    
    d_forfit = d[awayfromedges] * um_per_px
    h_forfit = phase_unwrapped[awayfromedges] / (2 * np.pi) * lambd /2
    
    result = linregress(d_forfit, h_forfit)
    dhdx = result.slope

    # find the zeros
    zeros = []
    for i_t in range(chunk_length):
        zeros.append(utility.find_roots(d, l_normalized[i_t]))

    lbd = 1/utility.estimatesignalfrequency(l_normalized[0], x=d)
    try:
        zeros = np.array(zeros)
    except:
        utility.log_debug("AAAAAAH CERTAINS SIGNAUX ONT DES ZEROS QUE D'AUTRE N'ONT PAS")
        zeros = [zero[(zero > d.min() + lbd/8) * (zero < d.max() - lbd/8)] for zero in zeros]

        try:
            # print([len(zero) for zero in zeros])
            zeros = np.array(zeros)
        except:
            utility.log_error("AAAAAAH CERTAINS SIGNAUX ONT DES ZEROS QUE D'AUTRE N'ONT PAS")
    # removce the zeros too close to the edges
    if zeros[:, 0].mean() < d.min() + lbd/8:
        zeros = zeros[:, 1:]
    if zeros[:, -1].mean() > d.max() - lbd/8:
        zeros = zeros[:, :-1]

    # intercept the zeros
    n_z = zeros.shape[1]
    slopes, intercetps = np.empty(n_z, dtype=float), np.empty(n_z, dtype=float)
    for i_z in range(n_z):
        slopes[i_z], intercetps[i_z] = np.polyfit(np.arange(chunk_length), zeros[:, i_z], 1)

    slope_mean = np.mean(slopes)
    slope_spread = np.std(slopes)

    ### LOOK AT THE CORRELATION
    # windowing to make edges match the 0-padding
    l_windowed = l * np.expand_dims( utility.get_window('tukey', len(d)), 0)

    lags = np.zeros((chunk_length * (chunk_length - 1))// 2, dtype=float)
    shifts = np.zeros((chunk_length * (chunk_length - 1)) // 2, dtype=float)

    pxlags = correlation_lags(l_windowed.shape[1], l_windowed.shape[1], mode='full')

    i_0 = 0
    for lag in np.arange(1, chunk_length):
        for i_t in range(chunk_length - lag):
            lags[i_0 + i_t] = lag
            shifts[i_0 + i_t] = utility.find_global_peak(pxlags, correlate(l_windowed[i_t], l_windowed[i_t + lag], mode='full'), peak_category='max')
        i_0 += chunk_length - lag


    weights = lags / (chunk_length-1)

    weights = np.array([lag / (chunk_length-1) for lag in lags])
    # weights[weights < 1/2] *= 0
    weights[weights > 0] = weights[weights > 0] - weights[weights > 0].min()
    weights[weights > 0] = weights[weights > 0] / weights[weights > 0].max()

    # linear regression
    popt, pcov = curve_fit(fn, lags, shifts, p0=[1], sigma=[1/weight if weight > 0 else np.inf for weight in weights])
    shiftspeed_reg = popt[0]
    # the longest lag
    shiftspeed_long = shifts[lags==(chunk_length-1)][0] / (chunk_length-1)

    shiftspeed = shifts / lags
    shiftspeed_weights = weights

    # just a good ol'mean
    shiftspeed_avg = np.average(shiftspeed, weights=shiftspeed_weights)

    shiftspeed_std = np.sqrt(np.average((shiftspeed - shiftspeed_avg) ** 2, weights=shiftspeed_weights))

    movespeeds_avg_down[i_chunk] = -slope_mean
    movespeeds_std_down[i_chunk] = slope_spread
    shiftspeeds_down.append(shiftspeed)
    shiftspeeds_weights_down.append(shiftspeed_weights)
    shiftspeeds_avg_down[i_chunk] = shiftspeed_avg
    shiftspeeds_std_down[i_chunk] = shiftspeed_std
    slopes_down[i_chunk] = dhdx


# <codecell>

t_chunks_up = np.array([tfr_glob_up[chunk].mean() for chunk in chunks_up])
t_chunks_down = np.array([tfr_glob_down[chunk].mean() for chunk in chunks_down])


# <codecell>




# <codecell>

fig, axes = plt.subplots(3,1, sharex=True)

ax = axes[0]
for i_chunk, chunk in enumerate(chunks_up):
    ax.scatter([t_chunks_up[i_chunk]] * len(shiftspeeds_up[i_chunk]), shiftspeeds_up[i_chunk], s=10, alpha=shiftspeeds_weights_up[i_chunk])
for i_chunk, chunk in enumerate(chunks_down):
    ax.scatter([t_chunks_down[i_chunk]] * len(shiftspeeds_down[i_chunk]), shiftspeeds_down[i_chunk], s=10, alpha=shiftspeeds_weights_down[i_chunk])
# ax.scatter(t_chunks, shiftspeed_mean, c='k', s=50)

ax.errorbar(t_chunks_up, shiftspeeds_avg_up, yerr=shiftspeeds_std_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_down, shiftspeeds_avg_down, yerr=shiftspeeds_std_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (down) by correl')

ax.legend()
# ax.scatter(t_chunks_up, shiftspeeds_best_up)
# ax.set_xlabel('Time [frames]')
ax.set_ylabel('Shift speed [px/frame]')
ax.set_ylim(0, 1.5)


ax = axes[1]
ax.errorbar(t_chunks_up, shiftspeeds_avg_up, yerr=shiftspeeds_std_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_down, shiftspeeds_avg_down, yerr=shiftspeeds_std_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_up, movespeeds_avg_up, yerr=movespeeds_std_up, ls='', marker='^', color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (up)')
ax.errorbar(t_chunks_down, movespeeds_avg_down, yerr=movespeeds_std_down, ls='', marker='v', color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (down)')

ax.legend()
ax.set_xlabel('Time [frames]')
ax.set_ylabel('Shift speed [px/frame]')
ax.set_ylim(0, 1.5)

ax = axes[2]
ax.errorbar(t_chunks_up, slopes_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='(up)')
ax.errorbar(t_chunks_down, slopes_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='(down)')

ax.legend()
ax.set_xlabel('Time [frames]')
ax.set_ylabel('Slope along crest [um/um]')
ax.set_ylim(0, 1e-4)


# <codecell>

fig, axes = plt.subplots(3,1, sharex=True)

ax = axes[0]
for i_chunk, chunk in enumerate(chunks_up):
    ax.scatter([t_chunks_up[i_chunk]] * len(shiftspeeds_up[i_chunk]), shiftspeeds_up[i_chunk], s=10, alpha=shiftspeeds_weights_up[i_chunk])
for i_chunk, chunk in enumerate(chunks_down):
    ax.scatter([t_chunks_down[i_chunk]] * len(shiftspeeds_down[i_chunk]), shiftspeeds_down[i_chunk], s=10, alpha=shiftspeeds_weights_down[i_chunk])
# ax.scatter(t_chunks, shiftspeed_mean, c='k', s=50)

ax.errorbar(t_chunks_up, shiftspeeds_avg_up, yerr=shiftspeeds_std_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_down, shiftspeeds_avg_down, yerr=shiftspeeds_std_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (down) by correl')

ax.legend()
# ax.scatter(t_chunks_up, shiftspeeds_best_up)
# ax.set_xlabel('Time [frames]')
ax.set_ylabel('Shift speed [px/frame]')
ax.set_ylim(0, 1.5)


ax = axes[1]
ax.errorbar(t_chunks_up, shiftspeeds_avg_up, yerr=shiftspeeds_std_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_down, shiftspeeds_avg_down, yerr=shiftspeeds_std_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
ax.errorbar(t_chunks_up, movespeeds_avg_up, yerr=movespeeds_std_up, ls='', marker='^', color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (up)')
ax.errorbar(t_chunks_down, movespeeds_avg_down, yerr=movespeeds_std_down, ls='', marker='v', color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (down)')

ax.legend()
ax.set_xlabel('Time [frames]')
ax.set_ylabel('Shift speed [px/frame]')
ax.set_ylim(0, 1.5)

ax = axes[2]
ax.errorbar(t_chunks_up, slopes_up, ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='(up)')
ax.errorbar(t_chunks_down, slopes_down, ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='(down)')

ax.legend()
ax.set_xlabel('Time [frames]')
ax.set_ylabel('Slope along crest [um/um]')
ax.set_ylim(0, 1e-4)


# <codecell>

meanslope = np.concatenate((slopes_up, slopes_down)).mean()


# <codecell>

# activate_saveplot()


# <codecell>

fig, axes = plt.subplots(2,1, sharex=True, figsize=(utility.genfig.figw['simple'], utility.genfig.figw['simple']/1.618))

vitunit = um_per_px * acquisition_frequency

markertop = '^'
markerbot = 'v'

ax = axes[0]
color = 'c'
s = 5
for i_chunk, chunk in enumerate(chunks_up):
    ax.scatter([t_chunks_up[i_chunk] / acquisition_frequency] * len(shiftspeeds_up[i_chunk]), shiftspeeds_up[i_chunk]*vitunit, 
               s=s, alpha=shiftspeeds_weights_up[i_chunk]**2, color=color)
for i_chunk, chunk in enumerate(chunks_down):
    alpha = shiftspeeds_weights_down[i_chunk]**2
    # alpha = np.log(shiftspeeds_weights_down[i_chunk])
    ax.scatter([t_chunks_down[i_chunk] / acquisition_frequency] * len(shiftspeeds_down[i_chunk]), shiftspeeds_down[i_chunk]*vitunit, 
               s=s, alpha=alpha, color=color)
ax.scatter([], [],
           s=s, alpha=1, color=color, label='For different timelags')

ax.errorbar(t_chunks_up / acquisition_frequency, shiftspeeds_avg_up*vitunit, yerr=shiftspeeds_std_up*vitunit, ls='', marker=markertop, color='k', lw=1, mfc='w', capsize = 3, label='Average value')
ax.errorbar(t_chunks_down / acquisition_frequency, shiftspeeds_avg_down*vitunit, yerr=shiftspeeds_std_down*vitunit, ls='', marker=markerbot, color='k', lw=1, mfc='w', capsize = 3, label='shift spd ')

ax.legend()
# ax.scatter(t_chunks_up, shiftspeeds_best_up)
ax.set_ylabel(r'Displacement speed [$\mu$m/s]')
ax.set_ylim(0, 500)


# ax = axes[1]
# ax.errorbar(t_chunks_up / acquisition_frequency, shiftspeeds_avg_up*vitunit, yerr=shiftspeeds_std_up*vitunit, ls='', marker=markertop, color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
# ax.errorbar(t_chunks_down / acquisition_frequency, shiftspeeds_avg_down*vitunit, yerr=shiftspeeds_std_down*vitunit, ls='', marker=markerbot, color='k', lw=1, mfc='w', capsize = 3, label='shift spd (up) by correl')
# ax.errorbar(t_chunks_up / acquisition_frequency, movespeeds_avg_up*vitunit, yerr=movespeeds_std_up*vitunit, ls='', marker=markertop, color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (up)')
# ax.errorbar(t_chunks_down / acquisition_frequency, movespeeds_avg_down*vitunit, yerr=movespeeds_std_down*vitunit, ls='', marker=markerbot, color='r', lw=1, mfc='w', capsize = 3, label='dhdt/dhdx (down)')
# 
# ax.legend()
# ax.set_ylabel(r'Displacement speed [$\mu$m/s]')
# ax.set_ylim(0, 400)

ax = axes[1]
ax.axhline(meanslope, color='gray', linestyle='--', label='Average slope')
ax.errorbar(t_chunks_up / acquisition_frequency, slopes_up, ls='', marker=markertop, color='k', lw=1, mfc='w', capsize = 3, label='Measurement points')
ax.errorbar(t_chunks_down / acquisition_frequency, slopes_down, ls='', marker=markerbot, color='k', lw=1, mfc='w', capsize = 3, label='(down)')
ax.set_ylim(0, 1e-4)
ax.set_yticks(np.arange(0, 10+1, 2)*1e-5)
ax.set_yticklabels(['0'] + [fr'${n}\cdot$'+r'$10^{-5}$' for n in np.arange(0, 10+1, 2)][1:-1] + [r'$10^{-4}$'])


ax.legend()
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel(r'Slope $|\partial_x h|$')
ax.set_ylim(0, 1e-4)

plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
# utility.save_graphe('slope')


# <codecell>


nu = 1.00
g = 9.81

vdrift_up = shiftspeeds_avg_up * um_per_px * acquisition_frequency
u_vdrift_up = shiftspeeds_std_up * um_per_px * acquisition_frequency

vdrift_down = shiftspeeds_avg_down * um_per_px * acquisition_frequency
u_vdrift_down = shiftspeeds_std_down * um_per_px * acquisition_frequency

h_fromdrift_up = np.sqrt(vdrift_up * nu / g)
herr_up = h_fromdrift_up * u_vdrift_up / vdrift_up / 2
h_fromdrift_down = np.sqrt(vdrift_down * nu / g)
herr_down = h_fromdrift_down * u_vdrift_down / vdrift_down / 2

t1 = 1345/acquisition_frequency
h1 = 22*h0
dh1 = 3*h0
v1 = h1**2 * g / nu
dv1 = dh1/h1 * v1 * 2

t2 = 1367/acquisition_frequency
h2 = 22*h0
dh2 = 3*h0
v2 = h2**2 * g / nu
dv2 = dh2/h2 * v2 * 2


# N_samples = np.array([np.sum(shiftspeeds_weights_up[i_chunk]) for i_chunk in range(len(chunks_up))])
# 
# herr = shiftspeed_std / shiftspeed_avg * h_fromdrift/2 / np.sqrt(N_samples)


# <codecell>

h0


# <codecell>

fig, axes = plt.subplots(1,1, sharex=True)

ax = axes

ax.errorbar(t_chunks_up / acquisition_frequency, vdrift_up, yerr=u_vdrift_up, 
            ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3)
ax.errorbar(t_chunks_down / acquisition_frequency, vdrift_down, yerr=u_vdrift_down, 
            ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3)

ax.errorbar(t1, v1, yerr=dv1,
            ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3, label='Direct measurement (counting fringes)')
ax.errorbar(t2, v2, yerr=dv2,
            ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3, label='Direct measurement (counting fringes)')

for p in range(1, 30):
    plt.axhspan(((2*p-1)*h0)**2*g/nu, ((2*p)*h0)**2*g/nu, color='gray', alpha=.1)
    
ax.set_xlabel('Time [s]')
ax.set_ylabel('Shift speed [um/s]')

ax.set_ylim(0, 300)



# <codecell>




# <codecell>

fig, axes = plt.subplots(1,1, sharex=True)

ax = axes

ax.errorbar(t_chunks_up / acquisition_frequency, vdrift_up * slopes_up*1000, yerr=u_vdrift_up * slopes_up*1000, 
            ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='Measured from fringes movement')
ax.errorbar(t_chunks_down / acquisition_frequency, vdrift_down * slopes_down*1000, yerr=u_vdrift_down * slopes_down*1000, 
            ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='Measured from fringes displacement')

# ax.errorbar(t1, v1 * slopes_down.mean(), yerr=dv1 * slopes_up.mean(),
#             ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3, label='Direct measurement (counting fringes)')
# ax.errorbar(t2, v2 * slopes_down.mean(), yerr=dv2 * slopes_down.mean(),
#             ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3, label='Direct measurement (counting fringes)')
ax.axhspan((v1 - dv1)*slopes_up.mean()*1000, (v1 + dv1)*slopes_up.mean()*1000, color='g', alpha=.2, label=rf'Poiseuille drainage $g\,h^2\,\partial_x h/\nu$')


# for p in range(1, 30):
#     plt.axhspan(((2*p-1)*h0)**2*g/nu, ((2*p)*h0)**2*g/nu, color='gray', alpha=.1)
    
ax.set_xlabel(r'Time $t$ [s]')
ax.set_ylabel(r'Height change $|\partial h/\partial t|$ [nm/s]')

ax.set_ylim(0, 20)
ax.legend()

# plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
# utility.save_graphe('testinch')


# <codecell>

# fig, axes = plt.subplots(1,1, sharex=True)
# ax = axes
# 
# # ax.axhline(3.2, color='r', ls='--')
# ax.errorbar(t_chunks_up / acquisition_frequency, h_fromdrift_up, yerr=herr_up,
#             ls='', marker='^', color='k', lw=1, mfc='w', capsize = 3, label='Estimated using drift speed')
# ax.errorbar(t_chunks_down / acquisition_frequency, h_fromdrift_down, yerr=herr_down,
#             ls='', marker='v', color='k', lw=1, mfc='w', capsize = 3, label='Estimated using drift speed')
# if acquisition=='1Hz_start':
# 
#     ax.errorbar(t1, h1, yerr=dh1,
#                 ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3)
#     ax.errorbar(t2, h2, yerr=dh2,
#                 ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3)
#     ax.errorbar([], [], yerr=[],
#                 ls='', marker='o', color='r', lw=1, mfc='w', capsize = 3, label='Direct measurement (counting fringes)')
#     
# for p in range(1, 30):
#     plt.axhspan((2*p-1)*h0, (2*p)*h0, color='gray', alpha=.1)
# ax.set_xlabel('Time [frames]')
# ax.set_ylabel('Film crest height [um]')
# # ax.set_ylim(0, (h_fromdrift+herr).max() * 1.1)
# # ax.set_ylim(0, 4)
# ax.legend()
# 
# if SAVEPLOT:
#     utility.save_graphe('directmeasuredrift')


# <codecell>

h0


# <codecell>

slopes_down.mean()


# <codecell>




# <codecell>




# <codecell>




# <codecell>




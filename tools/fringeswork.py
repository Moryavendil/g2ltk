from numpy import ndarray, dtype
from typing import Optional, Any, Tuple, Dict, List
import numpy as np


###### MATHS

import math
from scipy.interpolate import CubicSpline, make_smoothing_spline # for cubic interpolation
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter, butter, filtfilt, correlate, correlation_lags, find_peaks, hilbert
from scipy.stats import linregress

from tools import display, log_error, log_warn, log_info, log_debug, log_trace, log_subtrace
from tools import utility


def singlepeak_gauss(z_, position, width):
    # modelisation = gaussienne
    return utility.gaussian_unnormalized(z_, position, width)
    # # modelisation = doubletanch
    # return (np.tanh(z_-position-width/2) - np.tanh(z_-position+width/2) + 1)/2

def signal_bridge(z_, bckgnd_light, bridge_centre, depth, peak_width, peak_spacing):
    return bckgnd_light - depth*(singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) + singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width))

def signal_uneven_bridge(z_, bckgnd_light, bridge_centre, depth_1, depth_2, peak_width, peak_spacing):
    return bckgnd_light - depth_1*singlepeak_gauss(z_, bridge_centre-peak_spacing/2, peak_width) - depth_2*singlepeak_gauss(z_, bridge_centre+peak_spacing/2, peak_width)

def find_symmetric_bridge_centre(z_, l_, local_only=False):
    bckgnd_light_0 = np.percentile(l_, 75)
    depth_0 = bckgnd_light_0 - l_.min()

    # brutal estimates
    peak_width_0 = 20
    peak_spacing_0 = 50
    peaks = find_peaks(l_.max()-l_, height = depth_0/2, width = peak_width_0, distance = peak_spacing_0)[0]
    # if len(peaks) < 2:
    #     print('LESS THAN TWO PEAKS FOUND !')
    # elif len(peaks) > 2:
    #     print('More than 2 peaks found')
    peaks_height = (l_.max()-l_)[peaks]
    peak1 = peaks[np.argsort(peaks_height)[-1]] if len(peaks) > 0 else -1
    peak2 = peaks[np.argsort(peaks_height)[-2]] if len(peaks) > 1 else -1
    if len(peaks) > 2 and local_only:
        pairsofvalidpeaks = []
        for i in range(len(peaks)-1):
            z1, z2 = peaks[i], peaks[i+1]
            l1, l2 = l_[z1], l_[z2]
            if (np.abs(l1 - l2) < 20) and (np.abs(l1 - l2) < 2*peak_spacing_0):
                pairsofvalidpeaks.append([z1, z2])

        bestpairofvalidpeaks = pairsofvalidpeaks[np.argmin([(l_[pairofvalidpeaks[0]]+l_[pairofvalidpeaks[1]])/2for pairofvalidpeaks in pairsofvalidpeaks])]
        peak1, peak2 = bestpairofvalidpeaks

    bridge_centre_0 = (peak1 + peak2)/2
    peak_spacing_0 = np.abs(peak1 - peak2) # refine peak spacing

    p0 = (bckgnd_light_0, bridge_centre_0, depth_0, peak_width_0, peak_spacing_0)

    xfit = z_
    yfit = l_
    sigma=None

    if local_only:
        crit = (z_ < max(peak1, peak2) + peak_spacing_0*1.) * (z_ > min(peak1, peak2) - peak_spacing_0*1.)
        xfit = z_[crit]
        yfit = l_[crit]
        sigma = None

    popt, pcov = curve_fit(signal_bridge, xfit, yfit, p0=p0, sigma=sigma)
    return popt[1]

def find_uneven_bridge_centre(z_, l_, peak_width_0=30, peak_depth_0=90, peak_spacing_0 = 60, peak_spacing_max = 100, peak_spacing_min = 40, hint=None,
                              hint_zone_size=None):
    # brutal estimates

    l_ = savgol_filter(l_, peak_width_0, 2)

    l_findmainpeak = 255 - l_
    if hint is not None:
        hint_zone_size = hint_zone_size or (peak_width_0+peak_spacing_max)*2
        l_findmainpeak *= utility.gaussian_unnormalized(z_, hint, hint_zone_size)
    # super brutal
    peak1_z = z_[np.argmax(l_findmainpeak)]

    zone_findpeak2 = (z_ < peak1_z + peak_spacing_max + 5) * (z_ > peak1_z - peak_spacing_max + 5)
    z_peak2 = z_[zone_findpeak2]
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
def find_riv_pos_raw(slice, z = None, problem_threshold = None):
    if z is None:
        z = np.arange(slice.shape[1])

    pos_raw = np.array([find_uneven_bridge_centre(z, slice[i_t]) for i_t in range(len(slice))])

    if problem_threshold is not None:
        for i in range(1, len(pos_raw)):
            if np.abs(pos_raw[i] - pos_raw[i-1]) > problem_threshold:
                hint = pos_raw[i-1]
                hint_zone_size = None
                if i > 1:
                    hint_zone_size = np.abs((pos_raw[i-1] - pos_raw[i-2])*2)
                    hint = pos_raw[i-1] + (pos_raw[i-1] - pos_raw[i-2])
                pos_raw[i] = find_uneven_bridge_centre(z, slice[i], hint=hint, hint_zone_size=hint_zone_size)

    return pos_raw

### 2D PHASE PICTURES
def findminmaxs(signal, x=None, prominence=5, distance=100, forcedmins=None, forcedmaxs=None, width=None):
    if x is not None:
        distance = distance / utility.step(x)
    if distance < 1:
        distance=None
    maxs = find_peaks( signal, prominence = prominence, distance=distance, width=width)[0]
    mins = find_peaks(-signal, prominence = prominence, distance=distance, width=width)[0]
    if forcedmins is not None:
        mins = np.concatenate((mins, forcedmins))
    if forcedmaxs is not None:
        maxs = np.concatenate((maxs, forcedmaxs))
    # We sort the mins and maxs so they are in order
    mins.sort()
    maxs.sort()
    # remove consecutive mins
    minstoremove = np.array([], dtype=int)
    for i in range(len(mins)-1):
        # si il n'y a pas de max entre deux mins
        if np.sum( (maxs > mins[i])*(maxs < mins[i+1]) ) == 0:
            suspects = np.array([i, i+1])
            # en conserve le meilleur, i.e. plus petit des suspects (donc le minimum le plus profond)
            toremove = np.delete(suspects, np.argmin(mins[suspects]))
            # On enleve les autres suspects
            minstoremove = np.concatenate((minstoremove, toremove))
    minsremoved = mins[minstoremove]
    mins = np.delete(mins, minstoremove)
    # remove consecutive maxs
    maxstoremove = np.array([], dtype=int)
    for i in range(len(maxs)-1):
        # si il n'y a pas de min entre deux maxs
        if np.sum( (mins > maxs[i])*(mins < maxs[i+1]) ) == 0:
            suspects = np.array([i, i+1])
            # en conserve le meilleur, i.e. plus grand des suspects (donc le maximum le plus élevé)
            toremove = np.delete(suspects, np.argmax(maxs[suspects]))
            # On enleve les autres suspects
            maxstoremove = np.concatenate((maxstoremove, toremove))
    maxsremoved = maxs[maxstoremove]
    maxs = np.delete(maxs, maxstoremove)
    return mins, maxs

def find_cminmax(signal, x=None, prominence=5, distance=100, forcedmins=None, forcedmaxs=None):
    mins, maxs = findminmaxs(signal, x=x, prominence=prominence, distance=distance, forcedmins=forcedmins, forcedmaxs=forcedmaxs)
    if x is None:
        x = np.arange(len(signal))

    l_maxs_cs = np.poly1d(np.polyfit(x[maxs], signal[maxs], 1))
    if len(maxs) > 5:
        l_maxs_cs = make_smoothing_spline(x[maxs], signal[maxs], lam=None)
    l_mins_cs = np.poly1d(np.polyfit(x[mins], signal[mins], 1))
    if len(maxs) > 5:
        l_mins_cs = make_smoothing_spline(x[mins], signal[mins], lam=None)
    return l_mins_cs, l_maxs_cs

def normalize_for_hilbert(signal, x=None, prominence=5, distance=100, forcedmins=None, forcedmaxs=None):
    l_mins_cs, l_maxs_cs = find_cminmax(signal, x=x, prominence=prominence, distance=distance, forcedmins=forcedmins, forcedmaxs=forcedmaxs)
    if x is None:
        x = np.arange(len(signal))

    offset = (l_maxs_cs(x) + l_mins_cs(x))/2
    amplitude = (l_maxs_cs(x) - l_mins_cs(x))/2
    # amplitude = 1
    signal_normalized = (signal - offset) / amplitude
    return signal_normalized

def prepare_signal_for_hilbert(signal, x=None, oversample=True, usesplines=False):
    if x is None:
        x = np.arange(len(signal))
    x_hilbert = x.copy()
    signal_hilbert = signal.astype(float, copy=True)
    # we want to gain a bit in resolution especially on the edges where the extrema can be very close
    if oversample:
        x_oversampled = np.linspace(x.min(), x.max(), len(x) + (len(x)-1)*2, endpoint=True)
        signal_oversampled = np.interp(x_hilbert, x, signal)
        x_hilbert = x_oversampled
        signal_hilbert = signal_oversampled
    if usesplines:
        # we smooth everything a bit to ease the phase reconstruction process via Hilbert transform
        hilbert_spline = make_smoothing_spline(x, signal, lam=None)
        signal_hilbert = hilbert_spline(x_hilbert)
    return x_hilbert, signal_hilbert

def instantaneous_phase(signal, x=None, oversample=True, usesplines=False, symmetrize=True):
    x_hilbert, sig_hilbert = prepare_signal_for_hilbert(signal, x=x, oversample=oversample, usesplines=usesplines)

    ## Now we use the hilbert transform to do some **magic**
    analytic_signal = None
    if not symmetrize:
        # this is brutal (but sometimes works well, just not on film detection.
        # it avoid the phase bad definition if at the end we are not on a min / max
        analytic_signal = hilbert(sig_hilbert)
    if symmetrize:
        # We use a small symmetrization trick here because the hilbert thinks everything in life has to be periodic smh
        analytic_signal = hilbert(np.concatenate((sig_hilbert, sig_hilbert[::-1])))[:len(sig_hilbert)]
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase_wrapped = np.angle(analytic_signal)
    return x_hilbert, instantaneous_phase_wrapped
from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math
from scipy.signal.windows import get_window  # FFT windowing
from skimage import filters  # filters.window for 2D FFT windowing
from scipy import fft

from .. import log_error, log_warning, log_info, log_debug, log_trace, log_subtrace
from g2ltk.peakfinder import step, span, interp_roots, find_global_max
from . import floatarray1D, complexarray1D, attenuate_power


### Dual: changing from real space to frequency space
def dual1d(arr: floatarray1D, zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> floatarray1D:
    """
    Returns the dual, i.e. the frequencies.

    The unit is the inverse, e.g. time (s)-> frequency (Hz).

    Parameters
    ----------
    arr
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    N = arr.shape[0]
    log_trace(f'dual: Called with {arr.shape} array, zp = {zero_pad}, zpf = {zero_pad_factor}')

    pad_width = None
    if zero_pad_factor is not None:
        log_subtrace(f'rft2d: | zero_pad_factor={zero_pad_factor}')
        try:
            pad = np.rint(N * (zero_pad_factor - 1)).astype(int)
            pad_width = (pad // 2, pad // 2 + pad % 2)
        except:
            log_warning(f'rft2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
            pad_width = None
    elif zero_pad is not None:
        log_subtrace(f'rft2d: | zero_pad={zero_pad}')
        try:
            pad_width = (0, int(zero_pad))
        except:
            log_warning(f'rft2d: What is this zero-padding "{zero_pad}" ? I made it None')
            pad_width = None

    log_subtrace(f'rft2d: Padding (artificially better resolution) | pad={pad_width}')
    if pad_width is None:
        pad_width = (0, 0)

    n = N + pad_width[0] + pad_width[1]

    log_debug(f'dual: {N} -> {n}')

    return np.fft.fftshift(np.fft.fftfreq(n, step(arr)))


def rdual1d(arr: floatarray1D, zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> floatarray1D:
    """Returns the dual, i.e. the frequencies, in a numpy.fft.rfft-compatible style.

    The unit is the inverse, e.g. time (s)-> frequency (Hz).

    Parameters
    ----------
    arr
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    N = arr.shape[0]
    log_trace(f'rdual: Called with {arr.shape} array, zp = {zero_pad}, zpf = {zero_pad_factor}')

    pad_width = None
    if zero_pad_factor is not None:
        log_subtrace(f'rft2d: | zero_pad_factor={zero_pad_factor}')
        try:
            pad = np.rint(N * (zero_pad_factor - 1)).astype(int)
            pad_width = (pad // 2, pad // 2 + pad % 2)
        except:
            log_warning(f'rft2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
            pad_width = None
    elif zero_pad is not None:
        log_subtrace(f'rft2d: | zero_pad={zero_pad}')
        try:
            pad_width = (0, int(zero_pad))
        except:
            log_warning(f'rft2d: What is this zero-padding "{zero_pad}" ? I made it None')
            pad_width = None

    log_subtrace(f'rft2d: Padding (artificially better resolution) | pad={pad_width}')
    if pad_width is None:
        pad_width = (0, 0)

    n = N + pad_width[0] + pad_width[1]

    log_debug(f'rdual: {N} -> {n}')

    return np.fft.rfftfreq(n, step(arr))

### FT: computing the Fourier Transform
def prepare_signal_for_ft1d(arr: complexarray1D,
                            window: str = 'hann', remove_mean: bool = True,
                            zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> complexarray1D:
    N = arr.shape[0]

    # Removing mean
    z_nozero = arr - np.mean(arr)*remove_mean

    # windowing
    z_win = z_nozero * get_window(window, N)

    pad_width = None
    if zero_pad_factor is not None:
        log_subtrace(f'rft2d: | zero_pad_factor={zero_pad_factor}')
        try:
            pad = np.rint(N * (zero_pad_factor - 1)).astype(int)
            pad_width = (pad // 2, pad // 2 + pad % 2)
        except:
            log_warning(f'rft2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
            pad_width = None
    elif zero_pad is not None:
        log_subtrace(f'rft2d: | zero_pad={zero_pad}')
        try:
            pad_width = (0, int(zero_pad))
        except:
            log_warning(f'rft2d: What is this zero-padding "{zero_pad}" ? I made it None')
            pad_width = None

    log_subtrace(f'rft2d: Padding (artificially better resolution) | pad={pad_width}')
    if pad_width is None:
        pad_width = (0, 0)
    z_pad = np.pad(z_win, pad_width=pad_width, mode='constant', constant_values=0)

    # z_treated -= np.mean(z_treated) * (1-1e-12) # this is to avoid having zero amplitude and problems when taking the log
    z_clean = z_pad - np.mean(z_pad)

    log_subtrace(f'rft2d: Rolling (restoring phase) | pad={pad_width}')
    z_roll = np.roll(z_clean, pad_width[0] + N // 2)

    return z_roll


def rft1d(arr: floatarray1D, x: Optional[floatarray1D] = None,
          window: str = 'hann', remove_mean: bool = True, norm=None,
          zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> complexarray1D:
    """ Returns the 1-D Fourier transform of the input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s) -> V/Hz(Hz)

    Parameters
    ----------
    arr
    x
    window
    norm
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    log_trace(f'rft2d: Computing 2-D FFT of array of shape {arr.shape}')

    z_roll = prepare_signal_for_ft1d(arr, window=window, remove_mean=remove_mean,
                                     zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)

    z_hat = fft.rfft(z_roll, norm=norm, n=z_roll.shape[0])

    return z_hat * step(x)


def ft1d(arr: complexarray1D, x: Optional[floatarray1D] = None,
         window: str = 'hann', remove_mean: bool = True, norm=None,
         zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> complexarray1D:
    """ Returns the 1-D Fourier transform of the input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s) -> V/Hz(Hz)

    Parameters
    ----------
    arr
    x
    window
    norm
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    log_trace(f'rft2d: Computing 2-D FFT of array of shape {arr.shape}')

    arr_prepared = prepare_signal_for_ft1d(arr, window=window, remove_mean=remove_mean,
                                           zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)

    z_hat = fft.fft(arr_prepared, norm=norm, n=arr_prepared.shape[0])

    return fft.fftshift(z_hat) * step(x)


def ifft1d(zhat: complexarray1D, xdual: Optional[np.ndarray] = None):
    return fft.fftshift(fft.ifft(np.fft.ifftshift(zhat))) * span(xdual)


### Power:

def window_factor1d(window: str):
    """
    Returns the factor by which the energy is multiplied when the signal is windowed.

    Parameters
    ----------
    window

    Returns
    -------

    """
    if window is None:
        return 1.
    elif window == 'boxcar':
        return 1.
    elif window == 'hann':
        return 8 / 3
    else:
        N = 10000
        return 1 / ((get_window(window, N) ** 2).sum() / N)



def psd1d(z: np.ndarray, x: Optional[np.ndarray] = None,
          window: str = 'hann', remove_mean: bool = True,
          zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None) -> floatarray1D:
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t) is [V(s)], then the unit of $\hat{z}$ is [V/Hz(Hz)]
    z_ft = rft1d(z, x=x, window=window, remove_mean=remove_mean, norm="backward",
                 zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the only thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this frequency
    # It is useful in itself for time-limited signals (impulsions)
    # if the unit of z(t) is [V(s)], then the unit of $ESD(z)$ is [V^2/Hz^2(Hz)]
    esd = np.abs(z_ft) ** 2 * window_factor1d(window)
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is periodic, then PSD = ESD / duration
    # Thus if the unit of z(t) is [V(s)], then the unit of PSD(z)$ is [V^2/Hz(Hz)]
    return esd / span(x)


def rpsd1d(z: floatarray1D, x: Optional[np.ndarray] = None,
           window: str = 'hann', remove_mean: bool = True,
           zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None) -> floatarray1D:
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t) is [V(s)], then the unit of $\hat{z}$ is [V/Hz(Hz)]
    z_ft = rft1d(z, x=x, window=window, norm="backward", remove_mean=remove_mean,
                 zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the only thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this frequency
    # It is useful in itself for time-limited signals (impulsions)
    # if the unit of z(t) is [V(s)], then the unit of $ESD(z)$ is [V^2/Hz^2(Hz)]
    esd = np.abs(z_ft) ** 2 * window_factor1d(window)
    esd[1:] *= 2  # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is periodic, then PSD = ESD / duration
    # Thus if the unit of z(t) is [V(s)], then the unit of PSD(z)$ is [V^2/Hz(Hz)]
    return esd / span(x)

### QOL functions

def estimatesignalfrequency(z: floatarray1D, x: Optional[floatarray1D] = None,
                            window: str = 'boxcar', zero_pad_factor: Optional[int] = 4,
                            bounds=None) -> float:
    """Estimates the frequency of a signal

    Parameters
    ----------
    z
    x
    window
        By default we use a rectangular ('boxcar') window since it is the most precise for single-frequency finding. But it is very sensitive to noise.
    zero_pad_factor
        By default we use zero-padding to have a 'more precise' frequency measurement.
    bounds

    Returns
    -------

    """
    if x is None:
        x = np.arange(len(z))
    fx: floatarray1D = rdual1d(x, zero_pad_factor=zero_pad_factor)
    pw: floatarray1D = rpsd1d(z, x, window=window, zero_pad_factor=zero_pad_factor)
    if bounds is not None:
        pw = pw[(fx > bounds[0]) & (fx < bounds[1])]
        fx = fx[(fx > bounds[0]) & (fx < bounds[1])]
    return find_global_max(pw, x=fx)


# find the edges of the peak (1D, everything is easy)
def peak_contour1d(peak_x, z: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D]=None,
                   peak_max_length: Optional[float] = None):
    if x is None:
        x = np.arange(z.shape[0])

    peak_index = np.argmin((x - peak_x) ** 2)

    min_peak_depth_dB = 10

    while peak_depth_dB > 0:  # infinite loop, return is in it
        zintercept = attenuate_power(z[peak_index], peak_depth_dB)
        x1_intercept = interp_roots(z - zintercept, x=x)
        x1_before = peak_x
        x1_after = peak_x
        if len(x1_intercept[x1_intercept < peak_x] > 0):
            x1_before = x1_intercept[x1_intercept < peak_x].max()
        if len(x1_intercept[x1_intercept > peak_x] > 0):
            x1_after = x1_intercept[x1_intercept > peak_x].min()

        length = x1_after - x1_before
        isnottoobig = True if (peak_max_length is None) else length < peak_max_length

        contour_is_valid = isnottoobig

        if peak_max_length is not None:
            log_trace(f'contour length: {length} | Not too big (< {peak_max_length}): {isnottoobig}')
        log_trace(f'Valid contour: {contour_is_valid}')

        # find the contour that contains the point
        if contour_is_valid or peak_depth_dB == min_peak_depth_dB:
            return x1_before, x1_after
        else:
            peak_depth_dB -= 10
            if peak_depth_dB < min_peak_depth_dB: peak_depth_dB = min_peak_depth_dB
            log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")
    if peak_depth_dB != min_peak_depth_dB:
        peak_depth_dB = min_peak_depth_dB
        log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")

    # peak_index = np.argmin((x - peak_x) ** 2)
    # zintercept = attenuate_power(z[peak_index], peak_depth_dB)
    # x1_intercept = find_roots(x, z - zintercept)
    # x1_before = peak_x
    # x1_after = peak_x
    # if len(x1_intercept[x1_intercept < peak_x] > 0):
    #     x1_before = x1_intercept[x1_intercept < peak_x].max()
    # if len(x1_intercept[x1_intercept > peak_x] > 0):
    #     x1_after = x1_intercept[x1_intercept > peak_x].min()

    log_debug(f'peak_contour1d: Around {peak_x} ({min_peak_depth_dB} dB) : {(x1_before, x1_after)}')
    return x1_before, x1_after


def peak_vicinity1d(peak_x, z: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D]=None,
                    peak_contour: Optional[List] = None):
    if x is None:
        x = np.arange(z.shape[0])

    if peak_contour is None:
        peak_contour = peak_contour1d(peak_x=peak_x, z=z, peak_depth_dB=peak_depth_dB, x=x)
    x1_before, x1_after = peak_contour

    log_trace(f'peak_vicinity1d: Vicinity of {(x1_before, x1_after)} (around {peak_x})')

    vicinity = (x >= x1_before) & (x <= x1_after)

    log_debug(f'peak_vicinity1d: Found {np.sum(vicinity)} points in zone {(x1_before, x1_after)} (around {peak_x})')

    return vicinity


def power_near_peak1d(peak_x, z: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D]=None,
                      peak_vicinity: Optional[np.ndarray] = None,
                      peak_contour: Optional[List] = None):
    # integrate the PSD along the peak
    log_debug(f'Measuring the power around     ({round(peak_x, 3)})')
    if peak_vicinity is None:
        peak_vicinity = peak_vicinity1d(peak_x=peak_x, z=z, peak_depth_dB=peak_depth_dB, x=x, peak_contour=peak_contour)
    # ### (abandoned) do a trapezoid integration
    # x_f = np.concatenate(([freqpre], freqs[(freqpre < freqs)*(freqs < freqpost)], [freqpost]))
    # y_f = np.concatenate(([np.interp(freqpre, freqs, zmeanx_psd)], zmeanx_psd[(freqpre < freqs)*(freqs < freqpost)], [np.interp(freqpost, freqs, zmeanx_psd)]))
    # p_ft_peak = trapezoid(y_f, x_f)
    ### Go bourrin (we are in log we do not care) : rectangular integration
    pw = np.sum(z[peak_vicinity]) * step(x)
    log_debug(f'Power: {pw} (amplitude: {np.sqrt(pw * 2)})')
    return pw

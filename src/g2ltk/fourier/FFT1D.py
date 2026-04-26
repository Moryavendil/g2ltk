from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math
from scipy import signal
from scipy.signal.windows import get_window  # FFT windowing
from skimage import filters  # filters.window for 2D FFT windowing
from scipy import fft

from .. import log_error, log_warning, log_info, log_debug, log_trace, log_subtrace
from g2ltk.peakfinder import step, span, interp_roots, find_global_max
from . import floatarray1D, complexarray1D, attenuate_power

default_window: str = 'boxcar'


### Dual: changing from real space to frequency space
def dual1d(arr: floatarray1D,
           zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> floatarray1D:
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


def rdual1d(arr: floatarray1D,
            zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None) -> floatarray1D:
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
def prepare_signal_for_ft1d(sig: complexarray1D,
                            window: str = default_window, remove_mean: bool = True,
                            zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None,
                            shift: Optional[int] = None) -> complexarray1D:
    N = sig.shape[0]

    # Removing mean
    log_trace(f'rft1d: Mean removal: {remove_mean}')
    sig_nozero = sig.copy() - np.mean(sig) * remove_mean

    # windowing
    sig_windowed = sig_nozero * get_window(window, N)

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

    log_subtrace(f'rft1d: Padding (artificially better resolution) | pad={pad_width}')
    if pad_width is None:
        pad_width = (0, 0)
    sig_padded = np.pad(sig_windowed, pad_width=pad_width, mode='constant', constant_values=0)

    # second mean removal. It is unorthodox, is it necessary?
    # sig_padded = sig_padded - np.mean(sig_padded) * remove_mean

    if shift is None: shift = 0
    log_subtrace(f'rft1d: Rolling (restoring phase) | pad={pad_width} | shift={shift}')
    sig_rolled = np.roll(sig_padded, -pad_width[0] - shift)

    return sig_rolled


def rft1d(sig: floatarray1D, x: Optional[floatarray1D] = None,
          window: str = default_window, remove_mean: bool = True, norm=None,
          zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None,
          shift: Optional[int] = None) -> complexarray1D:
    """ Returns the 1-D Fourier transform of the input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s) -> V/Hz(Hz)

    Parameters
    ----------
    sig
    x
    window
    remove_mean
    norm
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    log_trace(f'rft2d: Computing 1-D FFT of array of shape {sig.shape}')

    sig_prepared = prepare_signal_for_ft1d(sig, window=window, remove_mean=remove_mean,
                                     zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                                     shift=shift)

    sig_hat = fft.rfft(sig_prepared, norm=norm, n=sig_prepared.shape[0])

    return sig_hat * step(x)


def ft1d(sig: complexarray1D, x: Optional[floatarray1D] = None,
         window: str = default_window, remove_mean: bool = True, norm=None,
         zero_pad: Optional[int] = None, zero_pad_factor: Optional[int] = None,
         shift: Optional[int] = None) -> complexarray1D:
    """ Returns the 1-D Fourier transform of the input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s) -> V/Hz(Hz)

    Parameters
    ----------
    sig
    x
    window
    remove_mean
    norm
    zero_pad
    zero_pad_factor

    Returns
    -------

    """
    log_trace(f'rft2d: Computing 1-D FFT of array of shape {sig.shape}')

    sig_prepared = prepare_signal_for_ft1d(sig, window=window, remove_mean=remove_mean,
                                           zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                                           shift=shift)

    sig_hat = fft.fft(sig_prepared, norm=norm, n=sig_prepared.shape[0])

    return fft.fftshift(sig_hat) * step(x)


def ift1d(sig_hat: complexarray1D, xdual: Optional[np.ndarray] = None):
    return fft.ifft(np.fft.ifftshift(sig_hat)) * span(xdual)


### Power:

def window_factor1d(window: str, N: int = 16384):
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
        return 1 / ((get_window(window, N) ** 2).sum() / N)


def psd1d(sig: np.ndarray, x: Optional[np.ndarray] = None,
          window: str = default_window, remove_mean: bool = True,
          zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None) -> floatarray1D:
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t) is [V(s)], then the unit of $\hat{z}$ is [V/Hz(Hz)]
    sig_hat = ft1d(sig, x=x, window=window, remove_mean=remove_mean, norm="backward",
                   zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the only thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this frequency
    # It is useful in itself for time-limited signals (impulsions)
    # if the unit of z(t) is [V(s)], then the unit of $ESD(z)$ is [V^2/Hz^2(Hz)]
    esd = np.abs(sig_hat) ** 2 * window_factor1d(window, N=len(sig))
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is periodic, then PSD = ESD / duration
    # Thus if the unit of z(t) is [V(s)], then the unit of PSD(z)$ is [V^2/Hz(Hz)]
    return esd / span(x)


def rpsd1d(sig: floatarray1D, x: Optional[np.ndarray] = None,
           window: str = default_window, remove_mean: bool = True,
           zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None) -> floatarray1D:
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t) is [V(s)], then the unit of $\hat{z}$ is [V/Hz(Hz)]
    sig_hat = rft1d(sig, x=x, window=window, norm="backward", remove_mean=remove_mean,
                 zero_pad=zero_pad, zero_pad_factor=zero_pad_factor)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the only thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this frequency
    # It is useful in itself for time-limited signals (impulsions)
    # if the unit of z(t) is [V(s)], then the unit of $ESD(z)$ is [V^2/Hz^2(Hz)]
    esd = np.abs(sig_hat) ** 2 * window_factor1d(window, N=len(sig))
    esd[1:] *= 2  # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is periodic, then PSD = ESD / duration
    # Thus if the unit of z(t) is [V(s)], then the unit of PSD(z)$ is [V^2/Hz(Hz)]
    return esd / span(x)

def welch1d(sig: floatarray1D, x: Optional[np.ndarray] = None,
            window: str = default_window, remove_mean: bool = True,
            zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None):
    fs = 1/step(t)
    detrend = 'constant' if remove_mean else False
    f_welch, psd_welch =  signal.welch(sig, fs=fs, window=window, nperseg=256, nfft=256*zpf,
                                       return_onesided=False, scaling='density', detrend=detrend)
    f_welch, psd_welch = fft.fftshift(f_welch), fft.fftshift(psd_welch)
    return psd_welch

### QOL functions

def estimatesignalfrequency(sig: floatarray1D, x: Optional[floatarray1D] = None,
                            window: str = 'boxcar', zero_pad_factor: Optional[int] = 4,
                            bounds=None) -> float:
    """Estimates the frequency of a signal

    Parameters
    ----------
    sig
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
        x = np.arange(len(sig))
    fx: floatarray1D = rdual1d(x, zero_pad_factor=zero_pad_factor)
    pw: floatarray1D = rpsd1d(sig, x, window=window, zero_pad_factor=zero_pad_factor)
    if bounds is not None:
        pw = pw[(fx > bounds[0]) & (fx < bounds[1])]
        fx = fx[(fx > bounds[0]) & (fx < bounds[1])]
    return find_global_max(pw, x=fx)


# find the edges of the peak (1D, everything is easy)
def peak_contour1d(peak_x, sig_psd: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D] = None,
                   peak_max_length: Optional[float] = None):
    if x is None: x = np.arange(sig_psd.shape[0])

    peak_index = np.argmin((x - peak_x) ** 2)

    min_peak_depth_dB = 10

    while peak_depth_dB > 0:  # infinite loop, return is in it
        zintercept = attenuate_power(sig_psd[peak_index], peak_depth_dB)
        x1_intercept = interp_roots(sig_psd - zintercept, x=x)
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


def peak_vicinity1d(peak_x, sig_psd: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D] = None,
                    peak_contour: Optional[List] = None):
    if x is None: x = np.arange(sig_psd.shape[0])

    if peak_contour is None:
        peak_contour = peak_contour1d(peak_x=peak_x, sig_psd=sig_psd, peak_depth_dB=peak_depth_dB, x=x)
    x1_before, x1_after = peak_contour

    log_trace(f'peak_vicinity1d: Vicinity of {(x1_before, x1_after)} (around {peak_x})')

    vicinity = (x >= x1_before) & (x <= x1_after)

    log_debug(f'peak_vicinity1d: Found {np.sum(vicinity)} points in zone {(x1_before, x1_after)} (around {peak_x})')

    return vicinity


def power_near_peak1d(peak_x, sig_psd: floatarray1D, peak_depth_dB: Optional[int] = 40, x: Optional[floatarray1D] = None,
                      peak_vicinity: Optional[np.ndarray] = None,
                      peak_contour: Optional[List] = None):
    """Integrates the PSD around a given peak.

    Choose your windw with care

    Parameters
    ----------
    peak_x
    sig_psd
    peak_depth_dB
    x
    peak_vicinity
    peak_contour

    Returns
    -------

    """
    log_debug(f'Measuring the power around     ({round(peak_x, 3)})')
    if peak_vicinity is None:
        peak_vicinity = peak_vicinity1d(peak_x=peak_x, sig_psd=sig_psd, peak_depth_dB=peak_depth_dB, x=x, peak_contour=peak_contour)
    # ### (abandoned) do a trapezoid integration
    # x_f = np.concatenate(([freqpre], freqs[(freqpre < freqs)*(freqs < freqpost)], [freqpost]))
    # y_f = np.concatenate(([np.interp(freqpre, freqs, zmeanx_psd)], zmeanx_psd[(freqpre < freqs)*(freqs < freqpost)], [np.interp(freqpost, freqs, zmeanx_psd)]))
    # p_ft_peak = trapezoid(y_f, x_f)
    ### Go bourrin (we are in log we do not care) : rectangular integration
    pw = np.sum(sig_psd[peak_vicinity]) * step(x)
    log_debug(f'Power: {pw} (amplitude: {np.sqrt(pw * 2)})')
    return pw


# find the phase and amplitude of the signal

def estimate_phase(sig, x: Optional[floatarray1D] = None, frequency: Optional[float] = None) -> float:
    """
    Estimates the phase of the signal at a given frequency.

    If the frequency is unspecified, try to estimate it automatically.

    Parameters
    ----------
    sig
    x
    frequency

    Returns
    -------

    """
    if x is None: x = np.arange(sig.shape[0])
    if frequency is None: frequency = estimatesignalfrequency(sig, x=x)
    return np.angle(np.sum(sig * np.exp(-1j * 2 * np.pi * frequency * x)))


def hilbert(sig: floatarray1D, symmetrize=True, remove_mean=True) -> floatarray1D:
    """
    Performs a Hilbert transform of the signal.

    This is mainly a wrapper for scipy.signal.hilbert,
    only here the signal is symmetrized to diminish boundary effects.

    Parameters
    ----------
    sig
    symmetrize
    remove_mean

    Returns
    -------

    """
    sig_prepared = np.concatenate((sig, sig[::-1])) if symmetrize else sig.copy()
    if remove_mean: sig_prepared -= sig_prepared.mean()
    return signal.hilbert(sig_prepared)[:len(sig)]


def synchronous_enveloppe_detection(sig, x: Optional[floatarray1D] = None, frequency: Optional[float] = None,
                                    phase: Optional[float] = None) -> floatarray1D:
    """
    Performs synchronous detection / coherent demodulation.

    Parameters
    ----------
    sig
    x
    frequency
    phase

    Returns
    -------

    """
    if x is None: x = np.arange(sig.shape[0])
    if frequency is None: frequency = estimatesignalfrequency(sig, x=x)
    if phase is None: phase = estimate_phase(sig, x=x, frequency=frequency)
    fs = 1 / step(x)
    b, a = signal.butter(4, frequency * 2 / 3, btype='lowpass', analog=False, fs=fs)
    return signal.filtfilt(b, a, 2 * sig * np.cos(2 * np.pi * frequency * x + phase))

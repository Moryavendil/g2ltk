from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math

from .. import set_verbose, log_error, log_warn, log_warning, log_info, log_subinfo, log_debug, log_trace, log_subtrace

### ARRAYS QOL routines
# These are the real deal, with the dimensions encoded.
# floatarray1D = np.ndarray[tuple[int], np.dtype[np.floating]]
# floatarray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
# complexarray1D = np.ndarray[tuple[int], np.dtype[np.inexact]]
# complexarray2D = np.ndarray[tuple[int, int], np.dtype[np.inexact]]
# now using them is a pain in the ass, so we relax the dimension
floatarray1D = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
floatarray2D = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
complexarray1D = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]
complexarray2D = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]


### log
def attenuate_power(value, attenuation_factor_dB):
    return value / math.pow(10, attenuation_factor_dB / 20)


def log_amplitude_range(maximum_amplitude: float, range_db: Union[float, int]):
    return maximum_amplitude, attenuate_power(maximum_amplitude, range_db)


def log_amplitude_cbticks(maximum_amplitude: float, range_db: Union[int, float]):
    step_major = 20
    step_minor = 5
    if range_db < 60:
        step_major = 10
        step_minor = 2
    if range_db < 30:
        step_major = 5
        step_minor = 1
    # it seems unreasonable to have range_db > 100 or < 10
    att_db_major = np.arange(0, range_db + 1, step_major)
    att_db_minor = np.arange(0, range_db + 1, step_minor)
    cbticks_major = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_major]
    cbticks_minor = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_minor]
    cbticklabels = ['0 dB' if att_db == 0 else f'-{att_db} dB' for att_db in att_db_major]
    return cbticks_major, cbticklabels


# TODO priority 1/5 replace remove_mean by detrend, for other types of detrending?

### FFT AND PSD COMPUTATIONS

from functools import wraps

def alias_argument(new_name, old_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if new_name in kwargs:
                    raise TypeError(f"Received both '{new_name}' and '{old_name}'")
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


from .FFT1D import *

from .FFT2D import *


# todo: replace zero_pad by nfft
# todo accept 'auto' as nfft, which selects the nearest superior 5-smooth number

@alias_argument('zpf', 'zero_pad_factor')
def dual(x: floatarray1D, y: floatarray1D, zpf=None):
    if y is None:
        return dual1d(x, zero_pad_factor=zpf)
    return dual2d(x, y=y, zero_pad_factor=zpf)


@alias_argument('zpf', 'zero_pad_factor')

def rdual(x: floatarray1D, zpf=None):
    return rdual1d(x, zero_pad_factor=zpf)


@alias_argument('zpf', 'zero_pad_factor')
def ft(sig, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
       window: str = default_window, winstyle=None, remove_mean: bool = True,
       zpf=None, shift=None):
    if np.ndim(sig) == 2:
        return ft2d(sig, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                    zero_pad_factor=zpf, shift=shift)
    if np.ndim(sig) == 1:
        if y is not None:
            raise RuntimeError('y given but sig is 1-D?')
        if winstyle is not None:
            raise RuntimeError('winstyle given but sig is 1-D?')
        return ft1d(sig, x=x, window=window, remove_mean=remove_mean,
                    zero_pad_factor=zpf, shift=shift)
    raise RuntimeError(f'?? Called ft but sig is {np.ndim(sig)}-dimensional')


def ift(sig_hat: complexarray1D, xdual: Optional[np.ndarray] = None, ydual: Optional[np.ndarray] = None):
    if np.ndim(sig_hat) == 2:
        return ift2d(sig_hat, xdual=xdual, ydual=ydual)
    if np.ndim(sig_hat) == 1:
        if ydual is not None:
            raise RuntimeError('ydual given but sig is 1-D?')
        return ift1d(sig_hat, xdual=xdual)
    raise RuntimeError(f'?? Called ft but sig_hat is {np.ndim(sig_hat)}-dimensional')


@alias_argument('zpf', 'zero_pad_factor')
def rft(sig, x: Optional[np.ndarray] = None,
        window: str = default_window, remove_mean: bool = True,
        zpf=None, shift=None):
    if np.ndim(sig) == 1:
        return rft1d(sig, x=x, window=window, remove_mean=remove_mean,
                     zero_pad_factor=zpf, shift=shift)
    raise RuntimeError(f'?? Called rft but sig is {np.ndim(sig)}-dimensional')


@alias_argument('zpf', 'zero_pad_factor')
def psd(sig, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
        window: str = default_window, winstyle=None, remove_mean: bool = True,
        zpf=None, welch_factor=None):
    if np.ndim(sig) == 2:
        if welch_factor is not None:
            raise NotImplementedError('welch 2d not implemented yet')
        return psd2d(sig, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                     zero_pad_factor=zpf)
    if np.ndim(sig) == 1:
        if y is not None:
            raise RuntimeError('y given but sig is 1-D?')
        if winstyle is not None:
            raise RuntimeError('winstyle given but sig is 1-D?')
        if welch_factor is not None:
            return welch1d(sig, x=x, window=window, remove_mean=remove_mean,
                           zero_pad_factor=zpf, welch_factor=welch_factor)
        return psd1d(sig, x=x, window=window, remove_mean=remove_mean,
                     zero_pad_factor=zpf)
    raise RuntimeError('y given but sig is 1-D?')

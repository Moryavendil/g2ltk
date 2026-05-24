from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math

from g2ltk import customlog
from g2ltk.peakfinder import argval

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

from .FFTHelp import dtype_nfft

from .FFT1D import *

from .FFT2D import *


@alias_argument('zpf', 'zero_pad_factor')
def dual(x: floatarray1D, y: floatarray1D = None, zpf: int = None):
    if y is None:
        return dual1d(x, zpf=zpf)
    return dual2d(x, y=y, zero_pad_factor=zpf)

@alias_argument('zpf', 'zero_pad_factor')
def ft(sig, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
       window: str = default_window, winstyle=None, remove_mean: bool = True,
       zpf=None, shift=None, axes: Optional[Union[int, tuple[int]]] = None):
    if axes is None:
        axes = tuple(range(sig.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    if len(axes) == 1:
        if y is not None:
            raise RuntimeError('y given but sig is 1-D?')
        if winstyle is not None:
            raise RuntimeError('winstyle given but sig is 1-D?')
        return ft1d(sig, x=x, window=window, remove_mean=remove_mean,
                    zpf=zpf, shift=shift, axis=axes[0])
    elif len(axes) == 2:
        return ft2d(sig, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                    zero_pad_factor=zpf, shift=shift)
    else:
        raise NotImplementedError('FT 3D+ not implemented yet')


def ift(sig_hat: complexarray1D, xdual: Optional[np.ndarray] = None, ydual: Optional[np.ndarray] = None,
        axes: Optional[Union[int, tuple[int]]] = None):
    if axes is None:
        axes = tuple(range(sig_hat.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    if len(axes) == 1:
        if ydual is not None:
            raise RuntimeError('ydual given but sig is 1-D?')
        return ift1d(sig_hat, xdual=xdual, axis=axes[0])
    elif len(axes) == 1:
        return ift2d(sig_hat, xdual=xdual, ydual=ydual)
    else:
        raise NotImplementedError('iFT 3D+ not implemented yet')


@alias_argument('zpf', 'zero_pad_factor')
def psd(sig, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
        window: str = default_window, winstyle=None, remove_mean: bool = True,
        zpf=None, welch_factor=None, axes: Optional[Union[int, tuple[int]]] = None):
    if axes is None:
        axes = tuple(range(sig.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    if len(axes) == 1:
        if y is not None:
            raise RuntimeError('y given but sig is 1-D?')
        if winstyle is not None:
            raise RuntimeError('winstyle given but sig is 1-D?')
        if welch_factor is not None:
            return welch1d(sig, x=x, window=window, remove_mean=remove_mean,
                           zpf=zpf, welch_factor=welch_factor, axis=axes[0])
        return psd1d(sig, x=x, window=window, remove_mean=remove_mean,
                     zpf=zpf, axis=axes[0])
    elif len(axes) == 2:
        if welch_factor is not None:
            raise NotImplementedError('welch 2D not implemented yet')
        return psd2d(sig, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                     zero_pad_factor=zpf) # todo add axes here
    else:
        raise NotImplementedError('PSD 3D+ not implemented yet')

### R - transforms

@alias_argument('zpf', 'zero_pad_factor')
def rdual(x: floatarray1D, nfft: dtype_nfft = None, zpf: int = None):
    return rdual1d(x, nfft=nfft, zpf=zpf)

@alias_argument('zpf', 'zero_pad_factor')
def rft(sig, x: Optional[np.ndarray] = None,
        window: str = default_window, remove_mean: bool = True,
        nfft: dtype_nfft = None, zpf=None, shift=None, axis: int = -1):
    return rft1d(sig, x=x, window=window, remove_mean=remove_mean,
                 nfft=nfft, zpf=zpf, shift=shift, axis=axis)

@alias_argument('zpf', 'zero_pad_factor')
def rpsd(sig, x: Optional[np.ndarray] = None, welch_factor=None,
         window: str = default_window, remove_mean: bool = True,
         nfft: dtype_nfft = None, zpf=None, axis: int = -1):
    if welch_factor is not None:
        return rwelch1d(sig, x=x, window=window, remove_mean=remove_mean,
                        nfft=nfft, zpf=zpf, welch_factor=welch_factor, axis=axis)
    return rpsd1d(sig, x=x, window=window, remove_mean=remove_mean,
                  nfft=nfft, zpf=zpf, axis=axis)

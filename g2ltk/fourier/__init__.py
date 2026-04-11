from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import numpy.typing as npt

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

def step(arr: Optional[floatarray1D]) -> float:
    # Returns the spacing between points in a (hopefully) sorted and regularly spaced array
    if arr is None:
        return 1
    return np.diff(arr).mean()


def span(arr: Optional[floatarray1D]) -> float:
    # Returns the span between a (hopefully) sorted array
    # todo: add a warning here. if arr is none, the result will be nonsensical.
    # todo: This is allowed for convenience and fast prototyping, but should raise a warning
    if arr is None:
        return 1
    return (len(arr) - 1) * step(arr)


# find index of value
def argval(arr: np.ndarray, val: Union[int, float]) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
    # Returns the argument of the
    return np.argmin(np.abs(arr - val))



# root finding
def interp_unexplicit_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.abs(np.diff(np.sign(y))).astype(bool) * (y[:-1] != 0)

    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def interp_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Finds the roots of y(x) using linear interpolation

    Parameters
    ----------
    x
        x array
    y
        y(x)

    Returns
    -------
    np.ndarray
        The list of x for which y(x) should be zeros

    """

    return np.unique(np.concatenate((x[y == 0], interp_unexplicit_roots(x, y))))


# peaks (min and max) finding
def dirty_laplacian(x: np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # gives the second derivative of y(x)

    # TODO: implement this properly for uneven spacing. Using gradient(gradient()) ?
    # TODO: USE PATHON PACKAGE findiff https://github.com/maroba/findiff
    ddx = (x[2:] + x[:-2])/2
    ddy = (y[2:] - 2*y[1:-1] + y[:-2])/((x[2:] - x[:-2])/2)**2
    return ddx, ddy

def find_extrema(x: np.ndarray, y: np.ndarray, peak_category: str = 'all'):
    """

    Parameters
    ----------
    x
    y
    peak_category

    Returns
    -------
    If peak_category=='raw' : all the peaks x position, without any sign consideration
    If peak_category=='min' : all the minimums x position
    If peak_category=='max' : all the maximums x position
    If peak_category=='minmax' : minimums, maximums
    else : all, minimums, maximums
    """

    ### FIRST DERIVATIVE
    dy = np.gradient(y, x)

    ### ROOTS OF THE FIRST DERIVATIVE
    roots = interp_roots(x, dy)
    if peak_category == 'raw':
        return roots

    ### SECOND DERIVATIVE for sign
    ddx, ddy = dirty_laplacian(x, y)

    ### INTERPOLATE TO KNOW THE SIGN
    is_max = np.interp(roots, ddx, ddy) < 0
    is_min = np.bitwise_not(is_max)

    if peak_category == 'min':
        return roots[is_min]
    if peak_category == 'max':
        return roots[is_max]
    if peak_category == 'minmax':
        return roots[is_min], roots[is_max]
    return roots, roots[is_min], roots[is_max]


def find_global_peak(x: np.ndarray, y: np.ndarray, peak_category: str = 'raw') -> Optional[float]:
    gross_peak_position = None
    if peak_category == 'raw' or peak_category == 'all' or peak_category == 'minmax':
        peak_category = 'raw'
        gross_peak_position = x[np.abs(y).argmax()]
    elif peak_category == 'max':
        gross_peak_position = x[y.argmax()]
    elif peak_category == 'min':
        gross_peak_position = x[y.argmin()]
    else:
        print(f'Unsupported peak category: {peak_category}')
        return None

    try:
        peaks_positions = find_extrema(x, y, peak_category)
        precise_peak_position = peaks_positions[argval(peaks_positions, gross_peak_position)]
        return precise_peak_position
    except:
        return gross_peak_position


def find_global_max(x: floatarray1D, y: floatarray1D) -> Optional[float]:
    """
    Finds the global maximum of a function y(x).
    It first takes the maximum and then gets a subpixel resolution by taking the point when the derivative vanishes.

    :param x:
    :param y:
    :return:
    """
    return find_global_peak(x, y, 'max')

def correct_limits(arr: floatarray1D) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2


def correct_extent(arr_x: floatarray1D, arr_y: floatarray1D, origin='upper') -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    if origin == 'upper':
        return xlim[0], xlim[1], ylim[1], ylim[0]
    elif origin == 'lower':
        return xlim[0], xlim[1], ylim[0], ylim[1]


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

from .FFT1D import *

from .FFT2D import *

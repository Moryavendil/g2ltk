from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
from findiff import Diff

from g2ltk import customlog

# The names `from g2ltk.arrtools import *` gives (used by g2ltk.peakfinder): add new public functions here
__all__ = ['is_regularly_spaced', 'step', 'span', 'argval', 'interp_unexplicit_roots',
           'interp_roots', 'der', 'find_extrema', 'find_global_peak', 'find_global_max',
           'correct_limits', 'correct_extent', 'gaussian', 'gaussian_unnormalized']

# QOL for manipulating arrays

def is_regularly_spaced(arr: np.ndarray, rtol=1e-5) -> bool:
    """Returns True if the array is increasing and regularly spaced.

    Parameters
    ----------
    arr
    rtol : float, optional
        The relative tolerance. The default is 1e-5.

    Returns
    -------
    bool
        False if the array has less than 2 points, since its spacing is not defined.
    """
    if np.size(arr) < 2:
        return False
    darr = np.diff(arr)
    return np.allclose(darr, darr[0], atol=0, rtol=rtol) and darr[0] > 0


# find the step value
def step(arr: Optional[np.ndarray]) -> float:
    """
    Returns the spacing between points in a (hopefully) sorted and regularly spaced array

    Parameters
    ----------
    arr

    Returns
    -------
    float
        The spacing, 1. if arr is None or has less than 2 points (the spacing is then not defined),
        and the mean spacing if the array is not regularly spaced.
    """
    if arr is None:
        return 1.
    if np.size(arr) < 2:
        customlog.log_error(f'Cannot compute the step of an array of {np.size(arr)} point(s): returning 1.')
        return 1.
    if is_regularly_spaced(arr):
        return float(arr[1] - arr[0])
    darr = np.diff(arr)
    maxspan = darr.max()-darr.min()
    mean = darr.mean()
    if mean == 0:
        customlog.log_warning(f'The array has a mean spacing of 0 (constant array?): its step is 0.')
        return 0.
    # relative to |mean|, so that decreasing arrays are handled like increasing ones
    relative_maxspan = maxspan / abs(mean)
    if relative_maxspan < 1e-3:
        customlog.log_subinfo(f'The array is not exactly regularly spaced (a={maxspan} | r = {relative_maxspan:.6f})')
    else:
        customlog.log_warning(f'The array is not regularly spaced (a={maxspan} | r = {relative_maxspan:.6f})')
    return float(mean)


# finds the span of the array, as used by Fourier transforms
def span(arr: Optional[np.ndarray]) -> float:
    """
    Returns the span of a (hopefully) sorted and regularly spaced array, defined as len(arr) * step(arr).

    This is the span appropriate for Fourier transforms (the duration of a signal sampled at the times arr),
    which is one step more than arr.max() - arr.min(): span([0, 1, 2, 3, 4]) is 5, not 4.

    Parameters
    ----------
    arr
        If None, returns 1. for convenience and fast prototyping, with a warning since the result is then meaningless.

    Returns
    -------
    float
    """
    if arr is None:
        customlog.log_warning('span: the array is None, returning 1. (the result is meaningless)')
        return 1.
    return len(arr) * step(arr)


# find index of value
def argval(arr: np.ndarray, val: Union[int, float]) -> np.intp:
    # Returns the index of the element of arr closest to val
    return np.argmin(np.abs(arr - val))


# root finding
def interp_unexplicit_roots(y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    # Roots of y(x) strictly between two points (sign changes), by linear interpolation
    if x is None:
        x = np.arange(len(y))
    s = np.abs(np.diff(np.sign(y))).astype(bool) * (y[:-1] != 0)

    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def interp_roots(y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Finds the roots of y(x) using linear interpolation

    Parameters
    ----------
    y
        y(x)
    x : optional
        x array. The default is the indices of y.

    Returns
    -------
    np.ndarray
        The sorted x for which y(x) should be zero: the points where y is exactly 0,
        and the sign changes of y, interpolated linearly

    """
    if x is None:
        x = np.arange(len(y))

    return np.unique(np.concatenate((x[y == 0], interp_unexplicit_roots(y, x=x))))


# Finite difference

def der(y: np.ndarray, x: Optional[Union[np.ndarray, float]] = None, order: int = 1, acc: Optional[int] = None,
        axis: int = 0, periodic: bool = False):
    """
    Computes the order-th derivative of y(x).
    This is essentially a wrapper for findiff / np.gradient, adapted for quick and dirty data treatment

    Parameters
    ----------
    y
    x : optional
        Can be dx (grid spacing) or the grid directly. The default is a grid spacing of 1.
    order : int, optional
        The order of the derivative. The default is 1 (simple derivative).
    acc : int, optional
        The required accuracy, in powers of dx. Must be a positive even integer. The default is 2 (second order accurate schemes).
    axis
        The axis over which to compute the derivative. The default is 0.
    periodic : bool, optional
         Whether the axis is to be treated as periodic. The default is False.

    Returns
    -------

    """
    if x is None:
        x = 1.
    if acc is None and not periodic and order == 1:
        return np.gradient(y, x, axis=axis)
    if np.ndim(x) == 1 and is_regularly_spaced(x):
        x = x[1] - x[0]
    if acc is None:
        acc = 2
    diffoperator = Diff(axis=axis, grid=x, acc=acc, periodic=periodic)
    if order > 1:
        diffoperator = diffoperator**order
    return diffoperator(y)


def find_extrema(y: np.ndarray, x: Optional[np.ndarray] = None, peak_category: str = 'all'):
    """
    Finds the extrema of y(x), with subpixel resolution: the roots of its first derivative, interpolated linearly.
    The sign of the second derivative at each root tells minimums from maximums.

    Parameters
    ----------
    y
        y(x)
    x : optional
        x array. The default is the indices of y.
    peak_category : str, optional
        Which extrema to return (see Returns). The default is 'all'.

    Returns
    -------
    If peak_category=='raw' : all the peaks x position, without any sign consideration
    If peak_category=='min' : all the minimums x position
    If peak_category=='max' : all the maximums x position
    If peak_category=='minmax' : minimums, maximums
    else : all, minimums, maximums
    """
    if x is None:
        x = np.arange(len(y))

    ### FIRST DERIVATIVE
    dy = np.gradient(y, x)

    ### ROOTS OF THE FIRST DERIVATIVE
    roots = interp_roots(dy, x=x)
    if peak_category == 'raw':
        return roots

    ### SECOND DERIVATIVE for sign
    ddy = der(y, x, order=2)

    ### INTERPOLATE TO KNOW THE SIGN
    is_max = np.interp(roots, x, ddy) < 0
    is_min = np.bitwise_not(is_max)

    if peak_category == 'min':
        return roots[is_min]
    if peak_category == 'max':
        return roots[is_max]
    if peak_category == 'minmax':
        return roots[is_min], roots[is_max]
    return roots, roots[is_min], roots[is_max]


def find_global_peak(y: np.ndarray, x: Optional[np.ndarray] = None, peak_category: str = 'raw') -> Optional[float]:
    """
    Finds the position of the global peak of y(x), with subpixel resolution.

    It first takes the point of y with the largest value (of |y|, y or -y depending on peak_category), then refines its
    position by taking the closest extremum found by find_extrema (where the derivative vanishes).
    If no extremum can be found, the position of the point itself is returned.

    Parameters
    ----------
    y
        y(x)
    x : optional
        x array. The default is the indices of y.
    peak_category : str, optional
        'max' : the global maximum of y.
        'min' : the global minimum of y.
        'raw' : the global maximum of |y|, refined with any extremum (minimum or maximum). The default.
        'all' and 'minmax' are treated as 'raw'.

    Returns
    -------
    float or None
        The x position of the peak, or None if peak_category is not supported.
    """
    if x is None:
        x = np.arange(len(y))
    gross_peak_position = None
    if peak_category == 'raw' or peak_category == 'all' or peak_category == 'minmax':
        peak_category = 'raw'
        gross_peak_position = x[np.abs(y).argmax()]
    elif peak_category == 'max':
        gross_peak_position = x[y.argmax()]
    elif peak_category == 'min':
        gross_peak_position = x[y.argmin()]
    else:
        customlog.log_error(f'find_global_peak: unsupported peak category: {peak_category}')
        return None

    try:
        peaks_positions = find_extrema(y, x=x, peak_category=peak_category)
        precise_peak_position = peaks_positions[argval(peaks_positions, gross_peak_position)]
        return precise_peak_position
    except (ValueError, IndexError):
        # no extremum found (ValueError), or too few points to find one (IndexError)
        return gross_peak_position


def find_global_max(y: np.ndarray, x: Optional[np.ndarray] = None) -> Optional[float]:
    """
    Finds the global maximum of a function y(x).
    It first takes the maximum and then gets a subpixel resolution by taking the point when the derivative vanishes.

    Parameters
    ----------
    y
        y(x)
    x : optional
        x array. The default is the indices of y.

    Returns
    -------
    float
        The x position of the maximum.
    """
    return find_global_peak(y, x, 'max')


def correct_limits(arr: np.ndarray) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2


def correct_extent(arr_x: np.ndarray, arr_y: np.ndarray, origin='upper') -> Optional[Tuple[float, float, float, float]]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    if origin == 'upper':
        return xlim[0], xlim[1], ylim[1], ylim[0]
    elif origin == 'lower':
        return xlim[0], xlim[1], ylim[0], ylim[1]
    customlog.log_error(f"correct_extent: unsupported origin {origin!r} (use 'upper' or 'lower')")
    return None


# gaussian stuff
def gaussian(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x - mu)/sigma)**2)

def gaussian_unnormalized(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return np.exp(-1/2*((x - mu)/sigma)**2)

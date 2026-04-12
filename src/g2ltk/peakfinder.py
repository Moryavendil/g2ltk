from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
from findiff import Diff
import math

from . import set_verbose, log_error, log_warn, log_warning, log_info, log_subinfo, log_debug, log_trace, log_subtrace


# QOL for manipulating arrays

def is_regularly_spaced(arr: np.ndarray, tol=1e-6) -> bool:
    darr = np.diff(arr)
    return np.allclose(darr, darr[0], atol=0, rtol=tol) and darr[0] > 0


# find the step value
def step(arr: Optional[np.ndarray]) -> float:
    # Returns the spacing between points in a (hopefully) sorted and regularly spaced array
    if arr is None:
        return 1
    if is_regularly_spaced(arr):
        return arr[1] - arr[0]
    log_warning('NOT REGULARLY SPACED ARRAY!')
    return np.diff(arr).mean()


# finds the max-min value
def span(arr: Optional[np.ndarray]) -> float:
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


# Finite difference

def der(y: np.ndarray, x: Optional[Union[np.ndarray, float]] = None, order: int = 1, acc: Optional[int] = None,
        axis: int = 0, periodic: bool = False):
    """
    Computes the order-th derivative of y(x).
    This is essentially a wrapper for findiff, adapted for quick and dirty data treatment
    WARNING CURRENTLY ONLY WORKS ON 1D

    :param y:
    :param grid:
    :param order:
    :param acc:
    :param axis:
    :return:
    """
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

# peaks (min and max) finding
def dirty_laplacian(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # gives the second derivative of y(x)

    # TODO: implement this properly for uneven spacing. Using gradient(gradient()) ?
    # TODO: USE PATHON PACKAGE findiff https://github.com/maroba/findiff
    ddx = (x[2:] + x[:-2]) / 2
    ddy = (y[2:] - 2 * y[1:-1] + y[:-2]) / ((x[2:] - x[:-2]) / 2) ** 2
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


def find_global_max(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Finds the global maximum of a function y(x).
    It first takes the maximum and then gets a subpixel resolution by taking the point when the derivative vanishes.

    :param x:
    :param y:
    :return:
    """
    return find_global_peak(x, y, 'max')


def correct_limits(arr: np.ndarray) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2


def correct_extent(arr_x: np.ndarray, arr_y: np.ndarray, origin='upper') -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    if origin == 'upper':
        return xlim[0], xlim[1], ylim[1], ylim[0]
    elif origin == 'lower':
        return xlim[0], xlim[1], ylim[0], ylim[1]


# gaussian stuff
def gaussian(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x - mu)/sigma)**2)

def gaussian_unnormalized(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return np.exp(-1/2*((x - mu)/sigma)**2)

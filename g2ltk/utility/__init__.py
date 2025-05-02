from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np

from .. import set_verbose, log_error, log_warn, log_warning, log_info, log_subinfo, log_debug, log_trace, log_subtrace

###### MATHS

import math
from scipy.signal import savgol_filter # for smoothing
from scipy.interpolate import CubicSpline # for cubic interpolation

# derivating
def der1(x: np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # gives the derivative of y(x)
    dx = (x[1:] + x[:-1])/2
    dy = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    return dx, dy

def der2(x: np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # gives the derivative of y(x)
    dx = (x[2:] + x[:-2])/2
    dy = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    return dx, dy

def lap(x: np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # gives the second derivative of y(x)
    dx = (x[2:] + x[:-2])/2
    dy = (y[2:] - 2*y[1:-1] + y[:-2])/((x[2:] - x[:-2])/2)**2
    return dx, dy

# root finding
def find_unexplicit_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.abs( np.diff(np.sign(y)) ).astype(bool) * (y[:-1] != 0)

    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s] / y[:-1][s])+1)

def find_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    return np.unique(np.concatenate( (x[y == 0],
                                      find_unexplicit_roots(x,y)) ))

# peaks (min and max) finding
def find_extrema(x: np.ndarray, y: np.ndarray, peak_category: str = 'all', smooth_derivative=False):
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
    # dx, dy = der1(x, y)
    dx, dy = der2(x, y)

    ### ROOTS OF THE FIRST DERIVATIVE
    roots = find_roots(dx, dy)
    if peak_category == 'raw':
        return roots

    ### SMOOTH TO DERIVATE
    if smooth_derivative:
        dy = savgol_filter(dy, 25, 3)

    ### SECOND DERIVATIVE for sign
    ddx, ddy = lap(x, y)

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
        peaks_positions = find_extrema(x, y, peak_category, smooth_derivative=False)
        precise_peak_position = peaks_positions[np.abs(peaks_positions - gross_peak_position).argmin()]
    except:
        precise_peak_position = gross_peak_position
    return precise_peak_position

def find_global_max(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Finds the global maximum of a function y(x).
    It first takes the maximum and then gets a subpixel resolution by taking the point when the derivative vanishes.

    :param x:
    :param y:
    :return:
    """
    return find_global_peak(x, y, 'max')

def find_inflexion_point(x: np.ndarray, y: np.ndarray):
    dx, dy = der1(x, y)
    return find_global_peak(dx, dy)

# smooting
def smooth_1D(x: np.ndarray, y: np.ndarray, sigma:float=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies gaussian

    :param x:
    :param y:
    :param sigma:
    :return:
    """
    #Todo: THIS FOR LOOP IN PYTHON IS AWFULY SLOW !!!!!!
    x_, y_ = x.copy(), np.empty_like(y)
    for i_x in range(len(x)):
        w = gaussian_unnormalized(x, mu = x[i_x], sigma=sigma)
        y_[i_x] = np.average(y, weights=w)
    return x_, y_

def interp_cubic(x, xp, yp):
    return CubicSpline(xp, yp)(x)

# gaussian stuff
def gaussian(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x - mu)/sigma)**2)

def gaussian_unnormalized(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
    # 1 D gaussian
    return np.exp(-1/2*((x - mu)/sigma)**2)

def double_gauss(x, x1, a1, w1, x2, a2, w2, bckgnd_noise):
    return bckgnd_noise + a1 * gaussian_unnormalized(x, x1, w1) + a2 * gaussian_unnormalized(x, x2, w2)

# normalization
def normalize(y: Any):
    y_normalized = y.copy()
    y_normalized -= np.mean(y_normalized)
    y_normalized /= np.max(np.abs(y_normalized))
    return y_normalized

### ARRAYS

def step(arr:Optional[np.ndarray]) -> float:
    if arr is None:
        return 1
    # return arr[1] - arr[0]
    return (arr[1:] - arr[:-1]).mean()

def span(arr:Optional[np.ndarray]) -> float:
    if arr is None:
        return 1
    return (len(arr)-1)*step(arr)

def correct_limits(arr:np.ndarray) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2
def correct_extent(arr_x:np.ndarray, arr_y:np.ndarray, origin='upper') -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    if origin=='upper':
        return xlim[0], xlim[1], ylim[1], ylim[0]
    elif origin=='lower':
        return xlim[0], xlim[1], ylim[0], ylim[1]

### log
def attenuate_power(value, attenuation_factor_dB):
    return value / math.pow(10, attenuation_factor_dB / 20)

def log_amplitude_range(maximum_amplitude:float, range_db:Union[float, int]):
    return maximum_amplitude, attenuate_power(maximum_amplitude, range_db)

def log_amplitude_cbticks(maximum_amplitude:float, range_db:Union[int, float]):
    step_major = 20
    step_minor = 5
    if range_db < 60:
        step_major = 10
        step_minor = 2
    if range_db < 30:
        step_major = 5
        step_minor = 1
    # it seems unreasonable to have range_db > 100 or < 10
    att_db_major = np.arange(0, range_db+1, step_major)
    att_db_minor = np.arange(0, range_db+1, step_minor)
    cbticks_major = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_major]
    cbticks_minor = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_minor]
    cbticklabels = ['0 dB' if att_db == 0 else f'-{att_db} dB' for att_db in att_db_major]
    return cbticks_major, cbticklabels

def set_ticks_log_cb(cb, maximum_amplitude:float, range_db:Union[int, float], text:bool=True):
    step_major = 20
    step_minor = 5
    if range_db < 60:
        step_major = 10
        step_minor = 2
    if range_db < 30:
        step_major = 5
        step_minor = 1
    # it seems unreasonable to have range_db > 100 or < 10
    att_db_major = np.arange(0, range_db+1, step_major)
    att_db_minor = np.arange(0, range_db+1, step_minor)
    cbticks_major = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_major]
    cbticklabels = ['0 dB' if att_db == 0 else f'-{att_db} dB' for att_db in att_db_major]
    cbticks_minor = [attenuate_power(maximum_amplitude, att_db) for att_db in att_db_minor]

    cb.ax.set_yticks(cbticks_major, minor=False)
    cb.ax.set_yticklabels(cbticklabels if text else [], minor=False)
    cb.ax.set_yticks(cbticks_minor, minor=True)
    cb.ax.set_yticklabels([], minor=True)

### FFT AND PSD COMPUTATIONS

from .fourier import *

########## SAVE GRAPHE & FIG SIZE

from .genfig import *

from .plotting import *

########### DISPLAYS THE TIME
def convert_time(time:Any, origin_unit:str, target_unit:str):
    pass

def format_videotime(t_s:float, finaltime_s:Optional[float]=None) -> str:
    if finaltime_s is None:
        finaltime_s = t_s
    s:float = t_s
    ms:float = s*1000
    us:float = (ms - np.floor(ms))*1000
    seconds:int = np.floor(s).astype(int)
    milliseconds:int = np.floor(ms).astype(int) - 1000*seconds
    microseconds:int = np.floor(us).astype(int) - 1000*(milliseconds + 1000*seconds)
    if finaltime_s > 5:
        return f'{s:.3f} s'
    else:
        return f'{ms:.3f} ms'

def disptime(t:float) -> str:
    """Formats the time, adapted for waiting times"""
    if np.abs(t) < 1:return str(round(t, 3))+' s'
    if np.abs(t) < 5:return str(round(t, 1))+' s'
    s = int(t)
    if np.abs(s) < 60:return str(s)+' s'
    m = s // 60
    s = s % 60
    if np.abs(m) < 10:return str(m)+' m '+str(s)+' s'
    if np.abs(m) < 60:return str(m)+' m '
    h = m // 60
    m = m % 60
    if np.abs(h) < 6:return str(h)+' h '+str(m)+' m'
    if np.abs(h) < 24:return str(h)+' h '
    d = h // 24
    h = h % 24
    if np.abs(d) < 3:return str(d)+' d '+str(h)+' h'
    if np.abs(d) < 30:return str(d)+' d'
    m = d // 30
    d = d % 30
    if np.abs(m) < 3:return str(m)+' m '+str(d)+' d'
    if np.abs(m) < 12:return str(m)+' m'
    y = m // 12
    m = m % 12
    if np.abs(y) < 3:return str(y)+' y '+str(m)+' m'
    return str(y)+' years'


############# RIVULET
def w_from_borders(borders:np.ndarray) -> np.ndarray:
    return borders[:,1,:] - borders[:,0,:] # for some historical reason the borders are weirdly arranged lol

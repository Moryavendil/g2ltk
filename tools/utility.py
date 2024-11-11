from typing import Optional, Any, Tuple, Dict, List
import numpy as np


###### MATHS

import math
from scipy.signal import savgol_filter # for smoothing
from scipy.interpolate import CubicSpline # for cubic interpolation

# misc
def lin(x: Any, a: float, b: float) -> Any:
    return a * x + b

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
    #Todo: determine if it is better to use der1 or der2 here.
    # der1 is noisier ?

    # dx, dy = der1(x, y)
    dx, dy = der2(x, y)

    ### ROOTS OF THE FIRST DERIVATIVE
    roots = find_roots(dx, dy)
    if peak_category == 'raw':
        return roots

    ### SMOOTH TO DERIVATE
    if smooth_derivative:
        dy = savgol_filter(dy, 25, 3)

    ### SECOND DERIVATIVE
    #Todo: determine if it is better to use der1 or der2 here.
    # der1 is noisier ?

    ddx, ddy = der1(dx, dy)
    # ddx, ddy = der2(dx, dy)

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

### FFT AND ARRAYS

def step(arr:np.ndarray) -> float:
    if arr is None:
        return 1
    # return arr[1] - arr[0]
    return (arr[1:] - arr[:-1]).mean()

def span(arr:np.ndarray) -> float:
    return (len(arr)-1)*step(arr)

def correct_limits(arr:np.ndarray) -> Tuple[float, float]:
    return arr.min() - step(arr) / 2, arr.max() + step(arr) / 2
def correct_extent_spatio(arr_x:np.ndarray, arr_y:np.ndarray) -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    return xlim[0], xlim[1], ylim[1], ylim[0]
def correct_extent_fft(arr_x:np.ndarray, arr_y:np.ndarray) -> Tuple[float, float, float, float]:
    xlim = correct_limits(arr_x)
    ylim = correct_limits(arr_y)
    return xlim[0], xlim[1], ylim[0], ylim[1]

def dual(arr:np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(len(arr), step(arr)))

def rdual(arr:np.ndarray) -> np.ndarray:
    return np.fft.rfftfreq(len(arr), step(arr))

from scipy.signal.windows import get_window # FFT windowing

def ft1d(arr:np.ndarray, window:str= 'hann', norm=None) -> np.ndarray:
    """

    Parameters
    ----------
    arr
    window
    norm

    Returns
    -------

    """
    N = len(arr)
    z_treated = arr * get_window(window, N)
    # z_treated -= np.mean(z_treated) * (1-1e-12) # this is to avoid having zero amplitude and problems when taking the log
    z_treated -= np.mean(z_treated)

    z_hat = np.fft.rfft(z_treated, norm=norm)
    return z_hat

def ft2d(arr:np.ndarray, window:str='hann', norm=None) -> np.ndarray:
    Nt, Nx = arr.shape
    z_treated = arr * np.expand_dims(get_window(window, Nt), axis=1) * np.expand_dims(get_window(window, Nx), axis=0)
    # z_treated -= np.mean(z_treated) * (1-1e-12) # this is to avoid having zero amplitude and problems when taking the log
    z_treated -= np.mean(z_treated)

    z_hat = np.fft.rfft2(z_treated, norm=norm)
    return np.fft.fftshift(z_hat, axes=0)
    # return np.concatenate((z_hat[(Nt+1)//2:,:], z_hat[:(Nt+1)//2,:])) # reorder bcz of the FT

def window_factor(window:str):
    """
    Returns the factor by which the energy is multiplied when the signal is windowed

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
        return 8/3
    else:
        return 1/((get_window(window, 1000)**2).sum()/1000)

def psd1d(y, x1, window:str= 'hann') -> np.ndarray:
    z_ft = ft1d(y, window=window, norm="backward") * step(x1) # x dt for units
    # energy spectral density: energy of the signal at this frequency
    # useful for time-limited signals (impulsions)
    esd = np.abs(z_ft)**2 * window_factor(window) * 2 # x 2 because of rfft which truncates the spectrum
    esd[1:] *= 2 # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    # psd = esd / T
    return esd / span(x1)

def psd2d(y, x1, x2, window:str='hann') -> np.ndarray:
    y_ft = ft2d(y, window=window, norm="backward") * step(x1) * step(x2) # x dt for units
    # energy spectral density: energy of the signal at this frequency
    # useful for time-limited signals (impulsions)
    esd = np.abs(y_ft)**2 * window_factor(window)**2
    esd[:, 1:] *= 2 # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    # psd = esd / (T X)
    return esd / span(x1) / span(x2)

# find the edges of the peak
def attenuate_power(value, attenuation_factor_dB):
    return value / math.pow(10, attenuation_factor_dB / 20)
def peak_contour1d(peak_x1, z, peak_depth_dB, x1=None):
    if x1 is None:
        x1 = np.arange(z.shape[0])
    peak_index = np.argmin((x1-peak_x1)**2)
    zintercept = attenuate_power(z[peak_index], peak_depth_dB)
    x1_intercept = find_roots(x1, z - zintercept)
    x1_before = x1[x1_intercept < peak_x1].max()
    x1_after = x1[x1_intercept > peak_x1].min()
    return x1_before, x1_after
def peak_vicinity1d(peak_x1, z, peak_depth_dB, x1=None):
    if x1 is None:
        x1 = np.arange(z.shape[0])
    x1_before, x1_after = peak_contour1d(peak_x1=peak_x1, z=z, peak_depth_dB=peak_depth_dB, x1=x1)
    # return np.where((x1 >= x1_before)*(x1 <= x1_after))[0]
    return ((x1 >= x1_before)*(x1 <= x1_after)).astype(bool)

def power_near_peak1d(peak_x1, z, peak_depth_dB, x1=None):
    # powerlog_intercmor_incertitude = zmeanx_psd.max()/(10**(peak_depth_dB/10))
    # freq_for_intercept = utility.find_roots(freqs, zmeanx_psd - powerlog_intercmor_incertitude)
    # freqpre = freq_for_intercept[freq_for_intercept < freq_guess].max()
    # freqpost = freq_for_intercept[freq_for_intercept > freq_guess].min()
    # integrate the PSD along the peak
    # ### (abandoned) do a trapezoid integration
    # x_f = np.concatenate(([freqpre], freqs[(freqpre < freqs)*(freqs < freqpost)], [freqpost]))
    # y_f = np.concatenate(([np.interp(freqpre, freqs, zmeanx_psd)], zmeanx_psd[(freqpre < freqs)*(freqs < freqpost)], [np.interp(freqpost, freqs, zmeanx_psd)]))
    # p_ft_peak = trapezoid(y_f, x_f)
    ### Go bourrin (we are in log we do not care) : rectangular integration
    # p_ft_peak = np.sum(zmeanx_psd[(freqpre < freqs)*(freqs < freqpost)]) * utility.step(freqs)
    return np.sum(z[peak_vicinity1d(peak_x1=peak_x1, z=z, peak_depth_dB=peak_depth_dB, x1=x1)]) * step(x1)

from contourpy import contour_generator, ZInterp
from skimage.measure import points_in_poly, grid_points_in_poly

def peak_contour2d(peak_x1, peak_x2, z, peak_depth_dB, x1=None, x2=None):
    if x1 is None:
        x1 = np.arange(z.shape[1])
    if x2 is None:
        x2 = np.arange(z.shape[0])

    # to find the contour, we use contourpy which is fast, efficient and has the log-interpolation option that is relevant for us
    cg = contour_generator(x=x1, y=x2, z=z, z_interp=ZInterp.Log)
    zpeak = z[np.argmin((x2-peak_x2)**2)][np.argmin((x1-peak_x1)**2)]
    zintercept = attenuate_power(zpeak, peak_depth_dB)
    contours = cg.lines(zintercept)

    interestpoints = [[peak_x1, peak_x2]]
    if np.isclose(peak_x1, 0):
        interestpoints.append([peak_x1, -peak_x2])

    # find the contour that contains the point
    pointincontour = np.array([np.sum(points_in_poly(interestpoints, contour)) for contour in contours]).astype(bool)
    maincontours = [contours[index] for index in np.where(pointincontour == True)[0]]

    return maincontours

def peak_vicinity2d(peak_x1, peak_x2, z, peak_depth_dB, x1=None, x2=None):
    maincontours = peak_contour2d(peak_x1=peak_x1, peak_x2=peak_x2, z=z, peak_depth_dB=peak_depth_dB, x1=x1, x2=x2)

    # make a mask with the points inside the contourS
    mask = np.zeros_like(z).astype(bool)
    for contour in maincontours:
        # we want to use skimage's grid_points_in_poly but it works with integer coordinates
        # so we have to translate
        contour_gridcoord = contour.copy()
        if x1 is not None:
            contour_gridcoord[:,0] -= x1.min()
            contour_gridcoord[:,0] /= step(x1)
        if x2 is not None:
            contour_gridcoord[:,1] -= x2.min()
            contour_gridcoord[:,1] /= step(x2)
        # not sure why we do the transpose thingamabob but it works
        contour_mask = grid_points_in_poly(z.transpose().shape, contour_gridcoord).transpose()
        mask = np.bitwise_or(mask, contour_mask)

    return mask
def power_near_peak2d(peak_x1, peak_x2, z, peak_depth_dB, x1=None, x2=None):
    return np.sum(z[peak_vicinity2d(peak_x1=peak_x1, peak_x2=peak_x2, z=z, peak_depth_dB=peak_depth_dB, x1=x1, x2=x2)]) * step(x1) * step(x2)

############# RIVULET
def w_form_borders(borders:np.ndarray) -> np.ndarray:
    return borders[:,1,:] - borders[:,0,:] # for some historical reason the borders are weirdly arranged lol

########## SAVE GRAPHE
import os
import matplotlib.pyplot as plt
def save_graphe(graph_name, imageonly=False, **kwargs):
    figures_directory = 'figures'
    if not os.path.isdir(figures_directory):
        os.mkdir(figures_directory)
    raw_path = os.path.join(figures_directory, graph_name)
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'
    if 'pad_inches' not in kwargs:
        kwargs['pad_inches'] = 0
    if imageonly:
        plt.savefig(raw_path + '.jpg', **kwargs)
    else:
        if 'dpi' not in kwargs:
            kwargs['dpi'] = 600
        plt.savefig(raw_path + '.png', **kwargs)
        plt.savefig(raw_path + '.pdf', **kwargs)
        plt.savefig(raw_path + '.svg', **kwargs)

########### DISPLAYS THE TIME
def convert_time(time:Any, origin_unit:str, target_unit:str):
    pass
def disptime(t: float) -> str:
    if np.abs(t) < 1:return str(round(t, 2))+' s'
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




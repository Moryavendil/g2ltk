from numpy import ndarray, dtype
from typing import Optional, Any, Tuple, Dict, List
import numpy as np


###### MATHS

import math
from scipy.signal import savgol_filter # for smoothing
from scipy.interpolate import CubicSpline # for cubic interpolation

from tools import display, log_warn, log_info, log_debug, log_trace, log_retrace


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
    # der1 is noisier ? der2 should be better

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

### FFT AND PSD COMPUTATIONS

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

def psd1d(z, x, window:str= 'hann') -> np.ndarray:
    z_ft = ft1d(z, window=window, norm="backward") * step(x) # x dt for units
    # energy spectral density: energy of the signal at this frequency
    # useful for time-limited signals (impulsions)
    esd = np.abs(z_ft)**2 * window_factor(window)
    esd[1:] *= 2 # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    # psd = esd / T
    return esd / span(x)

def psd2d(z, x, y, window:str= 'hann') -> np.ndarray:
    y_ft = ft2d(z, window=window, norm="backward") * step(x) * step(y) # x dt for units
    # energy spectral density: energy of the signal at this frequency
    # useful for time-limited signals (impulsions)
    esd = np.abs(y_ft)**2 * window_factor(window)**2
    esd[:, 1:] *= 2 # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    # psd = esd / (T X)
    return esd / span(x) / span(y)

# find the edges of the peak (1D, everything is easy)
def attenuate_power(value, attenuation_factor_dB):
    return value / math.pow(10, attenuation_factor_dB / 20)
def peak_contour1d(peak_x, z, peak_depth_dB, x=None):
    if x is None:
        x = np.arange(z.shape[0])
    peak_index = np.argmin((x - peak_x) ** 2)
    zintercept = attenuate_power(z[peak_index], peak_depth_dB)
    x1_intercept = find_roots(x, z - zintercept)
    x1_before = peak_x
    x1_after = peak_x
    if len(x1_intercept[x1_intercept < peak_x] > 0):
        x1_before = x1_intercept[x1_intercept < peak_x].max()
    if len(x1_intercept[x1_intercept > peak_x] > 0):
        x1_after = x1_intercept[x1_intercept > peak_x].min()
    return x1_before, x1_after
def peak_vicinity1d(peak_x, z, peak_depth_dB, x=None):
    if x is None:
        x = np.arange(z.shape[0])
    x1_before, x1_after = peak_contour1d(peak_x=peak_x, z=z, peak_depth_dB=peak_depth_dB, x=x)
    # return np.where((x1 >= x1_before)*(x1 <= x1_after))[0]
    return ((x >= x1_before) * (x <= x1_after)).astype(bool)

def power_near_peak1d(peak_x, z, peak_depth_dB, x=None):
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
    return np.sum(z[peak_vicinity1d(peak_x=peak_x, z=z, peak_depth_dB=peak_depth_dB, x=x)]) * step(x)

### 2D peak finding (life is pain)
# TODO: Simplify this. We use multipolygons BUT it seems that every multipolygon we have does in fact contain only one polygon
# TODO: Which is dumb. We might be able to change that just by setting multipolygon = multipolygon.geoms[0] but honestly who has time for that ?

from contourpy import contour_generator, ZInterp, convert_filled
# from contourpy import convert_multi_filled # todo: works when contourpy >= 1.3, we have 1.2

### SHAPELY SHENANIGANS

from shapely import GeometryType, from_ragged_array, Point, LinearRing

def find_shapely_contours(contourgenerator, zintercept):
    contours = contourgenerator.filled(zintercept, np.inf)
    polygons = [([contours[0][i]], [contours[1][i]]) for i in range(len(contours[0]))]
    # polygons_for_shapely = convert_multi_filled(polygons, cg.fill_type, "ChunkCombinedOffsetOffset") # todo: works when contourpy >= 1.3, we have 1.2
    polygons_for_shapely = [convert_filled(polygon, contourgenerator.fill_type, "ChunkCombinedOffsetOffset") for polygon in polygons]

    log_trace(f'Found {len(polygons)} contours')

    multipolygons = []
    for i_poly in range(len(polygons_for_shapely)):
        points, offsets, outer_offsets = polygons_for_shapely[i_poly][0][0], polygons_for_shapely[i_poly][1][0], polygons_for_shapely[i_poly][2][0]
        multipolygon = from_ragged_array(GeometryType.MULTIPOLYGON, points, (offsets, outer_offsets, [0, len(outer_offsets)-1]))[0]
        # multipolygon = from_ragged_array(GeometryType.POLYGON, points, (offsets, outer_offsets)) # try to do a shapely Polygon is better if we really need to handle the holes
        multipolygons.append(multipolygon)
    return multipolygons
def contour_containspoint(contour, point:Tuple[float, float]) -> bool:
    """Returns whether or not the points are contained in the contour."""
    return contour.contains(Point(point[0], point[1]))
def contour_containspoints(contour, points:List[Tuple[float, float]]) -> np.ndarray:
    """Returns whether or not the points are contained in the contour."""
    return np.array([contour.contains(Point(point[0], point[1])) for point in points]).astype(bool)
def contours_containsanyofthepoints(contours, points:List[Tuple[float, float]]) -> np.ndarray:
    """Returns wheter any of the points are contained in the contours."""
    return np.array([contour_containspoints(contour, points).any() for contour in contours])
def contours_areas(contours, scale_factor_x, scale_factor_y, condition:Optional[np.ndarray[bool]]=None) -> np.ndarray:
    """Returns the area of a contour, in px² if the scale factors are right."""
    if condition is None:
        condition = np.ones(len(contours), dtype=bool)
    areas = np.ones(len(contours), dtype=float)
    for i, contour in enumerate(contours):
        if condition[i]:
            areas[i] = contour.area/scale_factor_x/scale_factor_y
    return areas
def contours_perimeters(contours, scale_factor_x, scale_factor_y, condition:Optional[np.ndarray[bool]]=None):
    """Returns the perimeter of a contour, in px if the scale factors are right."""
    log_trace(f'Perimeters of {len(contours)} contours')
    if condition is None:
        condition = np.ones(len(contours), dtype=bool)
    perimeters = np.ones(len(contours), dtype=float)
    for i, contour in enumerate(contours):
        if condition[i]:
            log_retrace(f'contour {i}: type: {contour.geom_type}')
            log_retrace(f'contour {i}: contained geometries: {len(contour.geoms)}')
            hull = contour.geoms[0]
            log_retrace(f'contour {i}: hull is : {hull.geom_type}')
            log_retrace(f'contour {i}: hull boundary is : {hull.boundary.geom_type}')
            log_retrace(f'contour {i}: hull exterior is : {hull.exterior.geom_type}')
            log_retrace(f'contour {i}: hull interior contains {len(hull.interiors)} rings')
            all_boundaries = [hull.exterior] + [interior for interior in hull.interiors]
            p = 0
            for boundary in all_boundaries:
                line = np.array(boundary.coords)
                line[:, 0] /= scale_factor_x
                line[:, 1] /= scale_factor_y
                p += LinearRing(line).length
            perimeters[i] = p
    return perimeters
def draw_multipolygon_edge(ax, multipolygon, xmin=None, **kwargs):
    for geom in multipolygon.geoms:
        log_retrace(f'geom is : {geom.geom_type}')
        log_retrace(f'geom boundary is : {geom.boundary.geom_type}')
        log_retrace(f'geom exterior is : {geom.exterior.geom_type}')
        log_retrace(f'geom interior contains {len(geom.interiors)} rings')
        all_boundaries = [geom.exterior] + [interior for interior in geom.interiors]
        for boundary in all_boundaries:
            line = np.array(boundary.coords)
            if xmin is not None:
                line[:, 0][line[:, 0] < xmin] = xmin
            ax.plot(line[:,0], line[:,1], **kwargs)


### CONTOUR FINDING 2D AND PEAK MEASUREMENT
peak_max_area_default = 100
peak_min_circularity_default = .3

def peak_contour2d(peak_x:float, peak_y:float, z:np.ndarray, peak_depth_dB:float, x:Optional[np.ndarray[float]]=None, y:Optional[np.ndarray[float]]=None,
                   fastmode:bool=True, peak_max_area:Optional[float]=None, peak_min_circularity:Optional[float]=None):
    """Finds a contour around a peak at a certain threshold level

    Parameters
    ----------
    peak_x
    peak_y
    z
    peak_depth_dB
    x
    y
    fastmode : do not compute aeras and perimeters of contour that do not contain the peak
    peak_max_area : dmaximum area, in px²
    peak_min_circularity : minimum circularity ([0-1]), for reasonable contours

    Returns
    -------

    """
    if x is None:
        x = np.arange(z.shape[1])
    if y is None:
        y = np.arange(z.shape[0])
    if peak_max_area is None:
        global peak_max_area_default
        peak_max_area = peak_max_area_default
    if peak_min_circularity is None:
        global peak_min_circularity_default
        peak_min_circularity = peak_min_circularity_default

    log_debug(f'Searching for a contour around ({round(peak_x, 3)}, {round(peak_y, 3)}) with attenuation -{peak_depth_dB} dB')

    interestpoints = [[peak_x, peak_y]]
    if np.isclose(peak_x, 0):
        interestpoints.append([peak_x, -peak_y])
    zpeak = z[np.argmin((y - peak_y) ** 2)][np.argmin((x - peak_x) ** 2)]

    # duplicate the first column to better find the points which are at k=0
    x_for_cg = np.concatenate(([x[0]-step(x)], x))
    y_for_cg = y.copy()
    z_for_cg = np.zeros((z.shape[0], z.shape[1]+1))
    z_for_cg[:, 1:] = z
    z_for_cg[:, 0] = z[:, 0]
    # to find the contour, we use contourpy which is fast, efficient and has the log-interpolation option that is relevant for us
    cg = contour_generator(x=x_for_cg, y=y_for_cg, z=z_for_cg, z_interp=ZInterp.Log, fill_type="OuterOffset")

    min_peak_depth_dB = 10

    while peak_depth_dB > 0: # infinite loop, return is in it
        zintercept = attenuate_power(zpeak, peak_depth_dB)

        multipolygons = find_shapely_contours(cg, zintercept)

        containspeak = contours_containsanyofthepoints(multipolygons, interestpoints)
        areas = contours_areas(multipolygons, step(x), step(y), condition=containspeak if fastmode else None)
        perimeters = contours_perimeters(multipolygons, step(x), step(y), condition=containspeak if fastmode else None)
        circularities = 4*np.pi*areas / perimeters**2

        for i in range(len(multipolygons)):
            if containspeak[i] or not fastmode:
                log_retrace(f'multipoligon {i}: centroid: {multipolygons[i].centroid}')
                log_retrace(f'multipoligon {i}: area: {round(areas[i], 1)} px^2, limit is {peak_max_area}')
                log_retrace(f'multipoligon {i}: perimeter: {round(perimeters[i], 1)} px')
                log_retrace(f'multipoligon {i}: circularity: {round(circularities[i], 3)} [0-1], limit is {peak_min_circularity}')

        isnottoobig = areas < peak_max_area
        iscircularenough = circularities > peak_min_circularity

        contour_is_valid = containspeak * isnottoobig * iscircularenough

        log_trace(f'Contains peak: {containspeak if not fastmode else containspeak[containspeak]}')
        log_trace(f'Not too big:   {isnottoobig if not fastmode else isnottoobig[containspeak]}')
        log_trace(f'Circular:      {iscircularenough if not fastmode else iscircularenough[containspeak]}')
        log_trace(f'Valid contour: {contour_is_valid if not fastmode else contour_is_valid[containspeak]}')

        # find the contour that contains the point
        if contour_is_valid.any() or peak_depth_dB == min_peak_depth_dB:
            maincontours = [multipolygons[index] for index in np.where(contour_is_valid == True)[0]]
            log_debug(f"Found {len(maincontours)} valid contours.")
            return maincontours
        else:
            peak_depth_dB -= 10
            if peak_depth_dB < min_peak_depth_dB: peak_depth_dB = min_peak_depth_dB
            log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")
    if peak_depth_dB != min_peak_depth_dB:
        peak_depth_dB = min_peak_depth_dB
        log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")

def grid_points_in_contour(contour, x:np.ndarray[float], y:np.ndarray[float]) -> ndarray:
    incontour = np.zeros((len(y), len(x)), dtype=bool)
    xmin, ymin, xmax, ymax = contour.bounds
    for i_x in np.where((x >= xmin)*(x <= xmax))[0]:
        for i_y in np.where((y >= ymin)*(y <= ymax))[0]:
            incontour[i_y, i_x] = contour_containspoint(contour, (x[i_x], y[i_y]))
    return incontour

def peak_vicinity2d(peak_x, peak_y, z:np.ndarray[np.ndarray[float]], peak_depth_dB, x=Optional[np.ndarray[float]], y=Optional[np.ndarray[float]],
                    peak_contours:Optional[List]=None,
                    peak_max_area:Optional[float]=None, peak_min_circularity:Optional[float]=None):
    if x is None:
        x = np.arange(z.shape[1])
    if y is None:
        y = np.arange(z.shape[0])
    log_debug(f'Searching for the vicinity of  ({round(peak_x, 3)}, {round(peak_y, 3)})')

    if peak_contours is None:
        peak_contours = peak_contour2d(peak_x=peak_x, peak_y=peak_y, z=z, peak_depth_dB=peak_depth_dB, x=x, y=y,
                                       peak_max_area=peak_max_area, peak_min_circularity=peak_min_circularity)

    # make a mask with the points inside the contourS
    mask = np.zeros_like(z).astype(bool)
    # take at least the interest point
    mask[np.argmin((y - peak_y) ** 2)][np.argmin((x - peak_x) ** 2)] = True
    for contour in peak_contours:
        contour_mask = grid_points_in_contour(contour, x, y)

        mask = np.bitwise_or(mask, contour_mask)

    log_debug(f'Found a vicinity of area {mask.sum()} px²')

    return mask
def power_near_peak2d(peak_x, peak_y, z, peak_depth_dB, x=None, y=None,
                      peak_contours:Optional[List]=None, peak_vicinity:Optional[np.ndarray]=None,
                      peak_max_area:Optional[float]=None, peak_min_circularity:Optional[float]=None):
    log_debug(f'Measuring the power around     ({round(peak_x, 3)}, {round(peak_y, 3)})')
    if peak_vicinity is None:
        peak_vicinity = peak_vicinity2d(peak_x=peak_x, peak_y=peak_y, z=z, peak_depth_dB=peak_depth_dB, x=x, y=y,
                                        peak_contours=peak_contours,
                                        peak_max_area=peak_max_area, peak_min_circularity=peak_min_circularity)
    pw = np.sum(z[peak_vicinity]) * step(x) * step(y)
    log_debug(f'Power: {pw} (amplitude: {np.sqrt(pw)*np.sqrt(2)})')
    return pw

############# RIVULET
def w_from_borders(borders:np.ndarray) -> np.ndarray:
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




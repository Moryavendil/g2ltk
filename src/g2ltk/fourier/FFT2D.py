from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import math
from scipy.signal.windows import get_window  # FFT windowing
from skimage import filters  # filters.window for 2D FFT windowing
from scipy import fft

from .. import log_error, log_warning, log_info, log_debug, log_trace, log_subtrace
from g2ltk.peakfinder import step, span
from . import floatarray1D, floatarray2D, complexarray1D, complexarray2D, attenuate_power
from .FFT1D import dual1d, rdual1d

default_window = 'boxcar'

### Dual: changing from real space to frequency space
def dual2d(x: floatarray1D, y: floatarray1D,
           zero_pad: Optional[Tuple[int, int]] = None,
           zero_pad_factor: Optional[Union[int, Tuple[int, int]]] = None) -> Tuple[floatarray1D, floatarray1D]:
    log_trace(f'dual2d: Called with {x.shape}x{y.shape} arrays, zp = {zero_pad}, zpf = {zero_pad_factor}')
    zero_pad_x, zero_pad_y = None, None
    if zero_pad is not None:
        try:
            int(zero_pad[0])
            int(zero_pad[1])
            zero_pad_y, zero_pad_x = zero_pad
        except:
            log_warning(f'What is this zero-padding "{zero_pad}" ? I made it None')
    zero_pad_factor_x, zero_pad_factor_y = None, None
    if zero_pad_factor is not None:
        try:
            float(zero_pad_factor[0])
            float(zero_pad_factor[1])
            log_trace('dual2d: zero_pad_factor seems to be a Tuple')
            zero_pad_factor_y, zero_pad_factor_x = zero_pad_factor
        except:
            log_trace('dual2d: zero_pad_factor seems to NOT be a Tuple')
            try:
                float(zero_pad_factor)
                log_trace('dual2d: zero_pad_factor seems to be a number')
                zero_pad_factor_y, zero_pad_factor_x = zero_pad_factor, zero_pad_factor
            except:
                log_warning(f'dual2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
    return (dual1d(x, zero_pad=zero_pad_x, zero_pad_factor=zero_pad_factor_x),
            dual1d(y, zero_pad=zero_pad_y, zero_pad_factor=zero_pad_factor_y))


def rdual2d(x: floatarray1D, y: floatarray1D,
            zero_pad: Optional[Tuple[int, int]] = None,
            zero_pad_factor: Optional[Union[int, Tuple[int, int]]] = None) -> Tuple[floatarray1D, floatarray1D]:
    log_trace(f'rdual2d: Called with {x.shape}x{y.shape} arrays, zp = {zero_pad}, zpf = {zero_pad_factor}')
    zero_pad_x, zero_pad_y = None, None
    if zero_pad is not None:
        try:
            int(zero_pad[0])
            int(zero_pad[1])
            zero_pad_y, zero_pad_x = zero_pad
        except:
            log_warning(f'dual2d: What is this zero-padding "{zero_pad}" ? I made it None')
    zero_pad_factor_x, zero_pad_factor_y = None, None
    if zero_pad_factor is not None:
        try:
            float(zero_pad_factor[0])
            float(zero_pad_factor[1])
            zero_pad_factor_y, zero_pad_factor_x = zero_pad_factor
        except:
            try:
                float(zero_pad_factor)
                zero_pad_factor_y, zero_pad_factor_x = zero_pad_factor, zero_pad_factor
            except:
                log_warning(f'dual2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
    return (rdual1d(x, zero_pad=zero_pad_x, zero_pad_factor=zero_pad_factor_x),
            dual1d(y, zero_pad=zero_pad_y, zero_pad_factor=zero_pad_factor_y))

### FT: computing the Fourier Transform

def prepare_signal_for_ft2d(arr: complexarray2D,
                            window: str = default_window, winstyle=None, remove_mean: bool = True,
                            zero_pad: Optional[Tuple[int, int]] = None,
                            zero_pad_factor: Optional[Union[int, Tuple[int, int]]] = None,
                            shift: Optional[Tuple[int, int]] = None) -> complexarray2D:
    """
    Returns the 2-D Fourier transform of a real input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
    y
    window
    winstyle
        Can be 'outer' (default) or 'circular'
    norm
    zero_pad
    zero_pad_factor
        Use powers of 2

    Returns
    -------

    """
    log_trace(f'ft2d: Preparing array of shape {arr.shape} to 2-D FFT')
    Nt, Nx = arr.shape

    # Removing mean
    z_nozero = arr - np.mean(arr) * remove_mean

    if winstyle is None:
        winstyle = 'outer'
    winstyle = str(winstyle)
    log_subtrace(f'ft2d: Using windowing | window={window} | style={winstyle}')
    if winstyle == 'outer':
        z_win = z_nozero * np.expand_dims(get_window(window, Nt), axis=1) * np.expand_dims(get_window(window, Nx), axis=0)
    elif winstyle == 'circular':
        z_win = z_nozero * filters.window(window, (Nt, Nx), warp_kwargs={'order': 3})
    else:
        log_warning(f'Unrecognized 2d-windowing style: {winstyle}. Using no windowing at all.')
        z_win = z_nozero

    pad_width = None
    if zero_pad_factor is not None:
        log_subtrace(f'ft2d: | zero_pad_factor={zero_pad_factor}')
        try:
            pad_t = np.rint(Nt * (zero_pad_factor[0] - 1)).astype(int)
            pad_x = np.rint(Nx * (zero_pad_factor[1] - 1)).astype(int)
            pad_width = ((pad_t // 2, pad_t // 2 + pad_t % 2), (pad_x // 2, pad_x // 2 + pad_x % 2))
        except:
            # case zero_pad_factor is an int
            try:
                pad_t = np.rint(Nt * (zero_pad_factor - 1)).astype(int)
                pad_x = np.rint(Nx * (zero_pad_factor - 1)).astype(int)
                pad_width = ((pad_t // 2, pad_t // 2 + pad_t % 2), (pad_x // 2, pad_x // 2 + pad_x % 2))
            except:
                log_warning(f'ft2d: What is this zero-padding factor "{zero_pad_factor}" ? I made it None')
                pad_width = None
    elif zero_pad is not None:
        log_subtrace(f'ft2d: | zero_pad={zero_pad}')
        try:
            pad_width = ((0, int(zero_pad[0])), (0, int(zero_pad[1])))
        except:
            log_warning(f'ft2d: What is this zero-padding "{zero_pad}" ? I made it None')
            pad_width = None

    log_subtrace(f'ft2d: Padding (artificially better resolution) | pad={pad_width}')
    if pad_width is None:
        pad_width = ((0, 0), (0, 0))
    z_pad = np.pad(z_win, pad_width=pad_width, mode='constant', constant_values=0)

    log_subtrace(f'ft2d: Removing (0,0)-freq component: {True}')
    # z_clean -= np.mean(z_win) * (1-1e-12) # this is to avoid having zero amplitude and problems when taking the log
    z_clean = z_pad - np.mean(z_pad) * remove_mean

    # log_subtrace(f'rft2d: Rolling (restoring phase) | pad={pad_width}')
    # z_roll = np.roll(z_clean, (pad_width[0][0]+Nt//2, pad_width[1][0]+Nx//2), axis = (0, 1))
    roll_offset = (pad_width[0][0] + Nt // 2, pad_width[1][0] + Nx // 2)

    roll_shift = None
    if shift is not None:
        try:
            roll_shift = (int(shift[0]), int(shift[1]))
        except:
            log_warning(f'ft2d: What is this shift "{roll_shift}" ? I made it None')
            roll_shift = None
    if roll_shift is None:
        roll_shift = (0, 0)
    roll_total = (roll_offset[0] + roll_shift[0], roll_offset[1] + roll_shift[1])
    log_subtrace(f'ft2d: Rolling (restoring phase) | roll={roll_total}')
    log_subtrace(f'ft2d: roll_offset={roll_offset} | roll_shift={roll_shift}')
    z_roll = np.roll(z_clean, roll_total, axis=(0, 1))

    log_debug(f'ft2d: Old size ({Nt},{Nx}) | New size={z_roll.shape}')

    return z_roll


def ft2d(arr: complexarray2D, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
         window: str = default_window, winstyle=None, remove_mean: bool = True, norm=None,
         zero_pad: Optional[Tuple[int, int]] = None, zero_pad_factor: Optional[Tuple[float, float]] = None,
         shift: Optional[Tuple[int, int]] = None) -> complexarray2D:
    """
    Returns the 2-D Fourier transform of a real input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
    y
    window
    norm
    zero_pad
    zero_pad_factor
        Use powers of 2

    Returns
    -------

    """
    log_trace(f'ft2d: Computing 2-D FFT of array of shape {arr.shape}')

    Nt_arr, Nx_arr = arr.shape
    Nx_x = len(x) if x is not None else None
    Nt_t = len(y) if y is not None else None
    if ((Nx_x is not None) and (Nx_x != Nx_arr)) or ((Nt_t is not None) and (Nt_t != Nt_arr)):
        log_warning(f'ft2d: array is shaped {arr.shape} != {Nt_t} x {Nx_x}')

    arr_prepared = prepare_signal_for_ft2d(arr, window=window, winstyle=winstyle, remove_mean=remove_mean,
                                           zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                                           shift=shift)

    log_subtrace(f'ft2d: Computing fft, norm={norm} | shape={arr_prepared.shape}')
    arr_hat = fft.fft2(arr_prepared, norm=norm, s=arr_prepared.shape)
    return fft.fftshift(arr_hat) * step(x) * step(y)


def rft2d(arr: floatarray2D, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
          window: str = default_window, winstyle=None, remove_mean: bool = True, norm=None,
          zero_pad: Optional[Tuple[int, int]] = None, zero_pad_factor: Optional[Tuple[float, float]] = None,
          shift: Optional[Tuple[int, int]] = None) -> complexarray2D:
    """
    Returns the 2-D Fourier transform of a real input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    winstyle
    shift
    arr
    x
    y
    window
    norm
    zero_pad
    zero_pad_factor
        Use powers of 2

    Returns
    -------

    """
    log_trace(f'rft2d: Computing 2-D RFFT of array of shape {arr.shape}')

    Nt_arr, Nx_arr = arr.shape
    Nx_x = len(x) if x is not None else None
    Nt_t = len(y) if y is not None else None
    if ((Nx_x is not None) and (Nx_x != Nx_arr)) or ((Nt_t is not None) and (Nt_t != Nt_arr)):
        log_warning(f'ft2d: array is shaped {arr.shape} != {Nt_t} x {Nx_x}')

    arr_prepared = prepare_signal_for_ft2d(arr, window=window, winstyle=winstyle, remove_mean=remove_mean,
                                           zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                                           shift=shift)

    log_subtrace(f'rft2d: Computing fft, norm={norm} | shape={arr_prepared.shape}')
    arr_hat = fft.rfft2(arr_prepared, norm=norm, s=arr_prepared.shape)
    return fft.fftshift(arr_hat, axes=0) * step(x) * step(y)

### Power:
from.FFT1D import window_factor1d

def window_factor2d(window: str, winstyle: Optional[str] = None):
    """
    Returns the factor by which the energy is multiplied when the signal is windowed.

    Parameters
    ----------
    window

    Returns
    -------
    :param winstyle:

    """
    if winstyle is None:
        winstyle = 'outer'
    winstyle = str(winstyle)
    log_subtrace(f'ft2d: Using windowing | window={window} | style={winstyle}')
    if winstyle == 'outer':
        return window_factor1d(window)**2
    elif winstyle == 'circular':
        N = 1000
        return 1 / (np.sum(filters.window(window, (N, N), warp_kwargs={'order': 3})**2)/N**2)
    else:
        log_warning(f'Unrecognized 2d-windowing style: {winstyle}.')
        return 1


def psd2d(z: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
          window: str = default_window, winstyle=None, remove_mean: bool = True,
          zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None,
          shift: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Returns the 2-D Fourier PSD of a real input array using the given windowing.

    quantitative ensures the real thing is computed. if False, the first column is mul

    if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/(Hz.mm^{-1})(Hz, mm-1)]

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
    y
    window
    zero_pad
    zero_pad_factor
        Use powers of 2

    Returns
    -------

    """
    log_trace('rpsd2d: Computing a 2-D PSD')
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $\hat{z}$ is [V/(Hz.mm^{-1})(Hz, mm-1)]
    y_ft = ft2d(z, x=x, y=y, window=window, winstyle=winstyle,
                remove_mean=remove_mean, norm="backward",
                zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                shift=shift)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the oly thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this 2-frequency
    # It is useful in itself for space-time-limited signals (wavelets)
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $ESD(z)$ is [V^2/Hz^2/mm^{-2}(Hz, mm-1)]
    esd = np.abs(y_ft) ** 2 * window_factor2d(window, winstyle=winstyle) ** 2
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is 2-D periodic, then PSD = ESD / (duration_1.duration_2)
    # Thus if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/Hz/mm^{-1}(Hz, mm-1)]
    return esd / span(x) / span(y)

def rpsd2d(z: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
           window: str = default_window, winstyle=None, remove_mean: bool = True,
           zero_pad: Optional[int] = None, zero_pad_factor: Optional[float] = None,
           shift: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Returns the 2-D Fourier PSD of a real input array using the given windowing.

    quantitative ensures the real thing is computed. if False, the first column is mul

    if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/(Hz.mm^{-1})(Hz, mm-1)]

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
    y
    window
    zero_pad
    zero_pad_factor
        Use powers of 2

    Returns
    -------

    """
    log_trace('rpsd2d: Computing a 2-D PSD')
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $\hat{z}$ is [V/(Hz.mm^{-1})(Hz, mm-1)]
    y_ft = rft2d(z, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean, norm="backward",
                 zero_pad=zero_pad, zero_pad_factor=zero_pad_factor,
                 shift=shift)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the oly thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this 2-frequency
    # It is useful in itself for space-time-limited signals (wavelets)
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $ESD(z)$ is [V^2/Hz^2/mm^{-2}(Hz, mm-1)]
    esd = np.abs(y_ft) ** 2 * window_factor2d(window, winstyle=winstyle) ** 2
    esd[:, 1:] *= 2  # x 2 because of rfft which truncates the spectrum (except the 0 harmonic)
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is 2-D periodic, then PSD = ESD / (duration_1.duration_2)
    # Thus if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/Hz/mm^{-1}(Hz, mm-1)]
    return esd / span(x) / span(y)


### 2D peak finding (life is pain)
# TODO: Simplify this. We use multipolygons BUT it seems that every multipolygon we have does in fact contain only one polygon
# TODO: Which is dumb. We might be able to change that just by setting multipolygon = multipolygon.geoms[0] but honestly who has time for that ?

from contourpy import contour_generator, ZInterp, convert_filled
# from contourpy import convert_multi_filled # todo: works when contourpy >= 1.3, sadly we have 1.2 ...

### SHAPELY SHENANIGANS

from shapely import GeometryType, from_ragged_array, Point, LinearRing


def find_shapely_contours(contourgenerator, zintercept):
    contours = contourgenerator.filled(zintercept, np.inf)
    polygons = [([contours[0][i]], [contours[1][i]]) for i in range(len(contours[0]))]
    # polygons_for_shapely = convert_multi_filled(polygons, cg.fill_type, "ChunkCombinedOffsetOffset") # todo: works when contourpy >= 1.3, we have 1.2
    polygons_for_shapely = [convert_filled(polygon, contourgenerator.fill_type, "ChunkCombinedOffsetOffset") for polygon
                            in polygons]

    log_trace(f'Found {len(polygons)} contours')

    multipolygons = []
    for i_poly in range(len(polygons_for_shapely)):
        points, offsets, outer_offsets = polygons_for_shapely[i_poly][0][0], polygons_for_shapely[i_poly][1][0], \
            polygons_for_shapely[i_poly][2][0]
        multipolygon = \
            from_ragged_array(GeometryType.MULTIPOLYGON, points, (offsets, outer_offsets, [0, len(outer_offsets) - 1]))[
                0]
        # multipolygon = from_ragged_array(GeometryType.POLYGON, points, (offsets, outer_offsets)) # try to do a shapely Polygon is better if we really need to handle the holes
        multipolygons.append(multipolygon)
    return multipolygons


def contour_containspoint(contour, point: Tuple[float, float]) -> bool:
    """Returns whether or not the points are contained in the contour."""
    return contour.contains(Point(point[0], point[1]))


def contour_containspoints(contour, points: List[Tuple[float, float]]) -> np.ndarray:
    """Returns whether or not the points are contained in the contour."""
    return np.array([contour.contains(Point(point[0], point[1])) for point in points]).astype(bool)


def contours_containsanyofthepoints(contours, points: List[Tuple[float, float]]) -> np.ndarray:
    """Returns wheter any of the points are contained in the contours."""
    return np.array([contour_containspoints(contour, points).any() for contour in contours])


def contours_areas(contours, scale_factor_x, scale_factor_y,
                   condition: Optional[np.ndarray[bool]] = None) -> np.ndarray:
    """Returns the area of a contour, in px² if the scale factors are right."""
    if condition is None:
        condition = np.ones(len(contours), dtype=bool)
    areas = np.ones(len(contours), dtype=float)
    for i, contour in enumerate(contours):
        if condition[i]:
            areas[i] = contour.area / scale_factor_x / scale_factor_y
    return areas


def contours_perimeters(contours, scale_factor_x, scale_factor_y, condition: Optional[np.ndarray[bool]] = None):
    """Returns the perimeter of a contour, in px if the scale factors are right."""
    log_trace(f'Perimeters of {len(contours)} contours')
    if condition is None:
        condition = np.ones(len(contours), dtype=bool)
    perimeters = np.ones(len(contours), dtype=float)
    for i, contour in enumerate(contours):
        if condition[i]:
            log_subtrace(f'contour {i}: type: {contour.geom_type}')
            log_subtrace(f'contour {i}: contained geometries: {len(contour.geoms)}')
            hull = contour.geoms[0]
            log_subtrace(f'contour {i}: hull is : {hull.geom_type}')
            log_subtrace(f'contour {i}: hull boundary is : {hull.boundary.geom_type}')
            log_subtrace(f'contour {i}: hull exterior is : {hull.exterior.geom_type}')
            log_subtrace(f'contour {i}: hull interior contains {len(hull.interiors)} rings')
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
        log_subtrace(f'geom is : {geom.geom_type}')
        log_subtrace(f'geom boundary is : {geom.boundary.geom_type}')
        log_subtrace(f'geom exterior is : {geom.exterior.geom_type}')
        log_subtrace(f'geom interior contains {len(geom.interiors)} rings')
        all_boundaries = [geom.exterior] + [interior for interior in geom.interiors]
        for boundary in all_boundaries:
            line = np.array(boundary.coords)
            if xmin is not None:
                line[:, 0][line[:, 0] < xmin] = xmin
            ax.plot(line[:, 0], line[:, 1], **kwargs)


### CONTOUR FINDING 2D AND PEAK MEASUREMENT
peak_max_area_default = 100
peak_min_circularity_default = .3


def peak_contour2d(peak_x: float, peak_y: float, z: np.ndarray, peak_depth_dB: float,
                   x: Optional[np.ndarray[float]] = None, y: Optional[np.ndarray[float]] = None,
                   fastmode: bool = True, peak_max_area: Optional[float] = None,
                   peak_min_circularity: Optional[float] = None,
                   truncated_spectrum: bool = False, real_signal: bool = True):
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
    if len(z.shape) != 2:
        log_warning(f'peak_contour2d: z should be a 2D array. Its shape is {z.shape}')
    if x is None:
        x = np.arange(z.shape[1])
    else:
        if len(x) != z.shape[1]:
            log_warning(f'peak_contour2d: Shapes do not match. {z.shape} != {y.shape}x{x.shape}')
    if y is None:
        y = np.arange(z.shape[0])
        if len(y) != z.shape[0]:
            log_warning(f'peak_contour2d: Shapes do not match. {z.shape} != {y.shape}x{x.shape}')
    if peak_max_area is None:
        global peak_max_area_default
        peak_max_area = peak_max_area_default
    if peak_min_circularity is None:
        global peak_min_circularity_default
        peak_min_circularity = peak_min_circularity_default

    log_debug(
        f'Searching for a contour around ({round(peak_x, 3)}, {round(peak_y, 3)}) with attenuation -{peak_depth_dB} dB')

    interestpoints = [[peak_x, peak_y]]
    if real_signal and truncated_spectrum and np.isclose(peak_x, 0):
        interestpoints.append([peak_x, -peak_y])
    if real_signal and not truncated_spectrum:
        interestpoints.append([-peak_x, -peak_y])  # Since we are interested in real signals (we are doing physics)...
    zpeak = z[np.argmin((y - peak_y) ** 2)][np.argmin((x - peak_x) ** 2)]

    # duplicate the first column to better find the points which are at k=0
    x_for_cg = x
    y_for_cg = y
    z_for_cg = z
    if truncated_spectrum:
        x_for_cg = np.concatenate(([x[0] - step(x)], x))
        z_for_cg = np.zeros((z.shape[0], z.shape[1] + 1))
        z_for_cg[:, 1:] = z
        z_for_cg[:, 0] = z[:, 0]
    # to find the contour, we use contourpy which is fast, efficient and has the log-interpolation option that is relevant for us
    cg = contour_generator(x=x_for_cg, y=y_for_cg, z=z_for_cg, z_interp=ZInterp.Log, fill_type="OuterOffset")

    min_peak_depth_dB = 10

    while peak_depth_dB > 0:  # infinite loop, return is in it
        zintercept = attenuate_power(zpeak, peak_depth_dB)

        multipolygons = find_shapely_contours(cg, zintercept)

        containspeak = contours_containsanyofthepoints(multipolygons, interestpoints)
        areas = contours_areas(multipolygons, step(x), step(y), condition=containspeak if fastmode else None)
        perimeters = contours_perimeters(multipolygons, step(x), step(y), condition=containspeak if fastmode else None)
        circularities = 4 * np.pi * areas / perimeters ** 2

        for i in range(len(multipolygons)):
            if containspeak[i] or not fastmode:
                log_subtrace(f'multipoligon {i}: centroid: {multipolygons[i].centroid}')
                log_subtrace(f'multipoligon {i}: area: {round(areas[i], 1)} px^2, limit is {peak_max_area}')
                log_subtrace(f'multipoligon {i}: perimeter: {round(perimeters[i], 1)} px')
                log_subtrace(
                    f'multipoligon {i}: circularity: {round(circularities[i], 3)} [0-1], limit is {peak_min_circularity}')

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
            peak_depth_dB -= 5
            if peak_depth_dB < min_peak_depth_dB: peak_depth_dB = min_peak_depth_dB
            log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")
    if peak_depth_dB != min_peak_depth_dB:
        peak_depth_dB = min_peak_depth_dB
        log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")


def grid_points_in_contour(contour, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray:
    incontour = np.zeros((len(y), len(x)), dtype=bool)
    xmin, ymin, xmax, ymax = contour.bounds
    for i_x in np.where((x >= xmin) * (x <= xmax))[0]:
        for i_y in np.where((y >= ymin) * (y <= ymax))[0]:
            incontour[i_y, i_x] = contour_containspoint(contour, (x[i_x], y[i_y]))
    return incontour


def peak_vicinity2d(peak_x, peak_y, z: np.ndarray[np.ndarray[float]], peak_depth_dB, x=Optional[np.ndarray[float]],
                    y=Optional[np.ndarray[float]],
                    peak_contours: Optional[List] = None,
                    peak_max_area: Optional[float] = None, peak_min_circularity: Optional[float] = None):
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
                      peak_contours: Optional[List] = None, peak_vicinity: Optional[np.ndarray] = None,
                      peak_max_area: Optional[float] = None, peak_min_circularity: Optional[float] = None):
    log_debug(f'Measuring the power around     ({round(peak_x, 3)}, {round(peak_y, 3)})')
    if peak_vicinity is None:
        peak_vicinity = peak_vicinity2d(peak_x=peak_x, peak_y=peak_y, z=z, peak_depth_dB=peak_depth_dB, x=x, y=y,
                                        peak_contours=peak_contours,
                                        peak_max_area=peak_max_area, peak_min_circularity=peak_min_circularity)
    pw = np.sum(z[peak_vicinity]) * step(x) * step(y)
    log_debug(f'Power: {pw} (amplitude: {np.sqrt(2*pw)})')
    return pw

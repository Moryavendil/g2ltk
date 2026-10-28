from typing import Optional, Any, Tuple, Dict, List, Union, Literal
import numpy as np
from scipy.signal.windows import get_window  # FFT windowing
from skimage import filters  # filters.window for 2D FFT windowing
from scipy import fft

from g2ltk import customlog
from g2ltk.peakfinder import step, span
from . import floatarray1D, floatarray2D, complexarray2D, attenuate_power, alias_argument
from .FFTHelp import sanitize_nfft_2d, sanitize_zpf_2d, dtype_nfft_2d, dtype_zpf_2d, dtype_shift_2d, split_2d_argument
from .FFT1D import dual1d, rdual1d

default_window = 'boxcar'

# todo fix directions, (axis0, axis 1) must correspond to x, y...
# todo because for now x and y are interverted?!
# todo or keep it idc but make it generlizable
# todo like axis n, n-1, 3,2, 1, 0  = ?, ?, ..., ?, z y x

### Arguments handling
# In 2D, nfft, zpf and shift can be given either as a single value (used for both axes)
# or as a pair (value_axis0, value_axis1).
# The arrays are shaped (len(y), len(x)): axis 0 corresponds to y, axis 1 corresponds to x.


### Dual: changing from real space to frequency space
@alias_argument('zpf', 'zero_pad_factor')
def dual2d(x: floatarray1D, y: floatarray1D,
           nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None) -> Tuple[floatarray1D, floatarray1D]:
    """
    Returns the duals of x and y, i.e. change from real space to frequency space (or vice versa).

    Parameters
    ----------
    x : np.ndarray
        Coordinates along axis 1 of the 2D array
    y : np.ndarray
        Coordinates along axis 0 of the 2D array
    nfft : int, 'auto' or (axis0, axis1), optional
        The size of the nfft
    zpf : int or (int, int), optional
        The zero padding factor. An alias is ```zero_pad_factor```

    Returns
    -------
    (x_dual, y_dual)
    """
    fname = 'dual2d'
    customlog.log_trace(f'{fname}: Called with {x.shape}x{y.shape} arrays, nfft = {nfft}, zpf = {zpf}')

    nfft_y, nfft_x = split_2d_argument(nfft, fname)
    zpf_y, zpf_x = split_2d_argument(zpf, fname)

    return (dual1d(x, nfft=nfft_x, zpf=zpf_x),
            dual1d(y, nfft=nfft_y, zpf=zpf_y))


@alias_argument('zpf', 'zero_pad_factor')
def rdual2d(x: floatarray1D, y: floatarray1D,
            nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None) -> Tuple[floatarray1D, floatarray1D]:
    """
    Returns the duals of x and y, in a scipy.fft.rfft2-compatible style (x is truncated, y is not).

    Parameters
    ----------
    x : np.ndarray
        Coordinates along axis 1 of the 2D array
    y : np.ndarray
        Coordinates along axis 0 of the 2D array
    nfft : int, 'auto' or (axis0, axis1), optional
        The size of the nfft
    zpf : int or (int, int), optional
        The zero padding factor. An alias is ```zero_pad_factor```

    Returns
    -------
    (x_dual, y_dual)
    """
    fname = 'rdual2d'
    customlog.log_trace(f'{fname}: Called with {x.shape}x{y.shape} arrays, nfft = {nfft}, zpf = {zpf}')

    nfft_y, nfft_x = split_2d_argument(nfft, fname)
    zpf_y, zpf_x = split_2d_argument(zpf, fname)

    return (rdual1d(x, nfft=nfft_x, zpf=zpf_x),
            dual1d(y, nfft=nfft_y, zpf=zpf_y))


### FT: computing the Fourier Transform

@alias_argument('zpf', 'zero_pad_factor')
def prepare_signal_for_ft2d(arr: complexarray2D,
                            window: str = default_window, winstyle: Optional[str] = None, remove_mean: bool = True,
                            nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None,
                            shift: dtype_shift_2d = None) -> complexarray2D:
    """
    Applies mean removal, windowing, padding and phase shift.

    Parameters
    ----------
    arr
    window : str, optional
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    remove_mean : bool, optional
        Default is True
    nfft : int, 'auto' or (axis0, axis1), optional
    zpf : int or (int, int), optional
    shift : int or (int, int), optional

    Returns
    -------

    """
    fname = 'prepare_signal_for_ft2d'
    customlog.log_trace(f'{fname}: Called with {arr.shape} array, nfft = {nfft}, zpf = {zpf}')

    N0, N1 = arr.shape

    # Removing mean
    customlog.log_trace(f'{fname}: Mean removal: {remove_mean}')
    arr_nozero = arr - np.mean(arr) * remove_mean

    # windowing
    if winstyle is None:
        winstyle = 'outer'
    winstyle = str(winstyle)
    customlog.log_subtrace(f'{fname}: Using windowing | window={window} | style={winstyle}')
    if winstyle == 'outer':
        arr_windowed = arr_nozero * np.expand_dims(get_window(window, N0), axis=1) \
                       * np.expand_dims(get_window(window, N1), axis=0)
    elif winstyle == 'circular':
        arr_windowed = arr_nozero * filters.window(window, (N0, N1), warp_kwargs={'order': 3})
    else:
        customlog.log_warning(f'{fname}: Unrecognized 2d-windowing style: {winstyle}. Using no windowing at all.')
        arr_windowed = arr_nozero

    # padding
    nfft0, nfft1 = sanitize_nfft_2d(nfft, (N0, N1))
    zpf0, zpf1 = sanitize_zpf_2d(zpf)
    nfft_final = (nfft0 * zpf0, nfft1 * zpf1)

    customlog.log_debug(f'{fname}: ({N0}, {N1}) -> {nfft_final} (({nfft0}, {nfft1}) x ({zpf0}, {zpf1}))')

    # truncate along the axes where nfft_final is smaller than the signal, zero-pad (at the end) the others
    arr_cut = arr_windowed[:nfft_final[0], :nfft_final[1]]
    pad_width = ((0, nfft_final[0] - arr_cut.shape[0]), (0, nfft_final[1] - arr_cut.shape[1]))
    arr_padded = np.pad(arr_cut, pad_width=pad_width, mode='constant', constant_values=0)

    # second mean removal. It is unorthodox. Is it necessary? most often, not, it is useless
    # arr_padded = arr_padded - np.mean(arr_padded) * remove_mean

    shift0, shift1 = split_2d_argument(shift, fname)
    shift0 = 0 if shift0 is None else int(shift0)
    shift1 = 0 if shift1 is None else int(shift1)
    roll = (-pad_width[0][0] - shift0, -pad_width[1][0] - shift1)
    customlog.log_subtrace(f'{fname}: Rolling (restoring phase) by pre_pad=({pad_width[0][0]}, {pad_width[1][0]})'
                           f' + shift=({shift0}, {shift1})')
    arr_rolled = np.roll(arr_padded, roll, axis=(0, 1))

    return arr_rolled


def _check_shapes_2d(fname: str, arr: np.ndarray, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> None:
    N0, N1 = arr.shape
    Nx = len(x) if x is not None else None
    Ny = len(y) if y is not None else None
    if ((Nx is not None) and (Nx != N1)) or ((Ny is not None) and (Ny != N0)):
        customlog.log_warning(f'{fname}: array is shaped {arr.shape} != {Ny} x {Nx}')


@alias_argument('zpf', 'zero_pad_factor')
def ft2d(arr: complexarray2D, x: Optional[floatarray1D] = None, y: Optional[floatarray1D] = None,
         window: str = default_window, winstyle: Optional[str] = None, remove_mean: bool = True,
         nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None,
         shift: dtype_shift_2d = None, norm: Optional[str] = 'backward') -> complexarray2D:
    """
    Returns the 2-D Fourier transform of the input array using the given windowing.

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
        Coordinates along axis 1
    y
        Coordinates along axis 0
    window : str, optional
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    remove_mean : bool, optional
        Default is True
    nfft : int, 'auto' or (axis0, axis1), optional
    zpf : int or (int, int), optional
    shift : int or (int, int), optional
    norm : str, optional
        The scipy.fft normalization. Keep 'backward' (default) for the units to be right.

    Returns
    -------

    """
    fname = 'ft2d'
    customlog.log_trace(f'{fname}: Computing 2-D FFT of array of shape {arr.shape}')
    _check_shapes_2d(fname, arr, x, y)

    arr_prepared = prepare_signal_for_ft2d(arr, window=window, winstyle=winstyle, remove_mean=remove_mean,
                                           nfft=nfft, zpf=zpf, shift=shift)

    customlog.log_subtrace(f'{fname}: Computing fft, norm={norm} | shape={arr_prepared.shape}')
    arr_hat = fft.fft2(arr_prepared, norm=norm, s=arr_prepared.shape)
    return fft.fftshift(arr_hat) * step(x) * step(y)


@alias_argument('zpf', 'zero_pad_factor')
def rft2d(arr: floatarray2D, x: Optional[floatarray1D] = None, y: Optional[floatarray1D] = None,
          window: str = default_window, winstyle: Optional[str] = None, remove_mean: bool = True,
          nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None,
          shift: dtype_shift_2d = None, norm: Optional[str] = 'backward') -> complexarray2D:
    """
    Returns the 2-D Fourier transform of a real input array using the given windowing.

    The spectrum is truncated along axis 1 (x), as with scipy.fft.rfft2.

    The unit is in amplitude/(inverse period), e.g. V(s, m) -> V/Hz/m$^{-1}$(Hz.m$^{-1}$)

    Parameters
    ----------
    arr
    x
        Coordinates along axis 1
    y
        Coordinates along axis 0
    window : str, optional
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    remove_mean : bool, optional
        Default is True
    nfft : int, 'auto' or (axis0, axis1), optional
    zpf : int or (int, int), optional
    shift : int or (int, int), optional
    norm : str, optional
        The scipy.fft normalization. Keep 'backward' (default) for the units to be right.

    Returns
    -------

    """
    fname = 'rft2d'
    customlog.log_trace(f'{fname}: Computing 2-D RFFT of array of shape {arr.shape}')
    _check_shapes_2d(fname, arr, x, y)

    arr_prepared = prepare_signal_for_ft2d(arr, window=window, winstyle=winstyle, remove_mean=remove_mean,
                                           nfft=nfft, zpf=zpf, shift=shift)

    customlog.log_subtrace(f'{fname}: Computing fft, norm={norm} | shape={arr_prepared.shape}')
    arr_hat = fft.rfft2(arr_prepared, norm=norm, s=arr_prepared.shape)
    return fft.fftshift(arr_hat, axes=0) * step(x) * step(y)


def ift2d(sig_hat: complexarray2D, xdual: Optional[np.ndarray] = None, ydual: Optional[np.ndarray] = None,
          norm: Optional[str] = 'backward'):
    return fft.ifft2(np.fft.ifftshift(sig_hat), norm=norm) * span(xdual) * span(ydual)


### Power:
from .FFT1D import window_factor1d


def window_factor2d(window: str, winstyle: Optional[str] = None, NxNy: Tuple[int, int] = (16384, 16384)):
    """
    Returns the factor by which the energy is multiplied when the signal is windowed.

    Parameters
    ----------
    window
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    NxNy : (int, int), optional
        The size over which to evaluate the window corrected factor.

    Returns
    -------

    """
    if winstyle is None:
        winstyle = 'outer'
    winstyle = str(winstyle)
    customlog.log_subtrace(f'window_factor2d: Using windowing | window={window} | style={winstyle}')
    Nx, Ny = NxNy
    if winstyle == 'outer':
        return window_factor1d(window, Nx) * window_factor1d(window, Ny)
    elif winstyle == 'circular':
        return 1 / (np.sum(filters.window(window, (Nx, Ny), warp_kwargs={'order': 3}) ** 2) / (Nx * Ny))
    else:
        customlog.log_warning(f'Unrecognized 2d-windowing style: {winstyle}.')
        return 1


@alias_argument('zpf', 'zero_pad_factor')
def psd2d(z: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
          window: str = default_window, winstyle: Optional[str] = None, remove_mean: bool = True,
          nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None) -> np.ndarray:
    """
    Returns the 2-D Fourier PSD of an input array using the given windowing.

    if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/(Hz.mm^{-1})(Hz, mm-1)]

    Parameters
    ----------
    z
    x
        Coordinates along axis 1
    y
        Coordinates along axis 0
    window : str, optional
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    remove_mean : bool, optional
        Default is True
    nfft : int, 'auto' or (axis0, axis1), optional
    zpf : int or (int, int), optional

    Returns
    -------

    """
    customlog.log_trace('psd2d: Computing a 2-D PSD')
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $\hat{z}$ is [V/(Hz.mm^{-1})(Hz, mm-1)]
    y_ft = ft2d(z, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                nfft=nfft, zpf=zpf)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the oly thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this 2-frequency
    # It is useful in itself for space-time-limited signals (wavelets)
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $ESD(z)$ is [V^2/Hz^2/mm^{-2}(Hz, mm-1)]
    esd = np.abs(y_ft) ** 2 * window_factor2d(window, winstyle=winstyle, NxNy=z.shape)
    ### Step 3 : compute the PSD (Power Spectral Density)
    # Assuming that the signal is 2-D periodic, then PSD = ESD / (duration_1.duration_2)
    # Thus if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/Hz/mm^{-1}(Hz, mm-1)]
    return esd / span(x) / span(y)


@alias_argument('zpf', 'zero_pad_factor')
def rpsd2d(z: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
           window: str = default_window, winstyle: Optional[str] = None, remove_mean: bool = True,
           nfft: dtype_nfft_2d = None, zpf: dtype_zpf_2d = None) -> np.ndarray:
    """
    Returns the 2-D Fourier PSD of a real input array using the given windowing.

    The spectrum is truncated along axis 1 (x), as with scipy.fft.rfft2.

    if the unit of z(t, x) is [V(s, mm)], then the unit of PSD(z)$ is [V^2/(Hz.mm^{-1})(Hz, mm-1)]

    Parameters
    ----------
    z
    x
        Coordinates along axis 1
    y
        Coordinates along axis 0
    window : str, optional
    winstyle : str, optional
        Can be 'outer' (default) or 'circular'
    remove_mean : bool, optional
        Default is True
    nfft : int, 'auto' or (axis0, axis1), optional
    zpf : int or (int, int), optional

    Returns
    -------

    """
    customlog.log_trace('rpsd2d: Computing a 2-D PSD')
    ### Step 1 : do the dimensional Fourier transform
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $\hat{z}$ is [V/(Hz.mm^{-1})(Hz, mm-1)]
    y_ft = rft2d(z, x=x, y=y, window=window, winstyle=winstyle, remove_mean=remove_mean,
                 nfft=nfft, zpf=zpf)
    ### Step 2 : compute the ESD (Energy Spectral Density)
    # Rigorously, this is the oly thing we can really measure with discretized inputs and FFT
    # It is the total energy (i.e., during all the sampling time) of the signal at this 2-frequency
    # It is useful in itself for space-time-limited signals (wavelets)
    # if the unit of z(t, x) is [V(s, mm)], then the unit of $ESD(z)$ is [V^2/Hz^2/mm^{-2}(Hz, mm-1)]
    esd = np.abs(y_ft) ** 2 * window_factor2d(window, winstyle=winstyle, NxNy=z.shape)
    n1 = sanitize_nfft_2d(nfft, z.shape)[1] * sanitize_zpf_2d(zpf)[1]
    esd[:, 1:esd.shape[1] - (1 - n1 % 2)] *= 2  # x 2 because of rfft which truncates the spectrum (except the 0 harmonic, and last depending on the parity)
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

    customlog.log_trace(f'Found {len(polygons)} contours')

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
    customlog.log_trace(f'Perimeters of {len(contours)} contours')
    if condition is None:
        condition = np.ones(len(contours), dtype=bool)
    perimeters = np.ones(len(contours), dtype=float)
    for i, contour in enumerate(contours):
        if condition[i]:
            customlog.log_subtrace(f'contour {i}: type: {contour.geom_type}')
            customlog.log_subtrace(f'contour {i}: contained geometries: {len(contour.geoms)}')
            hull = contour.geoms[0]
            customlog.log_subtrace(f'contour {i}: hull is : {hull.geom_type}')
            customlog.log_subtrace(f'contour {i}: hull boundary is : {hull.boundary.geom_type}')
            customlog.log_subtrace(f'contour {i}: hull exterior is : {hull.exterior.geom_type}')
            customlog.log_subtrace(f'contour {i}: hull interior contains {len(hull.interiors)} rings')
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
        customlog.log_subtrace(f'geom is : {geom.geom_type}')
        customlog.log_subtrace(f'geom boundary is : {geom.boundary.geom_type}')
        customlog.log_subtrace(f'geom exterior is : {geom.exterior.geom_type}')
        customlog.log_subtrace(f'geom interior contains {len(geom.interiors)} rings')
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
        customlog.log_warning(f'peak_contour2d: z should be a 2D array. Its shape is {z.shape}')
    if x is None:
        x = np.arange(z.shape[1])
    else:
        if len(x) != z.shape[1]:
            customlog.log_warning(f'peak_contour2d: Shapes do not match. {z.shape} != {y.shape}x{x.shape}')
    if y is None:
        y = np.arange(z.shape[0])
        if len(y) != z.shape[0]:
            customlog.log_warning(f'peak_contour2d: Shapes do not match. {z.shape} != {y.shape}x{x.shape}')
    if peak_max_area is None:
        global peak_max_area_default
        peak_max_area = peak_max_area_default
    if peak_min_circularity is None:
        global peak_min_circularity_default
        peak_min_circularity = peak_min_circularity_default

    customlog.log_debug(
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
                customlog.log_subtrace(f'multipoligon {i}: centroid: {multipolygons[i].centroid}')
                customlog.log_subtrace(f'multipoligon {i}: area: {round(areas[i], 1)} px^2, limit is {peak_max_area}')
                customlog.log_subtrace(f'multipoligon {i}: perimeter: {round(perimeters[i], 1)} px')
                customlog.log_subtrace(
                    f'multipoligon {i}: circularity: {round(circularities[i], 3)} [0-1], limit is {peak_min_circularity}')

        isnottoobig = areas < peak_max_area
        iscircularenough = circularities > peak_min_circularity

        contour_is_valid = containspeak * isnottoobig * iscircularenough

        customlog.log_trace(f'Contains peak: {containspeak if not fastmode else containspeak[containspeak]}')
        customlog.log_trace(f'Not too big:   {isnottoobig if not fastmode else isnottoobig[containspeak]}')
        customlog.log_trace(f'Circular:      {iscircularenough if not fastmode else iscircularenough[containspeak]}')
        customlog.log_trace(f'Valid contour: {contour_is_valid if not fastmode else contour_is_valid[containspeak]}')

        # find the contour that contains the point
        if contour_is_valid.any() or peak_depth_dB == min_peak_depth_dB:
            maincontours = [multipolygons[index] for index in np.where(contour_is_valid == True)[0]]
            customlog.log_debug(f"Found {len(maincontours)} valid contours.")
            return maincontours
        else:
            peak_depth_dB -= 5
            if peak_depth_dB < min_peak_depth_dB: peak_depth_dB = min_peak_depth_dB
            customlog.log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")
    if peak_depth_dB != min_peak_depth_dB:
        peak_depth_dB = min_peak_depth_dB
        customlog.log_debug(f"Couldn't find any valid contour: Trying peak_depth={peak_depth_dB} dB")


def grid_points_in_contour(contour, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray:
    incontour = np.zeros((len(y), len(x)), dtype=bool)
    xmin, ymin, xmax, ymax = contour.bounds
    for i_x in np.where((x >= xmin) * (x <= xmax))[0]:
        for i_y in np.where((y >= ymin) * (y <= ymax))[0]:
            incontour[i_y, i_x] = contour_containspoint(contour, (x[i_x], y[i_y]))
    return incontour


def peak_vicinity2d(peak_x, peak_y, z: np.ndarray[np.ndarray[float]], peak_depth_dB,
                    x: Optional[np.ndarray[float]] = None, y: Optional[np.ndarray[float]] = None,
                    peak_contours: Optional[List] = None,
                    peak_max_area: Optional[float] = None, peak_min_circularity: Optional[float] = None):
    if x is None:
        x = np.arange(z.shape[1])
    if y is None:
        y = np.arange(z.shape[0])
    customlog.log_debug(f'Searching for the vicinity of  ({round(peak_x, 3)}, {round(peak_y, 3)})')

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

    customlog.log_debug(f'Found a vicinity of area {mask.sum()} px²')

    return mask


def power_near_peak2d(peak_x, peak_y, z, peak_depth_dB, x=None, y=None,
                      peak_contours: Optional[List] = None, peak_vicinity: Optional[np.ndarray] = None,
                      peak_max_area: Optional[float] = None, peak_min_circularity: Optional[float] = None):
    customlog.log_debug(f'Measuring the power around     ({round(peak_x, 3)}, {round(peak_y, 3)})')
    if peak_vicinity is None:
        peak_vicinity = peak_vicinity2d(peak_x=peak_x, peak_y=peak_y, z=z, peak_depth_dB=peak_depth_dB, x=x, y=y,
                                        peak_contours=peak_contours,
                                        peak_max_area=peak_max_area, peak_min_circularity=peak_min_circularity)
    pw = np.sum(z[peak_vicinity]) * step(x) * step(y)
    customlog.log_debug(f'Power: {pw} (amplitude: {np.sqrt(2*pw)})')
    return pw

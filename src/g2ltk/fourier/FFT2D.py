from typing import Optional, Any, Tuple, Dict, List, Union, Literal
import numpy as np
from scipy.signal.windows import get_window  # FFT windowing
from skimage import filters  # filters.window for 2D FFT windowing
from scipy import fft

from g2ltk import customlog
from g2ltk.arrtools import step, span
from . import floatarray1D, floatarray2D, complexarray2D, attenuate_power, alias_argument, default_window
from .FFTHelp import sanitize_nfft_2d, sanitize_zpf_2d, dtype_nfft_2d, dtype_zpf_2d, dtype_shift_2d, split_2d_argument
from .FFT1D import dual1d, rdual1d

# The names g2ltk.fourier gets from `from .FFT2D import *`: add new public functions here
__all__ = ['dual2d', 'rdual2d', 'prepare_signal_for_ft2d', 'ft2d', 'rft2d', 'ift2d', 'riesz2d',
           'monogenic2d', 'local_frequency2d', 'window_factor2d', 'psd2d', 'rpsd2d',
           'find_shapely_contours', 'contour_containspoint', 'contour_containspoints',
           'contours_containsanyofthepoints', 'contours_areas', 'contours_perimeters',
           'draw_multipolygon_edge', 'peak_max_area_default', 'peak_min_circularity_default',
           'peak_contour2d', 'grid_points_in_contour', 'peak_vicinity2d', 'power_near_peak2d']

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
        The zero padding factor. An alias is ``zero_pad_factor``

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
        The zero padding factor. An alias is ``zero_pad_factor``

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


### Riesz transform
def riesz2d(arr: floatarray2D, x: Optional[floatarray1D] = None, y: Optional[floatarray1D] = None,
            symmetrize: bool = False, remove_mean: bool = True) -> Tuple[floatarray2D, floatarray2D]:
    """
    Returns the two components (R_x, R_y) of the Riesz transform of a real 2-D array.

    The Riesz transform is the 2-D generalization of the Hilbert transform. Its components are defined
    in Fourier space by::

        FT[R_x f](k) = -i k_x / |k| FT[f](k)    and    FT[R_y f](k) = -i k_y / |k| FT[f](k).

    For an array that only varies along x, R_x is the Hilbert transform of each row
    (the imaginary part of the analytic signal, see FFT1D.hilbert) and R_y is 0.

    For a plane wave A cos(k.r + phi), ``R_x = A sin(k.r + phi) k_x/|k|`` and ``R_y = A sin(k.r + phi) k_y/|k|``.
    (f, R_x f, R_y f) is the monogenic signal: f = A cos(phi), R_x f = A sin(phi) cos(theta),
    R_y f = A sin(phi) sin(theta), with A the local amplitude, phi the local phase and theta the local orientation.

    The ratios ``k_x/|k|``, ``k_y/|k|`` are computed with the physical frequencies, so the result depends on the steps
    of x and y when they differ: give them if the grid is not square.

    The components of the Nyquist frequency (along x for R_x, along y for R_y), for which the transform
    of a real signal cannot be real, are set to 0.

    Parameters
    ----------
    arr
        Real array, shaped (len(y), len(x))
    x
        Coordinates along axis 1. Only its step is used (1 if not given).
    y
        Coordinates along axis 0. Only its step is used (1 if not given).
    symmetrize : bool, optional
        If True, the array is mirrored along both axes (giving a 4 times bigger array) before the transform,
        to remove the discontinuities at the boundaries. This forces the phase to 0 or pi near the edges.
        Default is False.
    remove_mean : bool, optional
        Default is True

    Returns
    -------
    (R_x, R_y)
        Two real arrays shaped like arr
    """
    fname = 'riesz2d'
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f'{fname}: expected a 2-D array, got an array of shape {arr.shape}')
    if np.iscomplexobj(arr):
        raise ValueError(f'{fname}: the Riesz transform is only implemented for real arrays')
    customlog.log_trace(f'{fname}: Computing the Riesz transform of array of shape {arr.shape}')
    _check_shapes_2d(fname, arr, x, y)

    N0, N1 = arr.shape
    arr_prepared = arr.astype(float)
    if symmetrize:
        arr_prepared = np.block([[arr_prepared, arr_prepared[:, ::-1]],
                                 [arr_prepared[::-1, :], arr_prepared[::-1, ::-1]]])
    if remove_mean:
        arr_prepared -= arr_prepared.mean()
    M0, M1 = arr_prepared.shape

    # physical frequencies, in the layout of rfft2 (unshifted, half spectrum along axis 1 <-> x)
    ky = np.expand_dims(fft.fftfreq(M0, d=step(y)), axis=1)
    kx = np.expand_dims(fft.rfftfreq(M1, d=step(x)), axis=0)
    knorm = np.hypot(kx, ky)
    knorm[0, 0] = 1.  # the k = 0 component is set to 0 below
    mult_x = -1j * kx / knorm
    mult_y = -1j * ky / knorm
    mult_x[0, 0] = mult_y[0, 0] = 0.
    if M1 % 2 == 0:
        mult_x[:, -1] = 0.
    if M0 % 2 == 0:
        mult_y[M0 // 2, :] = 0.

    arr_hat = fft.rfft2(arr_prepared)
    riesz_x = fft.irfft2(arr_hat * mult_x, s=(M0, M1))[:N0, :N1]
    riesz_y = fft.irfft2(arr_hat * mult_y, s=(M0, M1))[:N0, :N1]
    return riesz_x, riesz_y


def _fold_orientation(orientation: np.ndarray, orientation_reference: float) -> np.ndarray:
    # brings the orientation (defined modulo pi) into (orientation_reference - pi/2, orientation_reference + pi/2]
    return orientation_reference + np.pi / 2 - np.mod(np.pi / 2 - (orientation - orientation_reference), np.pi)


def _monogenic2d_components(arr: floatarray2D, x: Optional[floatarray1D], y: Optional[floatarray1D],
                            symmetrize: bool, remove_mean: bool,
                            orientation_reference: float) -> Tuple[floatarray2D, floatarray2D, floatarray2D, floatarray2D]:
    # returns f (after mean removal), R_x f, R_y f and the folded orientation
    riesz_x, riesz_y = riesz2d(arr, x=x, y=y, symmetrize=symmetrize, remove_mean=remove_mean)
    # the mean of the symmetrized array is the mean of the array
    f = np.asarray(arr, dtype=float)
    if remove_mean:
        f = f - f.mean()
    orientation = _fold_orientation(np.arctan2(riesz_y, riesz_x), orientation_reference)
    return f, riesz_x, riesz_y, orientation


def monogenic2d(arr: floatarray2D, x: Optional[floatarray1D] = None, y: Optional[floatarray1D] = None,
                symmetrize: bool = False, remove_mean: bool = True,
                orientation_reference: float = 0.) -> Tuple[floatarray2D, floatarray2D, floatarray2D]:
    """
    Returns the local amplitude, local phase and local orientation of a real 2-D array, from its monogenic signal.

    The monogenic signal is (f, R_x f, R_y f), with (R_x, R_y) the Riesz transform (see riesz2d).
    It is the 2-D generalization of the analytic signal, and is written::

        f = A cos(phi),    R_x f = A sin(phi) cos(theta),    R_y f = A sin(phi) sin(theta).

    For a plane wave A cos(k.r + phi0), A is the amplitude, theta the direction of k and phi = k.r + phi0.

    Conventions
    -----------
    Let theta0 be orientation_reference (0 by default).

    - amplitude: A = sqrt(f^2 + (R_x f)^2 + (R_y f)^2) >= 0, with f the array after mean removal.
    - orientation: theta in (theta0 - pi/2, theta0 + pi/2], measured from the x axis (axis 1) towards the y axis
      (axis 0), in physical coordinates (so it depends on the steps of x and y if they differ).
      With the default theta0 = 0, theta is in (-pi/2, pi/2].
    - phase: phi = atan2(R_x f cos(theta) + R_y f sin(theta), f), in (-pi, pi].

    (A, phi, theta) and (A, -phi, theta + pi) describe the same monogenic signal: the orientation is only defined
    modulo pi, and the sign of the phase depends on it. The convention above picks the direction that is within
    pi/2 of theta0. In consequence:

    - for a plane wave whose wave vector k is more than pi/2 away from theta0, phi = -(k.r + phi0)
      (the wave is written with -k instead);
    - phi changes sign where the local orientation crosses theta0 +- pi/2. For a wave vector exactly in one of
      these directions, rounding errors decide between the two, so the sign of phi can change from point to point.
      Choose theta0 along the (rough) direction of the wave vector to avoid this: for fringes that are roughly
      perpendicular to y for example, use theta0 = pi/2.
    - the phase vector phi * (cos(theta), sin(theta)) does not depend on this choice.
    - for an array that only varies along x, with theta0 = 0, theta = 0 and (A, phi) are those given by
      FFT1D.monogenic1d.

    Where A sin(phi) = 0 the orientation is not defined (it is theta0 if R_x f = R_y f = 0 exactly,
    and rounding noise if they are only close to 0).

    Parameters
    ----------
    arr
        Real array, shaped (len(y), len(x))
    x
        Coordinates along axis 1. Only its step is used (1 if not given).
    y
        Coordinates along axis 0. Only its step is used (1 if not given).
    symmetrize : bool, optional
        See riesz2d. Default is False.
    remove_mean : bool, optional
        Default is True
    orientation_reference : float, optional
        The orientation theta0 (in radians) around which the orientation is folded. Default is 0.

    Returns
    -------
    (amplitude, phase, orientation)
        Three real arrays shaped like arr
    """
    f, riesz_x, riesz_y, orientation = _monogenic2d_components(arr, x, y, symmetrize, remove_mean,
                                                               orientation_reference)
    riesz_along_orientation = riesz_x * np.cos(orientation) + riesz_y * np.sin(orientation)

    amplitude = np.sqrt(f ** 2 + riesz_x ** 2 + riesz_y ** 2)
    phase = np.arctan2(riesz_along_orientation, f)
    return amplitude, phase, orientation


def _phase_vector_derivative(f: floatarray2D, riesz_x: floatarray2D, riesz_y: floatarray2D,
                             axis: int) -> Tuple[floatarray2D, floatarray2D]:
    # derivative of the phase vector phi * n along axis, per sample, returned as its (x, y) components.
    # Between neighbours m and m+1, the unit monogenic vectors (f, R f) / A are rotated by the phase difference
    # dphi, around a direction given by n:
    #     w = f_m (R f)_{m+1} - (R f)_m f_{m+1} = A_m A_{m+1} sin(dphi) n
    #     c = f_m f_{m+1} + (R f)_m . (R f)_{m+1} = A_m A_{m+1} cos(dphi)
    # so that dphi * n = atan2(|w|, c) w / |w|, which does not depend on the sign convention of (phi, n)
    # and does not use the orientation at each point (which is noise where sin(phi) ~ 0).
    # Central difference (mean of the forward and backward ones) inside, one-sided ones at the edges.
    f, riesz_x, riesz_y = (np.moveaxis(a, axis, 0) for a in (f, riesz_x, riesz_y))
    wx = f[:-1] * riesz_x[1:] - riesz_x[:-1] * f[1:]
    wy = f[:-1] * riesz_y[1:] - riesz_y[:-1] * f[1:]
    c = f[:-1] * f[1:] + riesz_x[:-1] * riesz_x[1:] + riesz_y[:-1] * riesz_y[1:]
    wnorm = np.hypot(wx, wy)
    scale = np.arctan2(wnorm, c) / np.where(wnorm > 0, wnorm, 1.) * (wnorm > 0)

    derivatives = []
    for difference in (wx * scale, wy * scale):  # differences between m and m+1
        derivative = np.empty_like(f)
        derivative[0] = difference[0]
        derivative[-1] = difference[-1]
        derivative[1:-1] = (difference[1:] + difference[:-1]) / 2
        derivatives.append(np.moveaxis(derivative, 0, axis))
    return derivatives[0], derivatives[1]


def local_frequency2d(arr: floatarray2D, x: Optional[floatarray1D] = None, y: Optional[floatarray1D] = None,
                      symmetrize: bool = False, remove_mean: bool = True,
                      orientation_reference: float = 0.) -> Tuple[floatarray2D, floatarray2D]:
    """
    Returns the local wave vector (f_x, f_y) of a real 2-D array, i.e. grad(phi) / (2 pi) with phi the local phase
    of the monogenic signal (see monogenic2d).

    The unit is the inverse of the unit of x and y (cycles per unit length, as dual2d):
    for a plane wave cos(2 pi (fx0 x + fy0 y) + phi0), (f_x, f_y) = (fx0, fy0).

    The wave vector of a real wave is only defined up to its sign: the one returned has an orientation within pi/2
    of orientation_reference (see monogenic2d), i.e. it is (fx0, fy0) or (-fx0, -fy0) for a plane wave.

    Method
    ------
    The phase is not unwrapped, and the orientation given by monogenic2d is not used (it is noise where
    sin(phi) ~ 0). Instead, the derivatives of the phase vector phi * n (n = (cos(theta), sin(theta))) along x and y
    are computed from the rotation of the unit monogenic vector (f, R_x f, R_y f) / A between neighbours, which
    does not depend on the sign convention. Central differences are used inside, one-sided ones at the edges.
    This gives the tensor J_ej = d(phi)/de n_j, which for a locally 1-D signal is ``|k| n_e n_j``:
    the wave vector is the eigenvector of (the symmetric part of) J with the largest eigenvalue (in absolute value),
    scaled by this eigenvalue. Its norm is the local frequency, and is negative where the phase decreases along n
    (which happens where the amplitude is small).

    The result is exact for a plane wave as long as the phase changes by less than pi between neighbours
    (i.e. below the Nyquist frequency along each axis). For an array that only varies along x,
    f_x is the local frequency given by FFT1D.local_frequency1d and f_y = 0.
    Where the amplitude is small (or the signal is not locally 1-D), the local wave vector is not meaningful.

    Parameters
    ----------
    arr
        Real array, shaped (len(y), len(x))
    x
        Coordinates along axis 1. Only its step is used (1 if not given).
    y
        Coordinates along axis 0. Only its step is used (1 if not given).
    symmetrize : bool, optional
        See riesz2d. Default is False.
    remove_mean : bool, optional
        Default is True
    orientation_reference : float, optional
        See monogenic2d. Default is 0.

    Returns
    -------
    (f_x, f_y)
        Two real arrays shaped like arr
    """
    f, riesz_x, riesz_y, _ = _monogenic2d_components(arr, x, y, symmetrize, remove_mean, orientation_reference)
    jxx, jxy = _phase_vector_derivative(f, riesz_x, riesz_y, axis=1)
    jyx, jyy = _phase_vector_derivative(f, riesz_x, riesz_y, axis=0)
    jxx, jxy = jxx / step(x), jxy / step(x)
    jyx, jyy = jyx / step(y), jyy / step(y)

    # eigen-decomposition of the symmetric part [[a, b], [b, c]] of J
    a, b, c = jxx, (jxy + jyx) / 2, jyy
    mean, radius = (a + c) / 2, np.hypot((a - c) / 2, b)
    angle = np.arctan2(2 * b, a - c) / 2  # orientation of the eigenvector of the eigenvalue mean + radius
    largest_is_positive = mean >= 0
    eigenvalue = np.where(largest_is_positive, mean + radius, mean - radius)
    orientation = _fold_orientation(np.where(largest_is_positive, angle, angle + np.pi / 2), orientation_reference)

    wavevector_x = eigenvalue * np.cos(orientation)
    wavevector_y = eigenvalue * np.sin(orientation)
    return wavevector_x / (2 * np.pi), wavevector_y / (2 * np.pi)


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

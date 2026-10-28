from typing import Optional, Literal, Union, Any, Tuple, TypeVar, Union
import math
import numbers

from g2ltk import customlog

## Multidim stuff

def split_2d_argument(value: Any, fname: str = 'split_2d_argument') -> Tuple[Any, Any]:
    """
    Splits a 2D argument into its (axis0, axis1) components.

    A single value (None, int, 'auto') is used for both axes, a pair is unpacked.

    Parameters
    ----------
    value
    fname : str, optional
        The name of the calling function, for logging.

    Returns
    -------
    (value_axis0, value_axis1)
    """
    if value is None or isinstance(value, str):
        return value, value
    if isinstance(value, numbers.Integral):
        return int(value), int(value)
    try:
        v0, v1 = value
    except (TypeError, ValueError):
        customlog.log_warning(f'{fname}: What is this "{value}"? Expected a single value or a pair. I made it None')
        return None, None
    v0 = int(v0) if isinstance(v0, numbers.Integral) else v0
    v1 = int(v1) if isinstance(v1, numbers.Integral) else v1
    return v0, v1


### NFFT
from scipy.fft import next_fast_len

def is_fast_len(n: int):
    return n == next_fast_len(n)

def next_fast_aligned_len(n: int, align: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    if align < 1:
        raise ValueError("align must be >= 1")
    if not is_fast_len(align):
        raise ValueError(
            f"align={align} has high prime factors"
        )
    start = math.ceil(n / align) * align
    candidate = start
    while True:
        if is_fast_len(candidate):
            return candidate
        candidate += align   # stay on multiples — skip non-multiples entirely

dtype_nfft = Optional[Union[int, Literal['auto']]]
dtype_nfft_2d = Union[dtype_nfft, Tuple[dtype_nfft, dtype_nfft]]

def sanitize_nfft_1d(nfft: dtype_nfft, N: int, aligned: Optional[int]=None):
    """
    Checks that nfft is suitable (an integer, or 'auto') or resets to default (array length N).

    Parameters
    ----------
    nfft
    N : int
        The array length
    aligned : int, optional
        The minimum factor of nfft (useful for welch averaging).

    Returns
    -------

    """
    if nfft is None:
        return N
    if isinstance(nfft, int):
        return int(nfft)
    if nfft == 'auto':
        if aligned is not None:
            if isinstance(aligned, int):
                if aligned > 1:
                    return next_fast_aligned_len(N, aligned)
            else:
                customlog.log_warning(f'{"sanitize_nfft"}: Why is aligned ({aligned}) not an int?')
        return next_fast_len(N)
    customlog.log_warning(f'{"sanitize_nfft"}: What is this nfft "{nfft}"? I made it None')
    return N

def sanitize_nfft_2d(nfft: dtype_nfft_2d, shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Checks that nfft is suitable (an integer, 'auto', or a pair of those) or resets to default (array shape).

    Parameters
    ----------
    nfft
    shape : (int, int)
        The array shape

    Returns
    -------
    (nfft_axis0, nfft_axis1)
    """
    nfft0, nfft1 = split_2d_argument(nfft, 'sanitize_nfft_2d')
    return sanitize_nfft_1d(nfft0, shape[0]), sanitize_nfft_1d(nfft1, shape[1])

### ZPF

dtype_zpf = Optional[Union[int]]
dtype_zpf_2d = Union[dtype_zpf, Tuple[dtype_zpf, dtype_zpf]]

def sanitize_zpf_1d(zpf):
    """
    Checks that zpf is suitable (an integer) or resets to default (1).

    Parameters
    ----------
    zpf

    Returns
    -------

    """
    default_zpf = 1
    if zpf is None:
        return default_zpf
    if isinstance(zpf, int):
        return zpf
    customlog.log_warning(f'{"sanitize_zpf_1d"}: What is this zpf "{zpf}" ? I made it None')
    return default_zpf

def sanitize_zpf_2d(zpf: dtype_zpf_2d) -> Tuple[int, int]:
    """
    Checks that zpf is suitable (an integer, or a pair of integers) or resets to default (1, 1).

    Parameters
    ----------
    zpf

    Returns
    -------
    (zpf_axis0, zpf_axis1)
    """
    zpf0, zpf1 = split_2d_argument(zpf, 'sanitize_zpf_2d')
    return sanitize_zpf_1d(zpf0), sanitize_zpf_1d(zpf1)

### SHIFT

dtype_shift = Optional[Union[int]]
dtype_shift_2d = Union[dtype_shift, Tuple[dtype_shift, dtype_shift]]

### WELCH

def recommanded_welch_overlap(n: int, window: str):
    if window == 'boxcar':
        return 0
    elif window in ['hann', 'hamming', 'bartlett']:
        return n//2
    customlog.log_warning(f'{"recommanded_overlap"}: What is this window "{window}"? I guessed 50%')
    return n//2

from typing import Optional, Literal, Union
import math

from g2ltk import customlog

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

### ZPF

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

### WELCH

def recommanded_welch_overlap(n: int, window: str):
    if window == 'boxcar':
        return 0
    elif window in ['hann', 'hamming', 'bartlett']:
        return n//2
    customlog.log_warning(f'{"recommanded_overlap"}: What is this window "{window}"? I guessed 50%')
    return n//2

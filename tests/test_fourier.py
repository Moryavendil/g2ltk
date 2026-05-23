import g2ltk.fourier as ft
import numpy as np
import pytest
from scipy import fft
from scipy import signal

np.random.seed(19970510)

def generate_sig1D(npts):
    t = np.linspace(0., 15.75234, npts, endpoint=False)
    freq1 = 1.
    phi1 = 0.4
    freq2 = 2.5
    phi2 = 0.7

    a1 = 1.
    a2 = 0.6

    anoise = 0.1

    sig = a1*np.cos(2*np.pi*freq1*t+phi1) + a2*np.cos(2*np.pi*freq2*t+phi2) + anoise*np.random.rand(len(t))

    return t, sig

N_to_test = [20, 63, 1024, 16384] # small to big, even and odd
window_to_test = ['boxcar', 'hann', 'hamming', 'tukey', 'blackman', 'flattop']
zpf_to_test = [1, 2, 3, 4, 16] # small to big, even and odd
@pytest.mark.parametrize("N", N_to_test)
def test_ft_axis0(N):
    t, sig = generate_sig1D(N)

    shapes = [(N, 1), (N, 1, 1), (N, 1, 1, 1)]
    for axes in [0, (0,)]:
        for shape in shapes:
            sig = sig.reshape(shape)
            sig_ft_scipy = fft.fft(sig, axis=0)
            sig_ft_g2l = ft.ft(sig, x=t, remove_mean=False, axes=axes)

            assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(t), sig_ft_g2l).all()

@pytest.mark.parametrize("N", N_to_test)
def test_ft_axis1(N):
    t, sig = generate_sig1D(N)

    shapes = [(1, N), (1, N, 1), (1, N, 1, 1)]
    for axes in [1, (1,)]:
        for shape in shapes:
            sig = sig.reshape(shape)
            sig_ft_scipy = fft.fft(sig, axis=1)
            sig_ft_g2l = ft.ft(sig, x=t, remove_mean=False, axes=axes)

            assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(t), sig_ft_g2l).all()

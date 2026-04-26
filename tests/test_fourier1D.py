import g2ltk.fourier as ft
import numpy as np
import pytest
from scipy import fft
from scipy import signal

np.random.seed(19970510)

def generate_sig1D(N):
    t = np.linspace(0, 15.75, 1024, endpoint=False)
    freq1 = 1.
    phi1 = 0.4
    freq2 = 2.5
    phi2 = 0.7

    a1 = 1.
    a2 = 0.6

    anoise = 0.1

    sig = a1*np.cos(2*np.pi*freq1*t+phi1) + a2*np.cos(2*np.pi*freq1*t+phi1) + anoise*np.random.rand(len(t))

    return t, sig

N_to_test = [20, 63, 1024, 6561, 16384] # small to big, even and odd
window_to_test = ['boxcar', 'hann', 'hamming', 'tukey', 'blackman', 'flattop']

@pytest.mark.parametrize("N", N_to_test)
def test_dual1d(N):
    t, sig = generate_sig1D(N)

    assert np.isclose(ft.dual1d(t), fft.fftshift(fft.fftfreq(len(t), t[1]-t[0]))).all()


@pytest.mark.parametrize("N", N_to_test)
def test_rdual1d(N):
    t, sig = generate_sig1D(N)

    assert np.isclose(ft.rdual1d(t), fft.rfftfreq(len(t), t[1]-t[0])).all()

@pytest.mark.parametrize("N", N_to_test)
def test_ft1d(N):
    t, sig = generate_sig1D(N)
    sig_ft_scipy = fft.fft(sig)
    sig_ft_g2l = ft.ft1d(sig, x=t, remove_mean=False)

    assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(t), sig_ft_g2l).all()

@pytest.mark.parametrize("N", N_to_test)
def test_ift1d(N):
    t, sig = generate_sig1D(N)
    f = ft.dual1d(t)
    sig_ft_g2l = ft.ft1d(sig, x=t, remove_mean=False)

    assert np.isclose(sig, ft.ift1d(sig_ft_g2l, f)).all()

@pytest.mark.parametrize("N", N_to_test)
def test_rft1d(N):
    t, sig = generate_sig1D(N)

    ft_scipy = np.fft.rfft(sig)
    ft_g2l = ft.rft1d(sig, x=t, remove_mean=False)

    assert np.isclose(ft_g2l, ft_scipy*ft.step(t)).all()

@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
def test_psd1d_definition(N, window):
    t, sig = generate_sig1D(N)

    ft_g2l = ft.ft1d(sig, x=t, remove_mean=False, window=window)
    psd_g2l_alter = np.abs(ft_g2l)**2 * ft.window_factor1d(window) / (len(t)*ft.step(t))

    psd_g2l = ft.psd1d(sig, x=t, remove_mean=False, window=window)

    assert np.isclose(psd_g2l_alter, psd_g2l).all()

@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
def test_psd1d_periodogram(N, window):
    t, sig = generate_sig1D(N)
    psd_g2l = ft.psd1d(sig, x=t, remove_mean=True, window=window)

    fs = 1/(t[1]-t[0])
    f_scipy, psd_scipy =  signal.periodogram(sig, fs=fs, window=window,
                                             return_onesided=False, scaling='density', detrend='constant')
    f_scipy, psd_scipy = fft.fftshift(f_scipy), fft.fftshift(psd_scipy)

    assert np.isclose(psd_scipy, psd_g2l).all()
def test_estimatesignalfrequency():
    # we test if we are able to find the frequency ona sinusoid drowned in a gaussian uncorrelated noise
    # with signal / noise ratio snr
    snr = 0.5
    x = np.linspace(0, 10, 1000)
    f_true = 0.812
    z = np.sin(2*np.pi*f_true*x) + np.random.randn(len(x)) / snr
    f_est = ft.estimatesignalfrequency(z, x=x,
    window='boxcar', zero_pad_factor=4, bounds=None)

    max_acceptable_error = 1/ft.span(x)
    assert np.abs(f_true - f_est) < max_acceptable_error

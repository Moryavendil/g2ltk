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
def test_ft1d_axis0(N):
    t, sig = generate_sig1D(N)

    axis = 0
    shapes = [(N, 1), (N, 1, 1), (N, 1, 1, 1)]
    for shape in shapes:
        sig = sig.reshape(shape)
        sig_ft_scipy = fft.fft(sig, axis=axis)
        sig_ft_g2l = ft.ft1d(sig, x=t, remove_mean=False, axis=axis)

        assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(t), sig_ft_g2l).all()

@pytest.mark.parametrize("N", N_to_test)
def test_ft1d_axis1(N):
    t, sig = generate_sig1D(N)

    axis = 1
    shapes = [(1, N), (1, N, 1), (1, N, 1, 1)]
    for shape in shapes:
        sig = sig.reshape(shape)
        sig_ft_scipy = fft.fft(sig, axis=axis)
        sig_ft_g2l = ft.ft1d(sig, x=t, remove_mean=False, axis=axis)

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
    psd_g2l_alter = np.abs(ft_g2l)**2 * ft.window_factor1d(window, N) / (len(t)*ft.step(t))

    psd_g2l = ft.psd1d(sig, x=t, remove_mean=False, window=window)

    assert np.isclose(psd_g2l_alter, psd_g2l).all()



@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zero_pad_factor", zpf_to_test)
def test_psd1d_periodogram(N, window, zero_pad_factor):
    t, sig = generate_sig1D(N)
    psd_g2l = ft.psd1d(sig, x=t, remove_mean=True, window=window, zero_pad_factor=zero_pad_factor)

    fs = 1/(t[1]-t[0])
    f_scipy, psd_scipy =  signal.periodogram(sig, fs=fs, window=window,
                                             nfft=len(sig)*zero_pad_factor,
                                             return_onesided=False, scaling='density', detrend='constant')
    f_scipy, psd_scipy = fft.fftshift(f_scipy), fft.fftshift(psd_scipy)

    assert np.isclose(psd_scipy, psd_g2l).all()


@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zero_pad_factor", zpf_to_test)
def test_rpsd1d_periodogram(N, window, zero_pad_factor):
    t, sig = generate_sig1D(N)
    rpsd_g2l = ft.rpsd1d(sig, x=t, remove_mean=True, window=window, zero_pad_factor=zero_pad_factor)

    fs = 1/(t[1]-t[0])
    f_scipy, rpsd_scipy =  signal.periodogram(sig, fs=fs, window=window,
                                             nfft=len(sig)*zero_pad_factor,
                                             return_onesided=True, scaling='density', detrend='constant')

    assert len(rpsd_scipy) == len(rpsd_g2l)
    assert np.isclose(f_scipy, ft.rdual1d(t, zero_pad_factor=zero_pad_factor)).all()
    assert np.isclose(rpsd_scipy, rpsd_g2l).all()


@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zpf", zpf_to_test)
def test_welch1d_compat(N, window, zpf):
    t, sig = generate_sig1D(N)
    psd_g2l = ft.psd1d(sig, x=t, remove_mean=True, window=window, zpf=zpf)
    welch_g2l = ft.welch1d(sig, x=t, remove_mean=True, window=window, zpf=zpf, welch_factor=1)
    assert np.allclose(psd_g2l, welch_g2l)


@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zpf", zpf_to_test)
def test_rwelch1d_compat(N, window, zpf):
    t, sig = generate_sig1D(N)
    rpsd_g2l = ft.rpsd1d(sig, x=t, remove_mean=True, window=window, zpf=zpf)
    welch_g2l = ft.rwelch1d(sig, x=t, remove_mean=True, window=window, zpf=zpf, welch_factor=1)
    assert np.allclose(rpsd_g2l, welch_g2l)


def test_estimatesignalfrequency():
    # we test if we are able to find the frequency on a sinusoid drowned in a gaussian uncorrelated noise
    # with signal / noise ratio snr
    snr = 0.25
    x = np.linspace(0, 10, 2048, endpoint=False)
    f_true = 0.812
    z = np.sin(2*np.pi*f_true*x) + np.random.randn(len(x)) / snr
    f_est = ft.estimatesignalfrequency(z, x=x,
    window='boxcar', zero_pad_factor=8, bounds=None)

    max_acceptable_error = 1/ft.span(x)
    assert np.abs(f_true - f_est) < max_acceptable_error


### MONOGENIC SIGNAL (analytic signal)

@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("symmetrize", [False, True])
def test_monogenic1d_wraps_hilbert(N, symmetrize):
    t, sig = generate_sig1D(N)
    amplitude, phase = ft.monogenic1d(sig, symmetrize=symmetrize)
    analytic = ft.hilbert(sig, symmetrize=symmetrize)
    assert np.allclose(amplitude, np.abs(analytic)) and np.allclose(phase, np.angle(analytic))
    # sig = amplitude * cos(phase)
    assert np.allclose(amplitude * np.cos(phase), sig - sig.mean())
    assert (amplitude >= 0).all() and (np.abs(phase) <= np.pi).all()


@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("k", [1, 3, -5])
def test_monogenic1d_wave(N, k):
    # on-grid wave: constant amplitude, linear phase
    # (for k < 0, the wave is written with a positive frequency: cos(-2 pi |k| n/N + 0.7) = cos(2 pi |k| n/N - 0.7))
    n = np.arange(N)
    phase_wave = 2*np.pi*k*n/N + 0.7
    phase_expected = np.sign(k) * phase_wave
    amplitude, phase = ft.monogenic1d(1.3 * np.cos(phase_wave))
    assert np.allclose(amplitude, 1.3)
    assert np.allclose(np.angle(np.exp(1j*(phase - phase_expected))), 0, atol=1e-10)


@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("k", [1, 3, -5])
def test_local_frequency1d_wave(N, k):
    x = np.arange(N) * 0.13
    sig = np.cos(2*np.pi*k*np.arange(N)/N + 0.7)
    assert np.allclose(ft.local_frequency1d(sig, x=x), abs(k) / ft.span(x))
    assert np.allclose(ft.local_frequency1d(sig), abs(k) / N)


@pytest.mark.parametrize("symmetrize", [False, True])
def test_local_frequency1d_chirp(symmetrize):
    # linear chirp: the local frequency is f0 + alpha x
    N = 512
    x = np.arange(N) * 0.02
    f0, alpha = 2., 0.8
    sig = np.cos(2*np.pi*(f0*x + alpha*x**2/2))
    local_frequency = ft.local_frequency1d(sig, x=x, symmetrize=symmetrize)
    inside = slice(N//8, -N//8)  # away from the boundary effects of the transform
    assert np.allclose(local_frequency[inside], (f0 + alpha*x)[inside], rtol=0.01)


def test_local_frequency1d_irregular_x():
    # the derivative is taken with respect to x, even if it is irregularly spaced
    N = 256
    sig = np.cos(2*np.pi*4*np.arange(N)/N)
    x = np.arange(N) * 0.1
    x_irregular = x + 0.02*np.sin(2*np.pi*np.arange(N)/N)
    assert np.allclose(ft.local_frequency1d(sig, x=x_irregular),
                       np.gradient(2*np.pi*4*np.arange(N)/N, x_irregular) / (2*np.pi))

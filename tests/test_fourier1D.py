import g2ltk.fourier as fourier
import numpy as np
from scipy import fft

np.random.seed(19970510)

def test_dual():
    t = np.linspace(1.23, 2.344, 103)
    f = fourier.dual1d(t)
    f_ = fft.fftshift(np.fft.fftfreq(len(t), fourier.step(t)))
    assert np.all(f == f_)


def test_estimatesignalfrequency():
    # we test if we are able to find the frequency ona sinusoid drowned in a gaussian uncorrelated noise
    # with signal / noise ratio snr
    snr = 0.5
    x = np.linspace(0, 10, 1000)
    f_true = 0.812
    z = np.sin(2*np.pi*f_true*x) + np.random.randn(len(x)) / snr
    f_est = fourier.estimatesignalfrequency(z, x=x,
    window='boxcar', zero_pad_factor=4, bounds=None)

    max_acceptable_error = 1/fourier.span(x)
    assert np.abs(f_true - f_est) < max_acceptable_error

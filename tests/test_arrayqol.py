import g2ltk.fourier as fourier
import numpy as np
from scipy import fft

def test_dual():
    t = np.linspace(1.23, 2.344, 103)
    f = fourier.dual1d(t)
    f_ = fft.fftshift(np.fft.fftfreq(len(t), fourier.step(t)))
    assert np.all(f == f_)



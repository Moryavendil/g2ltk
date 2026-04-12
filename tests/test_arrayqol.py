import g2ltk.fourier as fourier
import numpy as np

def test_step():
    dx = 1.234
    arr:fourier.floatarray1D = np.arange(0, 2, dx)
    assert np.isclose(fourier.step(arr),  dx)

def test_span():
    valmin, targetspan = -3467.34, 238.239
    arr:fourier.floatarray1D = np.linspace(valmin, valmin + targetspan, 199)
    assert np.isclose(fourier.span(arr),  targetspan)



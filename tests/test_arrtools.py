import numpy as np

from g2ltk import arrtools

def test_step():
    dx = 1.234
    arr = np.arange(0, 2, dx)
    assert np.isclose(arrtools.step(arr),  dx)

def test_span():
    valmin, targetspan = -3467.34, 238.239
    arr = np.linspace(valmin, valmin + targetspan, 199, endpoint=False)
    assert np.isclose(arrtools.span(arr),  targetspan)

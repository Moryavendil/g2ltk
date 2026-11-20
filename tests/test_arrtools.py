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


import pytest

from g2ltk import customlog


@pytest.fixture(autouse=True)
def info_logs():
    # The tests below check the logged warnings and errors
    previous_verbose = customlog.__VERBOSE__
    customlog.set_verbose('info')
    yield
    customlog.__VERBOSE__ = previous_verbose


### REGULAR SPACING, STEP AND SPAN

@pytest.mark.parametrize('arr, expected', [
    (np.arange(5.), True),
    (np.array([3., 2., 1.]), False),  # decreasing
    (np.array([0., 1., 3.]), False),
    (np.array([1.]), False),  # spacing not defined
    (np.array([]), False),
])
def test_is_regularly_spaced(arr, expected):
    assert arrtools.is_regularly_spaced(arr) == expected


def test_step_none():
    assert arrtools.step(None) == 1.


@pytest.mark.parametrize('arr', [np.array([2.]), np.array([])])
def test_step_of_less_than_two_points(arr, capsys):
    assert arrtools.step(arr) == 1.
    assert 'ERROR' in capsys.readouterr().out


def test_step_of_constant_array(capsys):
    with np.errstate(all='raise'):  # no division by zero
        assert arrtools.step(np.ones(4)) == 0.
    assert 'WARN' in capsys.readouterr().out


def test_step_of_decreasing_array(capsys):
    assert arrtools.step(np.array([3., 2., 1., 0.])) == pytest.approx(-1.)
    assert 'WARN' not in capsys.readouterr().out  # regularly spaced, only decreasing


def test_step_of_irregular_array(capsys):
    assert arrtools.step(np.array([0., 1., 3.])) == pytest.approx(1.5)
    assert 'WARN' in capsys.readouterr().out


def test_span_is_one_step_more_than_max_minus_min():
    assert arrtools.span(np.arange(5.)) == pytest.approx(5.)


def test_span_none_warns(capsys):
    assert arrtools.span(None) == 1.
    assert 'WARN' in capsys.readouterr().out


def test_argval():
    arr = np.array([0., 0.5, 2., 3.])
    assert arrtools.argval(arr, 1.9) == 2
    assert arrtools.argval(arr, -10) == 0


### ROOTS

def test_interp_roots():
    x = np.array([0., 1., 2., 3., 4.])
    y = np.array([-1., 1., 0., -3., 1.])  # sign change in [0, 1], exact zero at 2, sign change in [3, 4]
    np.testing.assert_allclose(arrtools.interp_roots(y, x), [0.5, 2., 3.75])


def test_interp_roots_default_x():
    np.testing.assert_allclose(arrtools.interp_roots(np.array([-1., 3.])), [0.25])


def test_interp_unexplicit_roots_default_x():
    np.testing.assert_allclose(arrtools.interp_unexplicit_roots(np.array([-1., 3.])), [0.25])


### DERIVATIVES

def test_der_default_x():
    y = np.arange(6.) ** 2
    np.testing.assert_allclose(arrtools.der(y), np.gradient(y))
    np.testing.assert_allclose(arrtools.der(y, order=2), 2.)
    np.testing.assert_allclose(arrtools.der(y, acc=2), arrtools.der(y, x=np.arange(6.), acc=2))


def test_der_with_grid():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(arrtools.der(x ** 2, x), np.gradient(x ** 2, x))
    np.testing.assert_allclose(arrtools.der(x ** 2, x)[1:-1], 2 * x[1:-1])  # exact inside, 1st order at the edges
    np.testing.assert_allclose(arrtools.der(x ** 2, x, order=2), 2.)
    np.testing.assert_allclose(arrtools.der(x ** 2, x[1] - x[0], order=2), 2.)  # grid spacing instead of grid


### EXTREMA AND PEAKS

def test_find_extrema():
    x = np.linspace(0, 2 * np.pi, 1001)
    y = np.sin(x)
    minimums, maximums = arrtools.find_extrema(y, x, peak_category='minmax')
    np.testing.assert_allclose(maximums, [np.pi / 2], atol=1e-3)
    np.testing.assert_allclose(minimums, [3 * np.pi / 2], atol=1e-3)


def test_find_global_max_subpixel():
    x = np.arange(20.)
    y = arrtools.gaussian(x, mu=7.3, sigma=2.)
    assert arrtools.find_global_max(y, x) == pytest.approx(7.3, abs=0.05)


def test_find_global_peak_categories():
    x = np.arange(20.)
    y = arrtools.gaussian_unnormalized(x, mu=5.2, sigma=2.) - 2 * arrtools.gaussian_unnormalized(x, mu=13.6, sigma=2.)
    assert arrtools.find_global_peak(y, x, peak_category='max') == pytest.approx(5.2, abs=0.05)
    assert arrtools.find_global_peak(y, x, peak_category='min') == pytest.approx(13.6, abs=0.05)
    assert arrtools.find_global_peak(y, x, peak_category='raw') == pytest.approx(13.6, abs=0.05)  # largest |y|


def test_find_global_peak_unsupported_category(capsys):
    assert arrtools.find_global_peak(np.arange(5.), peak_category='foo') is None
    output = capsys.readouterr().out
    assert 'ERROR' in output and 'foo' in output


### EXTENT

def test_correct_extent():
    x, y = np.arange(4.), np.arange(3.)
    assert arrtools.correct_extent(x, y, origin='upper') == (-0.5, 3.5, 2.5, -0.5)
    assert arrtools.correct_extent(x, y, origin='lower') == (-0.5, 3.5, -0.5, 2.5)


def test_correct_extent_unsupported_origin(capsys):
    assert arrtools.correct_extent(np.arange(4.), np.arange(3.), origin='Upper') is None
    assert 'ERROR' in capsys.readouterr().out


### DEPRECATED ALIAS

def test_peakfinder_is_an_alias_of_arrtools():
    from g2ltk import peakfinder
    assert peakfinder.step is arrtools.step
    assert peakfinder.find_global_max is arrtools.find_global_max

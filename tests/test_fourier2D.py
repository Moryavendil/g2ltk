import g2ltk.fourier as ft
from g2ltk.fourier.FFTHelp import split_2d_argument, sanitize_nfft_2d, sanitize_zpf_2d
import numpy as np
import pytest
from scipy import fft
from scipy import signal
from scipy.signal.windows import get_window
from skimage import filters

np.random.seed(19970510)

# Convention: arrays are shaped (len(y), len(x)), i.e. axis 0 <-> y and axis 1 <-> x.
# x and y deliberately have different steps and lengths, so that any x/y swap is detected.

def generate_sig2D(shape):
    N0, N1 = shape
    y = np.linspace(0., 15.75234, N0, endpoint=False)  # axis 0
    x = np.linspace(0., 7.31, N1, endpoint=False)  # axis 1
    Y, X = np.meshgrid(y, x, indexing='ij')
    fx1, fy1, phi1 = 0.8, 0.3, 0.4
    fx2, fy2, phi2 = 1.7, 0.9, 0.7

    a1 = 1.
    a2 = 0.6

    anoise = 0.1

    sig = a1*np.cos(2*np.pi*(fx1*X + fy1*Y) + phi1) \
          + a2*np.cos(2*np.pi*fx2*X + phi2)*np.cos(2*np.pi*fy2*Y) \
          + anoise*np.random.rand(N0, N1)

    return x, y, sig


def as_pair(value):
    if value is None or isinstance(value, (int, str)):
        return value, value
    return tuple(value)


def final_shape(shape, nfft=None, zpf=None):
    """The expected shape of the transform, computed independently of g2ltk."""
    out = []
    for N, nf, zp in zip(shape, as_pair(nfft), as_pair(zpf)):
        if nf is None:
            nf = N
        elif nf == 'auto':
            nf = fft.next_fast_len(N)
        out.append(nf * (1 if zp is None else zp))
    return tuple(out)


def outer_window(window, shape):
    return np.outer(get_window(window, shape[0]), get_window(window, shape[1]))


def circular_window(window, shape):
    return filters.window(window, shape, warp_kwargs={'order': 3})


def periodogram2d(sig, x, y, window, zpf=None, onesided=False):
    """Reference 2-D periodogram (outer window, global mean removal), built on scipy.signal.periodogram."""
    n0, n1 = final_shape(sig.shape, zpf=zpf)
    w0 = get_window(window, sig.shape[0])
    sig_d = sig - sig.mean()
    # transform along y (axis 0) by hand, with the periodogram normalization
    sig_y = fft.fft(sig_d * w0[:, None], n=n0, axis=0)
    # then let scipy do the periodogram along x (axis 1)
    _, p = signal.periodogram(sig_y, fs=1/ft.step(x), window=window, nfft=n1, axis=1,
                              return_onesided=False, scaling='density', detrend=False)
    p = p * ft.step(y) / np.sum(w0**2)
    if onesided:
        # fold the negative x-frequencies: double everything except kx = 0 and the Nyquist bin (if n1 is even)
        p = p[:, :n1//2 + 1].copy()
        p[:, 1:(n1 + 1)//2] *= 2
        return fft.fftshift(p, axes=0)
    return fft.fftshift(p)


shapes_to_test = [(20, 32), (63, 31), (64, 63), (17, 256), (256, 17)]  # even/odd, square-ish, very elongated both ways
window_to_test = ['boxcar', 'hann', 'hamming', 'tukey', 'blackman', 'flattop']
winstyle_to_test = ['outer', 'circular']
zpf2d_to_test = [1, 2, 3, (1, 2), (3, 1), (2, 4)]  # same on both axes, and different on each axis
zpf_psd_to_test = [1, 3, (1, 2)]
nfft2d_to_test = [None, 'auto', 64, (64, 'auto'), ('auto', 50), (30, 100)]  # includes truncation (30 < 63)
shifts_to_test = [0, 3, (0, 5), (2, 0), (-1, 4)]
modes_to_test = [(1, 3), (4, 0), (0, 2), (5, 7)]  # (kx, ky): integer number of periods along x, y

transform_functions = ['ft2d', 'rft2d', 'psd2d', 'rpsd2d']
dual_functions = ['dual2d', 'rdual2d']


def call_2d(fname, x, y, sig, **kwargs):
    func = getattr(ft, fname)
    if fname in dual_functions:
        return func(x, y, **kwargs)
    return func(sig, x=x, y=y, **kwargs)


def assert_same_result(res1, res2):
    if isinstance(res1, tuple):
        assert len(res1) == len(res2)
        for r1, r2 in zip(res1, res2):
            assert r1.shape == r2.shape
            assert np.isclose(r1, r2).all()
    else:
        assert res1.shape == res2.shape
        assert np.isclose(res1, res2).all()


### ARGUMENT HANDLING

@pytest.mark.parametrize("value, expected", [
    (None, (None, None)),
    ('auto', ('auto', 'auto')),
    (3, (3, 3)),
    (np.int64(3), (3, 3)),
    ((2, 3), (2, 3)),
    ([2, 3], (2, 3)),
    (np.array([2, 3]), (2, 3)),
    (('auto', 4), ('auto', 4)),
    ((None, 2), (None, 2)),
    ((np.int32(5), 'auto'), (5, 'auto')),
    ((1, 2, 3), (None, None)),
    (2.5, (None, None)),
])
def test_split_2d_argument(value, expected):
    result = split_2d_argument(value)
    assert result == expected
    for v in result:
        # numpy integers must be converted to python ints (sanitize_*_1d only accept int)
        if v is not None and not isinstance(v, str):
            assert type(v) is int


@pytest.mark.parametrize("nfft, expected", [
    (None, (63, 31)),
    ('auto', (fft.next_fast_len(63), fft.next_fast_len(31))),
    (50, (50, 50)),
    ((None, 'auto'), (63, fft.next_fast_len(31))),
    ((100, None), (100, 31)),
    (np.int64(40), (40, 40)),
    ('banana', (63, 31)),
    ((1, 2, 3), (63, 31)),
])
def test_sanitize_nfft_2d(nfft, expected):
    assert sanitize_nfft_2d(nfft, (63, 31)) == expected


@pytest.mark.parametrize("zpf, expected", [
    (None, (1, 1)),
    (2, (2, 2)),
    ((3, None), (3, 1)),
    (np.int64(4), (4, 4)),
    (2.0, (1, 1)),
    ((2, 3), (2, 3)),
])
def test_sanitize_zpf_2d(zpf, expected):
    assert sanitize_zpf_2d(zpf) == expected


@pytest.mark.parametrize("fname", dual_functions + transform_functions)
@pytest.mark.parametrize("zpf", [2, (1, 3)])
def test_zero_pad_factor_alias(fname, zpf):
    x, y, sig = generate_sig2D((20, 31))
    res_zpf = call_2d(fname, x, y, sig, zpf=zpf)
    res_alias = call_2d(fname, x, y, sig, zero_pad_factor=zpf)
    assert_same_result(res_zpf, res_alias)


@pytest.mark.parametrize("fname", dual_functions + transform_functions)
def test_zpf_and_alias_together_raise(fname):
    x, y, sig = generate_sig2D((20, 31))
    with pytest.raises(TypeError):
        call_2d(fname, x, y, sig, zpf=2, zero_pad_factor=2)


@pytest.mark.parametrize("fname", dual_functions + transform_functions)
def test_zero_pad_argument_removed(fname):
    # zero_pad (number of points added) was replaced by nfft (total length): it must fail loudly
    x, y, sig = generate_sig2D((20, 31))
    with pytest.raises(TypeError):
        call_2d(fname, x, y, sig, zero_pad=(2, 2))


### DUALS

@pytest.mark.parametrize("shape", shapes_to_test)
def test_dual2d(shape):
    x, y, sig = generate_sig2D(shape)
    fx, fy = ft.dual2d(x, y)

    assert np.isclose(fx, fft.fftshift(fft.fftfreq(len(x), x[1]-x[0]))).all()
    assert np.isclose(fy, fft.fftshift(fft.fftfreq(len(y), y[1]-y[0]))).all()


@pytest.mark.parametrize("shape", shapes_to_test)
def test_rdual2d(shape):
    x, y, sig = generate_sig2D(shape)
    fx, fy = ft.rdual2d(x, y)

    assert np.isclose(fx, fft.rfftfreq(len(x), x[1]-x[0])).all()
    assert np.isclose(fy, fft.fftshift(fft.fftfreq(len(y), y[1]-y[0]))).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", zpf2d_to_test)
def test_dual2d_zpf(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    n0, n1 = final_shape(shape, zpf=zpf)

    fx, fy = ft.dual2d(x, y, zpf=zpf)
    assert np.isclose(fx, fft.fftshift(fft.fftfreq(n1, x[1]-x[0]))).all()
    assert np.isclose(fy, fft.fftshift(fft.fftfreq(n0, y[1]-y[0]))).all()

    rfx, rfy = ft.rdual2d(x, y, zpf=zpf)
    assert np.isclose(rfx, fft.rfftfreq(n1, x[1]-x[0])).all()
    assert np.isclose(rfy, fft.fftshift(fft.fftfreq(n0, y[1]-y[0]))).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("nfft", nfft2d_to_test)
def test_dual2d_nfft(shape, nfft):
    x, y, sig = generate_sig2D(shape)
    n0, n1 = final_shape(shape, nfft=nfft)

    fx, fy = ft.dual2d(x, y, nfft=nfft)
    assert np.isclose(fx, fft.fftshift(fft.fftfreq(n1, x[1]-x[0]))).all()
    assert np.isclose(fy, fft.fftshift(fft.fftfreq(n0, y[1]-y[0]))).all()

    rfx, rfy = ft.rdual2d(x, y, nfft=nfft)
    assert np.isclose(rfx, fft.rfftfreq(n1, x[1]-x[0])).all()
    assert np.isclose(rfy, fft.fftshift(fft.fftfreq(n0, y[1]-y[0]))).all()


### FT

@pytest.mark.parametrize("shape", shapes_to_test)
def test_ft2d(shape):
    x, y, sig = generate_sig2D(shape)
    sig_ft_scipy = fft.fft2(sig)
    sig_ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=False)

    assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(x) * ft.step(y), sig_ft_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
def test_ft2d_complex(shape):
    x, y, sig = generate_sig2D(shape)
    _, _, sig2 = generate_sig2D(shape)
    sig_c = sig + 1j * sig2
    sig_ft_scipy = fft.fft2(sig_c)
    sig_ft_g2l = ft.ft2d(sig_c, x=x, y=y, remove_mean=False)

    assert np.isclose(fft.fftshift(sig_ft_scipy) * ft.step(x) * ft.step(y), sig_ft_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
def test_rft2d(shape):
    x, y, sig = generate_sig2D(shape)
    sig_ft_scipy = fft.rfft2(sig)
    sig_ft_g2l = ft.rft2d(sig, x=x, y=y, remove_mean=False)

    assert np.isclose(fft.fftshift(sig_ft_scipy, axes=0) * ft.step(x) * ft.step(y), sig_ft_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", zpf2d_to_test)
def test_ft2d_zpf(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    s = final_shape(shape, zpf=zpf)

    ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=False, zpf=zpf)
    assert np.isclose(fft.fftshift(fft.fft2(sig, s=s)) * ft.step(x) * ft.step(y), ft_g2l).all()

    rft_g2l = ft.rft2d(sig, x=x, y=y, remove_mean=False, zpf=zpf)
    assert np.isclose(fft.fftshift(fft.rfft2(sig, s=s), axes=0) * ft.step(x) * ft.step(y), rft_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("nfft", nfft2d_to_test)
def test_ft2d_nfft(shape, nfft):
    # scipy's s= pads at the end, or truncates, exactly like g2ltk should
    x, y, sig = generate_sig2D(shape)
    s = final_shape(shape, nfft=nfft)

    ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=False, nfft=nfft)
    assert np.isclose(fft.fftshift(fft.fft2(sig, s=s)) * ft.step(x) * ft.step(y), ft_g2l).all()

    rft_g2l = ft.rft2d(sig, x=x, y=y, remove_mean=False, nfft=nfft)
    assert np.isclose(fft.fftshift(fft.rfft2(sig, s=s), axes=0) * ft.step(x) * ft.step(y), rft_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("nfft", [None, 'auto', (30, 100)])
@pytest.mark.parametrize("zpf", [1, (2, 3)])
def test_ft2d_shapes_match_duals(shape, nfft, zpf):
    x, y, sig = generate_sig2D(shape)
    fx, fy = ft.dual2d(x, y, nfft=nfft, zpf=zpf)
    rfx, rfy = ft.rdual2d(x, y, nfft=nfft, zpf=zpf)

    assert ft.ft2d(sig, x=x, y=y, nfft=nfft, zpf=zpf).shape == (len(fy), len(fx))
    assert ft.psd2d(sig, x=x, y=y, nfft=nfft, zpf=zpf).shape == (len(fy), len(fx))
    assert ft.rft2d(sig, x=x, y=y, nfft=nfft, zpf=zpf).shape == (len(rfy), len(rfx))
    assert ft.rpsd2d(sig, x=x, y=y, nfft=nfft, zpf=zpf).shape == (len(rfy), len(rfx))


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", [1, 2, (1, 3)])
def test_ift2d(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    fx, fy = ft.dual2d(x, y, zpf=zpf)
    sig_ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=False, zpf=zpf)
    sig_back = ft.ift2d(sig_ft_g2l, fx, fy)

    N0, N1 = shape
    assert np.isclose(sig, sig_back[:N0, :N1]).all()
    # the zero-padding comes back as zeros
    padding = np.ones(sig_back.shape, dtype=bool)
    padding[:N0, :N1] = False
    assert np.isclose(sig_back[padding], 0).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("shift", shifts_to_test)
def test_ft2d_shift(shape, shift):
    x, y, sig = generate_sig2D(shape)
    s0, s1 = as_pair(shift)

    # without padding, shifting is rolling the signal
    ft_shifted = ft.ft2d(sig, x=x, y=y, remove_mean=False, shift=shift)
    ft_rolled = ft.ft2d(np.roll(sig, (-s0, -s1), axis=(0, 1)), x=x, y=y, remove_mean=False)
    assert np.isclose(ft_shifted, ft_rolled).all()

    # with or without padding, shifting is a phase ramp in Fourier space
    for zpf in [1, 2]:
        fx, fy = ft.dual2d(x, y, zpf=zpf)
        ft_ref = ft.ft2d(sig, x=x, y=y, remove_mean=False, zpf=zpf)
        ft_shifted = ft.ft2d(sig, x=x, y=y, remove_mean=False, zpf=zpf, shift=shift)
        ramp = np.exp(2j*np.pi*(fy[:, None] * s0 * ft.step(y) + fx[None, :] * s1 * ft.step(x)))
        assert np.isclose(ft_shifted, ft_ref * ramp).all()


@pytest.mark.parametrize("shape", shapes_to_test)
def test_ft2d_remove_mean(shape):
    x, y, sig = generate_sig2D(shape)
    ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=True)

    assert np.isclose(ft_g2l, ft.ft2d(sig - sig.mean(), x=x, y=y, remove_mean=False)).all()
    # the (0, 0) frequency is at the center after fftshift
    assert np.abs(ft_g2l[shape[0]//2, shape[1]//2]) < 1e-10 * np.abs(ft_g2l).max()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
def test_ft2d_separable(shape, window):
    # the transform of a(y) b(x) is the outer product of the 1D transforms (also with outer windowing)
    x, y, _ = generate_sig2D(shape)
    a = np.cos(2*np.pi*0.37*y + 0.2) + 0.1*np.random.rand(len(y))
    b = np.sin(2*np.pi*1.13*x + 0.5) + 0.1*np.random.rand(len(x))
    sig = np.outer(a, b)

    ft_g2l = ft.ft2d(sig, x=x, y=y, window=window, remove_mean=False)
    ft_1d = np.outer(ft.ft1d(a, x=y, window=window, remove_mean=False),
                     ft.ft1d(b, x=x, window=window, remove_mean=False))

    assert np.isclose(ft_g2l, ft_1d).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("mode", modes_to_test)
def test_plane_wave(shape, mode):
    # an on-grid plane wave must give exactly two peaks, at the right place (x <-> axis 1), with the right amplitude
    N0, N1 = shape
    x = np.arange(N1) * 0.13
    y = np.arange(N0) * 0.41
    Y, X = np.meshgrid(y, x, indexing='ij')
    kx, ky = mode
    fx0, fy0 = kx / ft.span(x), ky / ft.span(y)
    amplitude = 1.7
    sig = amplitude * np.cos(2*np.pi*(fx0*X + fy0*Y) + 0.3)

    # full transform
    fx, fy = ft.dual2d(x, y)
    ft_g2l = ft.ft2d(sig, x=x, y=y)
    peaks = {tuple(p) for p in np.argwhere(np.abs(ft_g2l) > 0.5 * np.abs(ft_g2l).max())}
    expected = {(np.argmin(np.abs(fy - fy0)), np.argmin(np.abs(fx - fx0))),
                (np.argmin(np.abs(fy + fy0)), np.argmin(np.abs(fx + fx0)))}
    assert peaks == expected
    assert np.isclose(np.abs(ft_g2l).max(), amplitude / 2 * ft.span(x) * ft.span(y))

    # truncated transform: only the peak with fx >= 0 is kept (both if fx = 0)
    rfx, rfy = ft.rdual2d(x, y)
    rft_g2l = ft.rft2d(sig, x=x, y=y)
    rpeaks = {tuple(p) for p in np.argwhere(np.abs(rft_g2l) > 0.5 * np.abs(rft_g2l).max())}
    rexpected = {(np.argmin(np.abs(rfy - fy0)), np.argmin(np.abs(rfx - fx0)))}
    if kx == 0:
        rexpected.add((np.argmin(np.abs(rfy + fy0)), 0))
    assert rpeaks == rexpected

    # power: the PSD integrates to the mean square amplitude A^2 / 2
    psd = ft.psd2d(sig, x=x, y=y)
    assert np.isclose(psd.sum() * ft.step(fx) * ft.step(fy), amplitude**2 / 2, rtol=1e-9, atol=0)
    rpsd = ft.rpsd2d(sig, x=x, y=y)
    assert np.isclose(rpsd.sum() * ft.step(rfx) * ft.step(rfy), amplitude**2 / 2, rtol=1e-9, atol=0)


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", ['boxcar', 'hann'])
def test_transpose(shape, window):
    # exchanging the roles of x and y must transpose the result
    x, y, sig = generate_sig2D(shape)

    assert np.isclose(ft.ft2d(sig.T, x=y, y=x, window=window, zpf=(2, 1)),
                      ft.ft2d(sig, x=x, y=y, window=window, zpf=(1, 2)).T).all()
    assert np.isclose(ft.psd2d(sig.T, x=y, y=x, window=window, zpf=(2, 1)),
                      ft.psd2d(sig, x=x, y=y, window=window, zpf=(1, 2)).T).all()
    fx, fy = ft.dual2d(x, y, zpf=(1, 2))
    fy_t, fx_t = ft.dual2d(y, x, zpf=(2, 1))
    assert np.isclose(fx, fx_t).all() and np.isclose(fy, fy_t).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", [1, (2, 3)])
def test_psd2d_real_signal_symmetry(shape, zpf):
    # for a real signal, PSD(f) = PSD(-f)
    x, y, sig = generate_sig2D(shape)
    psd = fft.ifftshift(ft.psd2d(sig, x=x, y=y, window='hann', zpf=zpf))
    psd_opposite = np.roll(psd[::-1, ::-1], 1, axis=(0, 1))

    assert np.isclose(psd, psd_opposite).all()


### WINDOWS

@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
def test_window_factor2d(shape, window):
    wf_expected = 1 / np.mean(outer_window(window, shape)**2)

    assert np.isclose(ft.window_factor2d(window, winstyle='outer', NxNy=shape), wf_expected)
    assert np.isclose(ft.window_factor2d(window, NxNy=shape), wf_expected)  # outer is the default
    assert np.isclose(ft.window_factor2d(window, NxNy=shape),
                      ft.window_factor1d(window, shape[0]) * ft.window_factor1d(window, shape[1]))


def test_window_factor2d_unknown_winstyle():
    assert ft.window_factor2d('hann', winstyle='banana', NxNy=(20, 31)) == 1


def test_ft2d_unknown_winstyle():
    # an unknown winstyle means no windowing at all
    x, y, sig = generate_sig2D((20, 31))
    assert np.isclose(ft.ft2d(sig, x=x, y=y, window='hann', winstyle='banana'),
                      ft.ft2d(sig, x=x, y=y, window='boxcar')).all()


### PSD

@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("winstyle", winstyle_to_test)
def test_psd2d_definition(shape, window, winstyle):
    x, y, sig = generate_sig2D(shape)

    ft_g2l = ft.ft2d(sig, x=x, y=y, remove_mean=False, window=window, winstyle=winstyle)
    psd_g2l_alter = np.abs(ft_g2l)**2 * ft.window_factor2d(window, winstyle=winstyle, NxNy=shape) \
                    / (len(x)*ft.step(x) * len(y)*ft.step(y))

    psd_g2l = ft.psd2d(sig, x=x, y=y, remove_mean=False, window=window, winstyle=winstyle)

    assert np.isclose(psd_g2l_alter, psd_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zpf", zpf_psd_to_test)
def test_psd2d_periodogram(shape, window, zpf):
    x, y, sig = generate_sig2D(shape)
    psd_g2l = ft.psd2d(sig, x=x, y=y, remove_mean=True, window=window, zpf=zpf)
    psd_ref = periodogram2d(sig, x, y, window=window, zpf=zpf)

    assert psd_ref.shape == psd_g2l.shape
    assert np.isclose(psd_ref, psd_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("zpf", zpf_psd_to_test)
def test_rpsd2d_periodogram(shape, window, zpf):
    x, y, sig = generate_sig2D(shape)
    rpsd_g2l = ft.rpsd2d(sig, x=x, y=y, remove_mean=True, window=window, zpf=zpf)
    rpsd_ref = periodogram2d(sig, x, y, window=window, zpf=zpf, onesided=True)

    n0, n1 = final_shape(shape, zpf=zpf)
    rfx, rfy = ft.rdual2d(x, y, zpf=zpf)
    assert rpsd_ref.shape == rpsd_g2l.shape
    assert np.isclose(rfx, fft.rfftfreq(n1, ft.step(x))).all()
    assert np.isclose(rfy, fft.fftshift(fft.fftfreq(n0, ft.step(y)))).all()
    assert np.isclose(rpsd_ref, rpsd_g2l).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("winstyle", winstyle_to_test)
def test_psd2d_parseval(shape, window, winstyle):
    # integrating the PSD gives back the (window-corrected) mean square of the signal, whatever the padding
    # (exact identity, hence the tight tolerance)
    x, y, sig = generate_sig2D(shape)
    W = outer_window(window, shape) if winstyle == 'outer' else circular_window(window, shape)
    expected = np.mean(((sig - sig.mean()) * W)**2) / np.mean(W**2)

    for zpf in [1, (2, 3)]:
        fx, fy = ft.dual2d(x, y, zpf=zpf)
        psd = ft.psd2d(sig, x=x, y=y, window=window, winstyle=winstyle, zpf=zpf)
        assert np.isclose(psd.sum() * ft.step(fx) * ft.step(fy), expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", zpf2d_to_test)
def test_rpsd2d_parseval(shape, zpf):
    # the truncated PSD must hold the same total power as the full one (even and odd lengths)
    # (exact identity: the tight tolerance matters, the Nyquist column holds little power for long signals)
    x, y, sig = generate_sig2D(shape)
    rfx, rfy = ft.rdual2d(x, y, zpf=zpf)
    rpsd = ft.rpsd2d(sig, x=x, y=y, zpf=zpf)

    assert np.isclose(rpsd.sum() * ft.step(rfx) * ft.step(rfy), np.mean((sig - sig.mean())**2), rtol=1e-9, atol=0)


@pytest.mark.parametrize("window", window_to_test)
@pytest.mark.parametrize("winstyle", winstyle_to_test)
def test_psd2d_white_noise_level(window, winstyle):
    # statistical test: on white noise, the windowed PSD must be an unbiased estimator of the variance
    # (this catches a wrong window correction factor, e.g. squared)
    rng = np.random.default_rng(19970510)
    shape = (128, 96)
    x = np.arange(shape[1]) * 0.02
    y = np.arange(shape[0]) * 0.1
    sigma = 3.
    fx, fy = ft.dual2d(x, y)
    powers = []
    for _ in range(20):
        sig = sigma * rng.standard_normal(shape)
        psd = ft.psd2d(sig, x=x, y=y, window=window, winstyle=winstyle)
        powers.append(psd.sum() * ft.step(fx) * ft.step(fy))

    assert np.isclose(np.mean(powers), sigma**2, rtol=0.05)


### GENERIC DISPATCHERS (ft, psd, dual)

@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", [None, 2, (1, 3)])
def test_ft_dispatch_2d(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    assert np.isclose(ft.ft(sig, x=x, y=y, zpf=zpf), ft.ft2d(sig, x=x, y=y, zpf=zpf)).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", [None, 2, (1, 3)])
def test_psd_dispatch_2d(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    assert np.isclose(ft.psd(sig, x=x, y=y, zpf=zpf, window='hann'),
                      ft.psd2d(sig, x=x, y=y, zpf=zpf, window='hann')).all()


@pytest.mark.parametrize("shape", shapes_to_test)
@pytest.mark.parametrize("zpf", [None, 2, (1, 3)])
def test_dual_dispatch_2d(shape, zpf):
    x, y, sig = generate_sig2D(shape)
    assert_same_result(ft.dual(x, y, zpf=zpf), ft.dual2d(x, y, zpf=zpf))

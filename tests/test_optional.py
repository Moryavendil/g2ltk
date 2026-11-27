"""
Tests for the check of the optional dependencies of the subpackages (g2ltk._optional).
"""
import pytest

from g2ltk._optional import require_extra


def test_installed_dependencies_pass():
    require_extra('g2ltk.fourier', 'whatever', {'numpy': 'numpy', 'scipy': 'scipy'})


def test_missing_dependencies_name_the_packages_and_the_extra():
    with pytest.raises(ModuleNotFoundError) as excinfo:
        require_extra('g2ltk.videoreading', 'video',
                      {'numpy': 'numpy', 'g2ltk_missing_a': 'missing-a', 'g2ltk_missing_b': 'missing-b'})
    message = str(excinfo.value)
    assert 'g2ltk.videoreading needs missing-a, missing-b,' in message
    assert 'pip install "g2ltk[video]"' in message
    assert 'numpy' not in message


def test_missing_dependencies_are_an_import_error():
    # code that catches ImportError around an import of g2ltk.videoreading still works
    with pytest.raises(ImportError):
        require_extra('g2ltk.plotting', 'plot', {'g2ltk_missing': 'missing'})

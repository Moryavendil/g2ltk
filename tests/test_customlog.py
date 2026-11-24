"""
Tests for g2ltk.customlog: the verbosity of g2ltk, which can be set from g2ltk or from any of its subpackages.
"""
import importlib

import pytest

from g2ltk import customlog

SUBPACKAGES = ['g2ltk', 'g2ltk.videoreading', 'g2ltk.fourier', 'g2ltk.plotting', 'g2ltk.rivulets']


@pytest.fixture(autouse=True)
def restore_level():
    previous_level = customlog.logger.level
    yield
    customlog.logger.setLevel(previous_level)


@pytest.mark.parametrize('subpackage', SUBPACKAGES)
def test_set_verbose_from_subpackage(subpackage, capsys):
    module = importlib.import_module(subpackage)
    # a single logger for all of g2ltk: setting the verbosity from a subpackage applies everywhere
    module.set_verbose('error')
    customlog.log_warning('hidden warning')
    module.set_verbose('debug')
    customlog.log_debug('shown debug')
    output = capsys.readouterr().out
    assert 'hidden warning' not in output
    assert 'shown debug' in output


@pytest.mark.parametrize('subpackage', SUBPACKAGES)
def test_verbosity_from_subpackage(subpackage, capsys):
    module = importlib.import_module(subpackage)
    module.set_verbose('info')
    with module.verbosity('error'):
        customlog.log_info('hidden info')
    customlog.log_info('shown info')
    output = capsys.readouterr().out
    assert 'hidden info' not in output
    assert 'shown info' in output

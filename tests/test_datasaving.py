"""
Tests for the saved-data index of g2ltk.datasaving and g2ltk.rivulets.datasaving:
data is always saved and looked up under the acquisition name without its video extension.
"""
import os

import numpy as np
import pytest

from g2ltk import customlog
from g2ltk import datasaving
from g2ltk.rivulets import datasaving as rivulets_datasaving

MODULES = [datasaving, rivulets_datasaving]


@pytest.fixture(autouse=True)
def quiet_logs():
    with customlog.verbosity('error'):
        yield


@pytest.fixture(params=MODULES, ids=['datasaving', 'rivulets.datasaving'])
def ds(request, tmp_path, monkeypatch):
    """The datasaving module, saving into a temporary directory"""
    module = request.param
    save_directory = str(tmp_path / 'cache')
    monkeypatch.setattr(module, 'save_directory', save_directory)
    monkeypatch.setattr(module, 'index_path', os.path.join(save_directory, module.index_name + '.npz'))
    return module


@pytest.mark.parametrize('acquisition, expected', [
    ('acq', 'acq'),
    ('acq.mp4', 'acq'),
    ('acq.gcv', 'acq'),
    ('flow_1.5mlmin', 'flow_1.5mlmin'),
    ('flow_1.5mlmin.mkv', 'flow_1.5mlmin'),
])
def test_cache_parameters(ds, acquisition, expected):
    parameters = {'datatype': 'bos', 'dataset': 'ds', 'acquisition': acquisition}
    assert ds.cache_parameters(parameters) == {'datatype': 'bos', 'dataset': 'ds', 'acquisition': expected}
    assert parameters['acquisition'] == acquisition  # the given parameters are not modified


def test_cache_parameters_without_acquisition(ds):
    assert ds.cache_parameters({'datatype': 'bos'}) == {'datatype': 'bos'}
    assert ds.cache_parameters({'datatype': 'bos', 'acquisition': None}) == {'datatype': 'bos', 'acquisition': None}


def test_saved_under_the_name_without_extension(ds):
    parameters = {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq.mp4'}
    ds.save_data(np.arange(5), parameters)

    index = ds.get_index()
    assert len(index) == 1
    (filename, stored_parameters), = index.items()
    assert filename.endswith('-ds-acq.npz')
    assert stored_parameters['acquisition'] == 'acq'
    assert parameters['acquisition'] == 'acq.mp4'  # the caller's parameters are not modified


@pytest.mark.parametrize('saved_as, fetched_as', [
    ('acq.mp4', 'acq'), ('acq', 'acq.mp4'), ('acq.mp4', 'acq.gcv'), ('acq.gcv', 'acq.gcv'),
])
def test_fetch_with_or_without_extension(ds, saved_as, fetched_as):
    ds.save_data(np.arange(5), {'datatype': 'bos', 'dataset': 'ds', 'acquisition': saved_as})
    data = ds.fetch_saved_data({'datatype': 'bos', 'dataset': 'ds', 'acquisition': fetched_as})
    np.testing.assert_array_equal(data, np.arange(5))


def test_extension_does_not_create_a_second_entry(ds):
    ds.save_data(np.arange(5), {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq'})
    ds.save_data(np.arange(7), {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq.mp4'})
    index = ds.get_index()
    assert len(index) == 1  # the second save replaced the first
    np.testing.assert_array_equal(ds.fetch_saved_data({'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq'}),
                                  np.arange(7))


def test_other_acquisitions_are_not_matched(ds):
    ds.save_data(np.arange(5), {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq.mp4'})
    assert ds.fetch_saved_data({'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq2'}) is None
    assert ds.fetch_saved_data({'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq.mp42'}) is None


def test_entries_saved_with_an_extension_before_are_found(ds):
    # An index entry written before acquisitions were stored without extension
    ds.ensure_save_directory_exists()
    filename = 'old-bos-ds-acq.mp4.npz'
    np.savez(os.path.join(ds.save_directory, filename), data=np.arange(3))
    ds.set_index({filename: {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq.mp4'}})

    for acquisition in ['acq', 'acq.mp4']:
        data = ds.fetch_saved_data({'datatype': 'bos', 'dataset': 'ds', 'acquisition': acquisition})
        np.testing.assert_array_equal(data, np.arange(3))


def test_erase_items_with_extension(ds):
    ds.save_data(np.arange(5), {'datatype': 'bos', 'dataset': 'ds', 'acquisition': 'acq'})
    ds.erase_items({'datatype': 'bos', 'acquisition': 'acq.gcv'})
    assert ds.get_index() == {}


def test_fetch_or_generate_with_extension(tmp_path, monkeypatch):
    # Generic mode (no framenumbers) of g2ltk.datasaving: 'acq.mp4' reuses what was generated for 'acq'
    save_directory = str(tmp_path / 'cache')
    monkeypatch.setattr(datasaving, 'save_directory', save_directory)
    monkeypatch.setattr(datasaving, 'index_path', os.path.join(save_directory, 'index.npz'))
    calls = []

    def generate(parameters):
        calls.append(parameters['acquisition'])
        return np.arange(4)

    for acquisition in ['acq', 'acq.mp4']:
        data = datasaving.fetch_or_generate('bos', dataset='ds', acquisition=acquisition, data_generating_fn=generate)
        np.testing.assert_array_equal(data, np.arange(4))
    assert calls == ['acq']  # generated once, fetched the second time


def test_video_mode_is_disabled(tmp_path, monkeypatch):
    # With 'framenumbers' in the parameters, fetch_or_generate_data_from_parameters returns None without generating
    save_directory = str(tmp_path / 'cache')
    monkeypatch.setattr(datasaving, 'save_directory', save_directory)
    monkeypatch.setattr(datasaving, 'index_path', os.path.join(save_directory, 'index.npz'))
    calls = []

    def generate(parameters):
        calls.append(parameters)
        return np.arange(4)

    data = datasaving.fetch_or_generate('bos', dataset='ds', acquisition='acq',
                                        parameters={'framenumbers': None}, data_generating_fn=generate)
    assert data is None
    assert calls == []

"""
Tests for g2ltk.videoreading on the reference videos in test_data/:
    - test_mp4.mp4  : lossy compressed video (H.264)
    - test_mkv.mkv  : losslessly compressed video (FFV1)
    - test_gcv.gcv/ : GevCapture video (meta + stamps + raw video)
"""
import os
from functools import lru_cache

import numpy as np
import pytest

from g2ltk import customlog
from g2ltk import videoreading as vr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data')

# Camera timestamps of the second and last frames of the gcv reference video, in s (from its stamps file)
GCV_T1_S = 0.040003
GCV_TLAST_S = 0.800878

# Expected properties of the reference videos
EXPECTED = {
    'test_mp4': {'type': 'mp4', 'n_frames': 43, 'height': 256, 'width': 256, 'fps': 30.},
    'test_mkv': {'type': 'mkv', 'n_frames': 53, 'height': 256, 'width': 256, 'fps': 30.},
    # gcv frequency is computed from the camera timestamps: (N - 1) / (t_last - t_0), close to the nominal 25 Hz
    'test_gcv': {'type': 'gcv', 'n_frames': 21, 'height': 256, 'width': 256, 'fps': 20 / GCV_TLAST_S},
}
ACQUISITIONS = list(EXPECTED)

TYPE_CHECKERS = {'mp4': vr.is_this_a_mp4, 'mkv': vr.is_this_a_mkv, 'gcv': vr.is_this_a_gcv}

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=f'No test data directory at {DATA_DIR}')


@pytest.fixture(autouse=True, scope='module')
def quiet_logs():
    # missing_frames() warns for every non-gcv video: keep the test output readable
    previous_verbose = customlog.__VERBOSE__
    customlog.set_verbose('error')
    yield
    customlog.__VERBOSE__ = previous_verbose


def path_of(acquisition: str) -> str:
    return os.path.join(DATA_DIR, acquisition)


@lru_cache(maxsize=None)
def all_frames(acquisition: str) -> np.ndarray:
    return vr.get_frames(path_of(acquisition))


### DETECTION

@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_is_a_video(acquisition):
    assert vr.is_this_a_video(path_of(acquisition))


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_detected_type(acquisition):
    expected_type = EXPECTED[acquisition]['type']
    for videotype, is_this_a in TYPE_CHECKERS.items():
        assert is_this_a(path_of(acquisition)) == (videotype == expected_type), videotype


def test_not_a_video():
    assert not vr.is_this_a_video(path_of('this_video_does_not_exist'))


### METADATA

@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_number_of_frames(acquisition):
    assert vr.get_number_of_available_frames(path_of(acquisition)) == EXPECTED[acquisition]['n_frames']


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_geometry(acquisition):
    exp = EXPECTED[acquisition]
    assert vr.get_geometry(path_of(acquisition)) == (exp['n_frames'], exp['height'], exp['width'])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_geometry_subregion_and_framenumbers(acquisition):
    # subregion is (start_x, start_y, end_x, end_y)
    geometry = vr.get_geometry(path_of(acquisition), framenumbers=[2, 5, 9], subregion=(10, 20, 110, 70))
    assert geometry == (3, 50, 100)


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_frame_rate(acquisition):
    fps = vr.get_acquisition_frequency(path_of(acquisition), unit='Hz')
    assert fps == pytest.approx(EXPECTED[acquisition]['fps'])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_times(acquisition):
    n_frames = EXPECTED[acquisition]['n_frames']
    t = vr.get_t_s(path_of(acquisition))
    assert len(t) == n_frames
    assert t[0] == 0.
    assert np.all(np.diff(t) > 0)
    fps = vr.get_acquisition_frequency(path_of(acquisition), unit='Hz')
    assert np.mean(np.diff(t)) == pytest.approx(1 / fps)


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_duration(acquisition):
    # The duration is the time elapsed between the first and the last frame, i.e. (N - 1) / fps, for every format
    acquisition_path = path_of(acquisition)
    n_frames = EXPECTED[acquisition]['n_frames']
    fps = vr.get_acquisition_frequency(acquisition_path, unit='Hz')
    duration = vr.get_acquisition_duration(acquisition_path, framenumbers=None, unit='s')
    t = vr.get_t_s(acquisition_path)

    assert duration == pytest.approx((n_frames - 1) / fps)
    assert duration == pytest.approx(t[-1] - t[0])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_duration_of_selected_frames(acquisition):
    acquisition_path = path_of(acquisition)
    t = vr.get_t_s(acquisition_path)
    duration = vr.get_acquisition_duration(acquisition_path, framenumbers=np.array([2, 5, 9]), unit='s')
    assert duration == pytest.approx(t[9] - t[2])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_no_missing_frames(acquisition):
    assert vr.missing_frames(path_of(acquisition)) == []
    assert not vr.are_there_missing_frames(path_of(acquisition))


### FRAMES

@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_all_frames(acquisition):
    exp = EXPECTED[acquisition]
    frames = all_frames(acquisition)
    assert frames.shape == (exp['n_frames'], exp['height'], exp['width'])
    assert frames.dtype == np.uint8


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_first_and_last_frames(acquisition):
    acquisition_path = path_of(acquisition)
    exp = EXPECTED[acquisition]
    last = exp['n_frames'] - 1

    first_frame = vr.get_frame(acquisition_path, 0)
    last_frame = vr.get_frame(acquisition_path, last)

    for frame in (first_frame, last_frame):
        assert frame.shape == (exp['height'], exp['width'])
        assert frame.dtype == np.uint8
        assert frame.min() < frame.max()  # not a blank image

    # the reference videos are moving patterns: first and last frames differ
    assert not np.array_equal(first_frame, last_frame)

    # reading one frame gives the same as reading the whole video
    np.testing.assert_array_equal(first_frame, all_frames(acquisition)[0])
    np.testing.assert_array_equal(last_frame, all_frames(acquisition)[last])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_frames_in_any_order(acquisition):
    last = EXPECTED[acquisition]['n_frames'] - 1
    framenumbers = np.array([last, 3, 7, 8, 0])
    frames = vr.get_frames(path_of(acquisition), framenumbers)
    np.testing.assert_array_equal(frames, all_frames(acquisition)[framenumbers])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_frames_subregion(acquisition):
    subregion = (10, 20, 110, 70)  # start_x, start_y, end_x, end_y
    frames = vr.get_frames(path_of(acquisition), np.array([0, 1]), subregion=subregion)
    np.testing.assert_array_equal(frames, all_frames(acquisition)[:2, 20:70, 10:110])


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_out_of_range_framenumbers(acquisition):
    acquisition_path = path_of(acquisition)
    n_frames = EXPECTED[acquisition]['n_frames']
    assert vr.get_frames(acquisition_path, np.array([n_frames])) is None
    assert vr.get_frames(acquisition_path, np.array([-1])) is None


### ROOT PATH

def test_set_default_root_path(tmp_path):
    previous_root_path = getattr(vr, '__ROOT_PATH__')
    try:
        vr.set_default_root_path(str(tmp_path))
        assert getattr(vr, '__ROOT_PATH__') == str(tmp_path)

        # None and paths that are not directories are ignored
        vr.set_default_root_path(None)
        vr.set_default_root_path(str(tmp_path / 'not_a_directory'))
        assert getattr(vr, '__ROOT_PATH__') == str(tmp_path)
    finally:
        vr.set_default_root_path(previous_root_path)


### DATASET DESCRIPTION

def test_describe_dataset_lists_types(tmp_path, capsys):
    # A dataset with one video of each type, and an acquisition that exists both as a gcv and as a mp4
    dataset_path = tmp_path / 'dataset'
    dataset_path.mkdir()
    for name, filename in [('test_gcv', 'test_gcv.gcv'), ('test_mp4', 'test_mp4.mp4'), ('test_mkv', 'test_mkv.mkv'),
                           ('both', 'test_gcv.gcv'), ('both', 'test_mp4.mp4')]:
        extension = os.path.splitext(filename)[1]
        (dataset_path / (name + extension)).symlink_to(os.path.abspath(os.path.join(DATA_DIR, filename)))

    customlog.set_verbose('info')
    try:
        vr.describe_dataset(dataset='dataset', root_path=str(tmp_path))
    finally:
        customlog.set_verbose('error')
    output = capsys.readouterr().out

    assert "- 'test_gcv' (gcv)" in output
    assert "- 'test_mp4' (mp4)" in output
    assert "- 'test_mkv' (mkv)" in output
    assert "- 'both' (gcv, mp4)" in output
    assert output.count("- 'both'") == 1  # listed once, with both types
    assert 'TODO' not in output


### GCV-SPECIFIC

def test_gcv_meta():
    meta = vr.retrieve_meta(path_of('test_gcv'))
    assert meta['cameraRegionX'] == '128'
    assert meta['cameraRegionY'] == '128'
    assert meta['cameraRegionWidth'] == '256'
    assert meta['cameraRegionHeight'] == '256'
    assert meta['usingROI'] == 'false'
    assert meta['pixelFormat'] == '"Mono8"'
    assert float(meta['captureFrequency']) == 25.
    assert vr.get_frame_geometry_gcv(path_of('test_gcv')) == (256, 256)  # (width, height)


def test_gcv_frequency_close_to_nominal():
    fps = vr.get_acquisition_frequency(path_of('test_gcv'), unit='Hz')
    nominal_fps = float(vr.retrieve_meta(path_of('test_gcv'))['captureFrequency'])
    assert fps == pytest.approx(nominal_fps, rel=1e-2)


def test_gcv_stamps_match_rawvideo():
    acquisition_path = path_of('test_gcv')
    n_frames = EXPECTED['test_gcv']['n_frames']
    assert vr.get_number_of_available_frames_stamps(acquisition_path) == n_frames
    assert vr.get_number_of_available_frames_rawvideo(acquisition_path) == n_frames


def test_gcv_timestamps():
    t = vr.get_t_s(path_of('test_gcv'))
    assert t[1] == pytest.approx(GCV_T1_S)
    assert t[-1] == pytest.approx(GCV_TLAST_S)


def test_gcv_frames_are_the_raw_bytes():
    raw_path = os.path.join(DATA_DIR, 'test_gcv.gcv', 'rawvideo.raw')
    exp = EXPECTED['test_gcv']
    raw = np.fromfile(raw_path, dtype=np.uint8).reshape((exp['n_frames'], exp['height'], exp['width']))
    np.testing.assert_array_equal(all_frames('test_gcv'), raw)

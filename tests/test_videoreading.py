"""
Tests for g2ltk.videoreading on the reference videos in test_data/:
    - test_mp4.mp4  : lossy compressed video (H.264)
    - test_mkv.mkv  : losslessly compressed video (FFV1)
    - test_gcv.gcv/ : GevCapture video (meta + stamps + raw video)
The mp4 and mkv videos were recorded with Limrendir: captured at 25 fps and played at 30 fps, with the recording
settings (capture frame rate...) in their comment tag.
"""
import os
import shutil
from functools import lru_cache

import numpy as np
import pytest

from g2ltk import customlog
from g2ltk import videoreading as vr
from g2ltk.videoreading import reading_gcv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data')

# Camera timestamps of the second and last frames of the gcv reference video, in s (from its stamps file)
GCV_T1_S = 0.040003
GCV_TLAST_S = 0.800878

# Expected properties of the reference videos
EXPECTED = {
    # mp4 and mkv frequency is the capture frame rate measured by Limrendir (measuredCaptureFrequency in their tags),
    # not the frame rate of the video file (30 fps)
    'test_mp4': {'type': 'mp4', 'n_frames': 43, 'height': 256, 'width': 256, 'fps': 24.999524},
    'test_mkv': {'type': 'mkv', 'n_frames': 53, 'height': 256, 'width': 256, 'fps': 24.994953},
    # gcv frequency is computed from the camera timestamps: (N - 1) / (t_last - t_0), close to the nominal 25 Hz
    'test_gcv': {'type': 'gcv', 'n_frames': 21, 'height': 256, 'width': 256, 'fps': 20 / GCV_TLAST_S},
}
ACQUISITIONS = list(EXPECTED)

TYPE_CHECKERS = {'mp4': vr.is_this_a_mp4, 'mkv': vr.is_this_a_mkv, 'gcv': vr.is_this_a_gcv}

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason=f'No test data directory at {DATA_DIR}')


@pytest.fixture(autouse=True, scope='module')
def quiet_logs():
    # missing_frames() warns for every non-gcv video: keep the test output readable
    with customlog.verbosity('error'):
        yield


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


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_get_video_type(acquisition):
    assert vr.get_video_type(path_of(acquisition)) == EXPECTED[acquisition]['type']


def test_not_a_video():
    assert not vr.is_this_a_video(path_of('this_video_does_not_exist'))
    assert vr.get_video_type(path_of('this_video_does_not_exist')) is None


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

    with customlog.verbosity('info'):
        vr.describe_dataset(dataset='dataset', root_path=str(tmp_path))
    output = capsys.readouterr().out

    assert "- 'test_gcv' (gcv)" in output
    assert "- 'test_mp4' (mp4)" in output
    assert "- 'test_mkv' (mkv)" in output
    assert "- 'both' (gcv, mp4)" in output
    assert output.count("- 'both'") == 1  # listed once, with both types
    assert 'TODO' not in output


def test_missing_dataset_has_no_videos(tmp_path, capsys):
    # a dataset that is not a directory has no videos: logged as an error, without raising
    with customlog.verbosity('error'):
        assert vr.find_available_videos(dataset='missing', root_path=str(tmp_path)) == []
        assert vr.find_available_videos(dataset='missing', root_path=str(tmp_path), videotype='gcv') == []
        vr.describe_acquisition(dataset='missing', root_path=str(tmp_path))
    output = capsys.readouterr().out
    assert 'is not a directory' in output


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


### GCV CACHE (the meta and stamps files are read once, see reading_gcv)

def copy_of_test_gcv(tmp_path) -> str:
    """Acquisition path of a copy of the gcv reference video, that a test can modify."""
    shutil.copytree(os.path.join(DATA_DIR, 'test_gcv.gcv'), tmp_path / 'acq.gcv')
    return str(tmp_path / 'acq')


def test_gcv_files_are_read_once():
    vr.clear_gcv_cache()
    for framenumber in [0, 5, 20]:
        vr.get_frame(path_of('test_gcv'), framenumber)
    assert reading_gcv._parse_stamps_file.cache_info().misses == 1
    assert reading_gcv._parse_meta_file.cache_info().misses == 1


def test_gcv_cached_stamps_cannot_be_changed_by_the_caller():
    stamps = vr.retrieve_stamps(path_of('test_gcv'))
    expected = {key: values.copy() for key, values in stamps.items()}
    stamps['camera_time'][:] = 0
    stamps['framenumber'] = None
    stamps = vr.retrieve_stamps(path_of('test_gcv'))
    assert stamps.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(stamps[key], expected[key])


def test_gcv_cached_meta_cannot_be_changed_by_the_caller():
    meta = vr.retrieve_meta(path_of('test_gcv'))
    meta['captureFrequency'] = '1'
    del meta['usingROI']
    meta = vr.retrieve_meta(path_of('test_gcv'))
    assert float(meta['captureFrequency']) == 25.
    assert meta['usingROI'] == 'false'


def test_gcv_stamps_file_read_again_when_it_changes(tmp_path):
    acquisition_path = copy_of_test_gcv(tmp_path)
    n_frames = EXPECTED['test_gcv']['n_frames']
    assert len(vr.retrieve_stamps(acquisition_path)['framenumber']) == n_frames
    stamps_path = os.path.join(acquisition_path + '.gcv', 'stamps.stamps')
    with open(stamps_path) as stamps_file:
        lines = stamps_file.readlines()
    with open(stamps_path, 'w') as stamps_file:
        stamps_file.writelines(lines[:-1])
    assert len(vr.retrieve_stamps(acquisition_path)['framenumber']) == n_frames - 1


def test_gcv_meta_file_read_again_when_it_changes(tmp_path):
    acquisition_path = copy_of_test_gcv(tmp_path)
    assert float(vr.retrieve_meta(acquisition_path)['captureFrequency']) == 25.
    meta_path = os.path.join(acquisition_path + '.gcv', 'metainfo.meta')
    mtime_ns = os.stat(meta_path).st_mtime_ns
    with open(meta_path) as meta_file:
        meta = meta_file.read()
    # same size: the change is seen through the modification time only
    with open(meta_path, 'w') as meta_file:
        meta_file.write(meta.replace('captureFrequency=25.000000', 'captureFrequency=50.000000'))
    # the modification time is only updated every few ms: make sure that it changed
    os.utime(meta_path, ns=(mtime_ns + 10 ** 9, mtime_ns + 10 ** 9))
    assert float(vr.retrieve_meta(acquisition_path)['captureFrequency']) == 50.


### ACQUISITION PATHS WITH AN EXTENSION

EXTENSIONS = {'test_mp4': '.mp4', 'test_mkv': '.mkv', 'test_gcv': '.gcv'}


@pytest.mark.parametrize('acquisition_path, expected', [
    ('d/acq', ('d/acq', None)),
    ('d/acq.gcv', ('d/acq', 'gcv')),
    ('d/acq.gcv/', ('d/acq', 'gcv')),  # trailing separator, from tab completion
    ('d/acq.mp4', ('d/acq', 'mp4')),
    ('d/acq.mkv', ('d/acq', 'mkv')),
    ('d/acq.MOV', ('d/acq', 'mov')),
    ('d/acq.mov', ('d/acq', 'mov')),  # extensions are recognized case-insensitively
    ('d/flow_1.5mlmin', ('d/flow_1.5mlmin', None)),  # unknown suffix: part of the name
    ('d/flow_1.5mlmin.mp4', ('d/flow_1.5mlmin', 'mp4')),
])
def test_split_acquisition_path(acquisition_path, expected):
    assert vr.split_acquisition_path(acquisition_path) == expected


def test_video_path_for_type():
    assert vr.video_path_for_type('d/acq', 'mp4') == 'd/acq.mp4'
    assert vr.video_path_for_type('d/acq', 'mov') == 'd/acq.MOV'
    assert vr.video_path_for_type('d/acq.mp4', 'mp4') == 'd/acq.mp4'
    assert vr.video_path_for_type('d/acq.gcv/', 'gcv') == 'd/acq.gcv'
    assert vr.video_path_for_type('d/acq.mp4', 'gcv') is None  # the extension names another type


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_acquisition_path_with_extension(acquisition):
    path_with_extension = path_of(acquisition) + EXTENSIONS[acquisition]
    exp = EXPECTED[acquisition]

    assert vr.is_this_a_video(path_with_extension)
    for videotype, is_this_a in TYPE_CHECKERS.items():
        assert is_this_a(path_with_extension) == (videotype == exp['type']), videotype
    assert not vr.is_this_a_t16(path_with_extension)
    assert not vr.is_this_a_t8(path_with_extension)

    assert vr.get_number_of_available_frames(path_with_extension) == exp['n_frames']
    assert vr.get_geometry(path_with_extension) == (exp['n_frames'], exp['height'], exp['width'])
    assert vr.get_acquisition_frequency(path_with_extension, unit='Hz') == pytest.approx(exp['fps'])
    np.testing.assert_array_equal(vr.get_t_s(path_with_extension), vr.get_t_s(path_of(acquisition)))
    np.testing.assert_array_equal(vr.get_frames(path_with_extension), all_frames(acquisition))


def test_gcv_path_with_trailing_separator():
    path = path_of('test_gcv') + '.gcv' + os.sep
    assert vr.is_this_a_gcv(path)
    np.testing.assert_array_equal(vr.get_frames(path), all_frames('test_gcv'))


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_acquisition_path_with_wrong_extension(acquisition):
    for extension in EXTENSIONS.values():
        if extension != EXTENSIONS[acquisition]:
            assert not vr.is_this_a_video(path_of(acquisition) + extension)


def make_dataset(tmp_path, links):
    """Creates tmp_path/dataset with symlinks {name in the dataset: reference video in test_data}"""
    dataset_path = tmp_path / 'dataset'
    dataset_path.mkdir()
    for name, filename in links.items():
        (dataset_path / name).symlink_to(os.path.abspath(os.path.join(DATA_DIR, filename)))
    return dataset_path


def test_extension_selects_the_video_type(tmp_path):
    # The same acquisition exists as a gcv and as a mp4: the extension chooses which one is read
    dataset_path = make_dataset(tmp_path, {'both.gcv': 'test_gcv.gcv', 'both.mp4': 'test_mp4.mp4'})
    both = str(dataset_path / 'both')

    assert vr.get_number_of_available_frames(both) == EXPECTED['test_gcv']['n_frames']  # gcv comes first
    assert vr.get_number_of_available_frames(both + '.gcv') == EXPECTED['test_gcv']['n_frames']
    assert vr.get_number_of_available_frames(both + '.mp4') == EXPECTED['test_mp4']['n_frames']
    assert not vr.is_this_a_gcv(both + '.mp4')
    np.testing.assert_array_equal(vr.get_frames(both + '.mp4'), all_frames('test_mp4'))


def test_acquisition_name_with_dots(tmp_path):
    dataset_path = make_dataset(tmp_path, {'flow_1.5mlmin.mp4': 'test_mp4.mp4'})
    assert vr.find_available_videos(dataset='dataset', root_path=str(tmp_path)) == ['flow_1.5mlmin']
    for acquisition in ['flow_1.5mlmin', 'flow_1.5mlmin.mp4']:
        assert vr.is_this_a_mp4(str(dataset_path / acquisition))
        assert vr.get_number_of_available_frames(str(dataset_path / acquisition)) == EXPECTED['test_mp4']['n_frames']


def test_generate_acquisition_path_with_extension(tmp_path):
    make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    acquisition_path = vr.generate_acquisition_path('acq.mp4', dataset='dataset', root_path=str(tmp_path))
    assert acquisition_path == os.path.join(str(tmp_path), 'dataset', 'acq.mp4')
    assert vr.is_this_a_mp4(acquisition_path)


def test_save_acquisition_with_extension_to_video(tmp_path):
    dataset_path = make_dataset(tmp_path, {'acq.gcv': 'test_gcv.gcv'})
    vr.save_acquisition_to_video(str(dataset_path / 'acq.gcv'), do_timestamp=False, filetype='mkv')
    # saved as 'acq.mkv', not 'acq.gcv.mkv'
    assert os.path.isfile(dataset_path / 'acq.mkv')
    assert not os.path.exists(dataset_path / 'acq.gcv.mkv')
    assert vr.get_number_of_available_frames(str(dataset_path / 'acq.mkv')) == EXPECTED['test_gcv']['n_frames']


### DESIGNATING AN ACQUISITION: BY DATASET, OR BY PATH

@pytest.fixture
def other_root_path(tmp_path):
    """A default root path containing an unrelated dataset, restored afterwards"""
    previous_root_path = getattr(vr, '__ROOT_PATH__')
    other_root = tmp_path / 'other_root'
    (other_root / 'other_dataset').mkdir(parents=True)
    (other_root / 'other_dataset' / 'other.mp4').symlink_to(os.path.abspath(os.path.join(DATA_DIR, 'test_mp4.mp4')))
    vr.set_default_root_path(str(other_root))
    try:
        yield str(other_root)
    finally:
        vr.set_default_root_path(previous_root_path)


def test_acquisition_by_dataset_uses_root_path(tmp_path, other_root_path):
    make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    # explicit root path
    acquisition_path = vr.generate_acquisition_path('acq', dataset='dataset', root_path=str(tmp_path))
    assert acquisition_path == os.path.join(str(tmp_path), 'dataset', 'acq')
    # default root path
    acquisition_path = vr.generate_acquisition_path('other', dataset='other_dataset')
    assert acquisition_path == os.path.join(other_root_path, 'other_dataset', 'other')
    assert vr.is_this_a_mp4(acquisition_path)


@pytest.mark.parametrize('extension', ['', '.mp4'])
def test_acquisition_by_path_ignores_root_path(tmp_path, other_root_path, extension):
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    for path in [str(dataset_path / 'acq') + extension, os.path.relpath(str(dataset_path / 'acq')) + extension]:
        # the default root path (which has a dataset that could be auto-selected) does not apply
        assert vr.generate_acquisition_path(path) == path
        # nor an explicit one, without a dataset
        assert vr.generate_acquisition_path(path, root_path=other_root_path) == path
        assert vr.get_number_of_available_frames(vr.generate_acquisition_path(path)) == EXPECTED['test_mp4']['n_frames']


def test_describe_acquisition_by_path(tmp_path, other_root_path, capsys):
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with customlog.verbosity('info'):
        vr.describe_acquisition(acquisition=str(dataset_path / 'acq'))
    output = capsys.readouterr().out
    assert f'Acquisition path: "{dataset_path / "acq"}"' in output
    assert 'Video type: mp4' in output
    assert f'{EXPECTED["test_mp4"]["n_frames"]} frames' in output
    assert 'other_dataset' not in output


@pytest.mark.parametrize('relative', [False, True])
def test_describe_acquisition_by_positional_path(tmp_path, other_root_path, capsys, relative):
    # like the functions reading videos, describe_acquisition takes the path as its first argument
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    acquisition_path = str(dataset_path / 'acq')
    if relative:
        acquisition_path = os.path.relpath(acquisition_path)
    with customlog.verbosity('info'):
        vr.describe_acquisition(acquisition_path)
    output = capsys.readouterr().out
    assert f'Acquisition path: "{acquisition_path}"' in output
    assert 'Video type: mp4' in output
    assert f'{EXPECTED["test_mp4"]["n_frames"]} frames' in output
    assert 'other_dataset' not in output


def test_describe_acquisition_old_positional_call_raises(tmp_path, other_root_path):
    # describe_acquisition(dataset, acquisition) used to be valid: it now fails clearly instead of misreading them
    make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with pytest.raises(TypeError, match='no longer supported'):
        vr.describe_acquisition('dataset', 'acq')
    with pytest.raises(TypeError, match='no longer supported'):
        vr.describe_acquisition('dataset', 'acq', str(tmp_path))


def test_describe_acquisition_designated_twice_raises(tmp_path, other_root_path):
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with pytest.raises(ValueError, match='designated twice'):
        vr.describe_acquisition(str(dataset_path / 'acq'), dataset='dataset', root_path=str(tmp_path))


def test_describe_acquisition_auto_selects_in_dataset(tmp_path, other_root_path, capsys):
    # dataset given without acquisition: the acquisition is auto-selected in this dataset
    make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with customlog.verbosity('info'):
        vr.describe_acquisition(dataset='dataset', root_path=str(tmp_path))
    output = capsys.readouterr().out
    assert '"acq" (dataset)' in output
    assert f'{EXPECTED["test_mp4"]["n_frames"]} frames' in output


def test_describe_acquisition_by_dataset(tmp_path, other_root_path, capsys):
    make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with customlog.verbosity('info'):
        vr.describe_acquisition(dataset='dataset', acquisition='acq', root_path=str(tmp_path))
    output = capsys.readouterr().out
    assert 'Acquisition: "acq" (dataset)' in output
    assert 'Acquisition path' not in output
    assert 'Video type: mp4' in output
    assert f'{EXPECTED["test_mp4"]["n_frames"]} frames' in output


def test_describe_acquisition_auto_selects(other_root_path, capsys):
    # neither dataset nor acquisition given: both are auto-selected in the default root path
    with customlog.verbosity('info'):
        vr.describe_acquisition()
    output = capsys.readouterr().out
    assert '"other" (other_dataset)' in output


### READING FUNCTIONS WITH THE dataset=, acquisition= KEYWORDS

# (function, extra arguments): every reading function accepting the dataset=, acquisition=, root_path= keywords
KEYWORD_FUNCTIONS = [
    (vr.is_this_a_video, {}),
    (vr.get_video_type, {}),
    (vr.get_number_of_available_frames, {}),
    (vr.get_acquisition_frequency, {'unit': 'Hz'}),
    (vr.get_acquisition_duration, {'unit': 's'}),
    (vr.get_t_frames, {}),
    (vr.get_times, {'unit': 's'}),
    (vr.get_t_s, {}),
    (vr.get_t_ms, {}),
    (vr.missing_frames, {}),
    (vr.missing_frames_in_framenumbers, {}),
    (vr.are_there_missing_frames, {}),
    (vr.get_geometry, {'framenumbers': [1, 4], 'subregion': (10, 20, 110, 70)}),
    (vr.get_z_px, {'resize_factor': 2}),
    (vr.get_x_px, {}),
    (vr.get_x_mm, {'px_per_mm': 10.}),
    (vr.get_x_m, {'px_per_mm': 10.}),
    (vr.get_x_um, {'px_per_mm': 10.}),
    (vr.get_frames, {'framenumbers': [3, 1], 'subregion': (10, 20, 110, 70)}),
    (vr.get_frame, {'framenumber': 2}),
]


def assert_same_result(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize('function, kwargs', KEYWORD_FUNCTIONS, ids=[f.__name__ for f, _ in KEYWORD_FUNCTIONS])
def test_reading_by_dataset_or_by_path(tmp_path, other_root_path, function, kwargs):
    dataset_path = make_dataset(tmp_path, {'acq.gcv': 'test_gcv.gcv'})
    expected = function(str(dataset_path / 'acq'), **kwargs)
    assert expected is not None
    # by dataset, with an explicit root path
    assert_same_result(function(dataset='dataset', acquisition='acq', root_path=str(tmp_path), **kwargs), expected)
    # by path, given as the acquisition keyword (the default root path does not apply)
    assert_same_result(function(acquisition=str(dataset_path / 'acq'), **kwargs), expected)


def test_reading_by_dataset_with_default_root_path(other_root_path):
    frames = vr.get_frames(dataset='other_dataset', acquisition='other', framenumbers=[0])
    np.testing.assert_array_equal(frames, all_frames('test_mp4')[[0]])
    assert vr.get_geometry(dataset='other_dataset', acquisition='other') == \
        (EXPECTED['test_mp4']['n_frames'], EXPECTED['test_mp4']['height'], EXPECTED['test_mp4']['width'])


def test_reading_acquisition_designated_twice(tmp_path):
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with pytest.raises(ValueError):
        vr.get_frames(str(dataset_path / 'acq'), dataset='dataset', acquisition='acq', root_path=str(tmp_path))
    with pytest.raises(ValueError):
        vr.get_geometry(str(dataset_path / 'acq'), root_path=str(tmp_path))


def test_reading_without_acquisition():
    with pytest.raises(ValueError):
        vr.get_frames()
    with pytest.raises(ValueError):
        vr.get_t_s(dataset='dataset')
    with pytest.raises(ValueError):
        vr.is_this_a_video()


def test_is_not_a_video_by_dataset_or_by_path(tmp_path, other_root_path, capsys):
    # no video: is_this_a_video answers False, without warning (unlike generate_acquisition_path)
    dataset_path = make_dataset(tmp_path, {'acq.mp4': 'test_mp4.mp4'})
    with customlog.verbosity('warning'):
        assert not vr.is_this_a_video(dataset='dataset', acquisition='missing', root_path=str(tmp_path))
        assert not vr.is_this_a_video(dataset='missing_dataset', acquisition='acq', root_path=str(tmp_path))
        assert not vr.is_this_a_video(acquisition=str(dataset_path / 'missing'))
        assert not vr.is_this_a_video(str(dataset_path / 'missing'))
        # the default root path applies with a dataset
        assert vr.is_this_a_video(dataset='other_dataset', acquisition='other')
        assert not vr.is_this_a_video(dataset='other_dataset', acquisition='acq')
    assert capsys.readouterr().out == ''


### DATASHEETS WITH AN EXPLICIT ROOT PATH

def test_datasheets_with_root_path(tmp_path, other_root_path):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('openpyxl')
    dataset_path = make_dataset(tmp_path, {'acq1.mp4': 'test_mp4.mp4', 'acq2.mp4': 'test_mp4.mp4'})
    # the metainfo sheet starts with two lines of header (read with skiprows=2)
    with pd.ExcelWriter(dataset_path / 'dataworkbook.xlsx', engine='openpyxl') as writer:
        pd.DataFrame([['header'], ['header'], ['acquisition'], ['acq1'], ['acq2']]).to_excel(
            writer, sheet_name='metainfo', index=False, header=False)

    root_path = str(tmp_path)
    assert vr.generate_dataworkbook_path('dataset', root_path=root_path) == str(dataset_path / 'dataworkbook.xlsx')
    assert list(vr.obtain_acquisition_titles('dataset', root_path=root_path)) == ['acq1', 'acq2']

    worksheet = vr.load_or_create_worksheet('dataset', 'results', columns=['height'], root_path=root_path)
    assert list(worksheet['acquisition']) == ['acq1', 'acq2']
    # adding a column rewrites the worksheet, in the dataset of the given root path
    worksheet = vr.load_or_create_worksheet('dataset', 'results', columns=['height', 'width'], root_path=root_path)
    assert 'width' in worksheet.keys()
    assert 'width' in pd.read_excel(dataset_path / 'dataworkbook.xlsx', sheet_name='results').keys()


### VIDEOS RECORDED WITH LIMRENDIR (mkv, mp4): RECORDING SETTINGS IN THEIR TAGS

from g2ltk.videoreading import reading_videotags

LIMRENDIR_SETTINGS = {
    'test_mp4': {'captureFrequency': '25.000000', 'playbackFrequency': '30', 'measuredCaptureFrequency': '24.999524',
                 'recordedFrames': '43', 'lostFrames': '0', 'captureProg': 'Limrendir version 0.8.2'},
    'test_mkv': {'captureFrequency': '25.000000', 'playbackFrequency': '30', 'measuredCaptureFrequency': '24.994953',
                 'recordedFrames': '53', 'lostFrames': '0', 'captureProg': 'Limrendir version 0.8.2'},
}


@pytest.mark.parametrize('acquisition', list(LIMRENDIR_SETTINGS))
def test_limrendir_recording_settings(acquisition):
    settings = vr.get_recording_settings(path_of(acquisition))
    for key, value in LIMRENDIR_SETTINGS[acquisition].items():
        assert settings[key] == value, key
    # the number of frames of the video is the one recorded
    assert vr.get_number_of_available_frames(path_of(acquisition)) == int(settings['recordedFrames'])


@pytest.mark.parametrize('acquisition', list(LIMRENDIR_SETTINGS))
def test_limrendir_times_use_capture_frequency(acquisition):
    # the video files are played at 30 fps, but the times are those of the capture
    fps = float(LIMRENDIR_SETTINGS[acquisition]['measuredCaptureFrequency'])
    t = vr.get_t_s(path_of(acquisition))
    np.testing.assert_allclose(t, np.arange(EXPECTED[acquisition]['n_frames']) / fps)
    t = vr.get_t_ms(dataset='.', acquisition=acquisition, root_path=DATA_DIR)
    np.testing.assert_allclose(t, np.arange(EXPECTED[acquisition]['n_frames']) / fps * 1e3)


def test_limrendir_settings_by_extension(tmp_path):
    # the settings are those of the video read: with an explicit extension, the one of that type
    dataset_path = make_dataset(tmp_path, {'both.mkv': 'test_mkv.mkv', 'both.mp4': 'test_mp4.mp4'})
    assert vr.get_recording_settings(str(dataset_path / 'both.mp4'))['recordedFrames'] == '43'
    assert vr.get_recording_settings(str(dataset_path / 'both.mkv'))['recordedFrames'] == '53'
    assert vr.get_recording_settings(str(dataset_path / 'both'))['recordedFrames'] == '53'  # mkv is read first


def test_video_without_recording_settings(tmp_path):
    # a mkv video written by g2ltk (from the gcv video) has no tags: its frame rate is the one of the file
    dataset_path = make_dataset(tmp_path, {'acq.gcv': 'test_gcv.gcv'})
    vr.save_acquisition_to_video(str(dataset_path / 'acq.gcv'), do_timestamp=False, filetype='mkv', fps=12.)
    acquisition_path = str(dataset_path / 'acq.mkv')
    assert vr.get_recording_settings(acquisition_path) is None
    assert vr.get_acquisition_frequency(acquisition_path, unit='Hz') == pytest.approx(12.)
    assert vr.get_recording_settings(path_of('test_gcv')) is None


def test_unreadable_tags(tmp_path):
    # files that are not valid mkv / mp4 have no settings, without error
    for name, content in [('garbage.mkv', b'\x1a\x45\xdf\xa3' + bytes(range(256)) * 3), ('garbage.mp4', b'\x00\x00\x00\x10moov' + b'\xff' * 50),
                          ('empty.mkv', b''), ('empty.mp4', b'')]:
        (tmp_path / name).write_bytes(content)
        assert reading_videotags.get_recording_settings_of_file(str(tmp_path / name)) is None
    assert reading_videotags.get_recording_settings_of_file(str(tmp_path / 'does_not_exist.mkv')) is None


def test_parse_recording_settings():
    tags = ['x264',
            'Recorded at 156.00 fps, played at 30 fps (5.20x slower)\ncaptureFrequency=156.000000\nslowdownFactor=5.2',
            'Recorded at 156.00 fps\ncaptureFrequency=156.000000\nlostFrames=2\nmeasuredCaptureFrequency=155.5\n',
            'title=not settings']
    settings = reading_videotags.parse_recording_settings(tags)
    # the later tags (written at the end of the recording) complete the earlier ones
    assert settings == {'captureFrequency': '156.000000', 'slowdownFactor': '5.2', 'lostFrames': '2',
                        'measuredCaptureFrequency': '155.5'}
    assert reading_videotags.parse_recording_settings(['x264', 'title=a=b']) is None
    assert reading_videotags.parse_recording_settings([]) is None


def test_recorded_capture_frequency():
    frequency = reading_videotags.recorded_capture_frequency
    assert frequency({'captureFrequency': '156.0', 'measuredCaptureFrequency': '155.5'}) == 155.5
    # the measured frequency is only there when the recording was ended properly
    assert frequency({'captureFrequency': '156.0'}) == 156.
    assert frequency({'captureFrequency': '156.0', 'measuredCaptureFrequency': 'nan?'}) == 156.
    assert frequency({'captureFrequency': '0'}) is None
    assert frequency({}) is None
    assert frequency(None) is None


def test_unescape_gstreamer_string():
    unescape = reading_videotags._unescape_gstreamer_string
    assert unescape(b'"Recorded\\ at\\ 25.00\\ fps\\,\\ x\\012captureFrequency\\=25.000000\\012"') == \
        b'Recorded at 25.00 fps, x\ncaptureFrequency=25.000000\n'
    assert unescape('"caf\\303\\251\\ \\"q\\"\\\\"'.encode()).decode() == 'caf\u00e9 "q"\\'
    # not serialized: as is
    assert unescape(b'captureFrequency=25') == b'captureFrequency=25'


### MISSING FRAMES OF VIDEOS WITHOUT FRAME NUMBERS

@pytest.mark.parametrize('acquisition', ['test_mp4', 'test_mkv'])
def test_no_missing_frames_warning(acquisition, capsys):
    with customlog.verbosity('warning'):
        vr.describe_acquisition(acquisition=path_of(acquisition))
        assert not vr.are_there_missing_frames(path_of(acquisition))
    assert 'WARN' not in capsys.readouterr().out


def test_lost_frames_warned_once(tmp_path, capsys):
    # a Limrendir recording in which frames were lost: the frames cannot be located, this is warned once
    content = open(os.path.join(DATA_DIR, 'test_mp4.mp4'), 'rb').read()
    assert content.count(b'lostFrames=0') == 1
    (tmp_path / 'lost.mp4').write_bytes(content.replace(b'lostFrames=0', b'lostFrames=3'))
    acquisition_path = str(tmp_path / 'lost')
    with customlog.verbosity('warning'):
        assert vr.missing_frames(acquisition_path) == []
        vr.describe_acquisition(acquisition=acquisition_path)
    output = capsys.readouterr().out
    assert output.count('3 frames were lost') == 1


@pytest.mark.parametrize('acquisition', ACQUISITIONS)
def test_time_units(acquisition):
    acquisition_path = path_of(acquisition)
    t_s = vr.get_t_s(acquisition_path)
    np.testing.assert_allclose(vr.get_t_ms(acquisition_path), t_s * 1e3)
    np.testing.assert_allclose(vr.get_times(acquisition_path, unit='us'), t_s * 1e6)
    duration_s = vr.get_acquisition_duration(acquisition_path, unit='s')
    assert vr.get_acquisition_duration(acquisition_path, unit='ms') == pytest.approx(duration_s * 1e3)

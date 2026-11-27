"""
Reading the tags (metadata) of mkv (Matroska) and mp4 files, and the recording settings that Limrendir writes in them.

Limrendir records mkv and mp4 videos that are played at a fixed frame rate (30 fps), whatever the capture frame rate:
the frame rate of the video file is not the one of the acquisition. The recording settings are written in the comment
tag of the file, as "key=value" lines, with the same keys as in the metainfo.meta file of gcv videos::

    captureFrequency=156.000000           frame rate of the camera when the recording started (Hz)
    playbackFrequency=30                  frame rate of the video file (Hz)
    measuredCaptureFrequency=155.980000   frame rate measured from the camera timestamps (Hz), when the recording ends
    recordedFrames=1234                   number of frames in the video, when the recording ends
    lostFrames=0                          number of frames lost during the recording, when the recording ends
    ...

The tags are read with the minimal parsers below, which need no other library than the standard one.
"""
import os
import struct
from typing import Optional, Dict, List, Tuple, Iterator, BinaryIO

from .. import customlog

# The names g2ltk.videoreading gets from `from .reading_videotags import *`: add new public functions here
__all__ = ['parse_recording_settings', 'recorded_capture_frequency', 'recorded_lost_frames',
           'get_recording_settings_of_file', 'read_mkv_tag_strings', 'read_mp4_tag_strings']

### SETTINGS WRITTEN BY LIMRENDIR

def parse_recording_settings(tag_strings: List[str]) -> Optional[Dict[str, str]]:
    """
    Recording settings found in the tags of a video ("key=value" lines of the tags that have a captureFrequency).

    When several tags have settings (e.g. the ones written when the recording started and when it ended), the later
    ones take precedence.

    :param tag_strings: the text of the tags, in the order of the file
    :return: the settings, or None if no tag has any
    """
    settings = None
    for text in tag_strings:
        lines = [line.strip() for line in text.splitlines()]
        if not any(line.startswith('captureFrequency=') for line in lines):
            continue
        if settings is None:
            settings = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                settings[key.strip()] = value.strip()
    return settings

def recorded_capture_frequency(settings: Optional[Dict[str, str]]) -> Optional[float]:
    """
    Frame rate (Hz) at which the video was captured, according to its recording settings.

    The frame rate measured from the camera timestamps is preferred (like the frame rate of gcv videos, which is
    computed from their timestamps), else the frame rate of the camera when the recording started.

    :param settings: recording settings (see parse_recording_settings)
    :return: the frame rate in Hz, or None if the settings do not give it
    """
    if settings is None:
        return None
    for key in ['measuredCaptureFrequency', 'captureFrequency']:
        try:
            frequency = float(settings[key])
        except (KeyError, ValueError):
            continue
        if frequency > 0:
            return frequency
    return None

def recorded_lost_frames(settings: Optional[Dict[str, str]]) -> Optional[int]:
    """
    Number of frames lost during the recording, according to its recording settings.

    :param settings: recording settings (see parse_recording_settings)
    :return: the number of lost frames, or None if the settings do not give it
    """
    if settings is None:
        return None
    try:
        return int(settings['lostFrames'])
    except (KeyError, ValueError):
        return None

# Cache of the recording settings of each file: (path, modification time, size) -> settings
_recording_settings_cache: Dict[Tuple[str, int, int], Optional[Dict[str, str]]] = {}

def get_recording_settings_of_file(video_path: str) -> Optional[Dict[str, str]]:
    """
    Recording settings written by Limrendir in the tags of a mkv or mp4 file.

    :param video_path: path of the mkv or mp4 file
    :return: the settings, or None if there are none (or if the file cannot be read)
    """
    try:
        stat = os.stat(video_path)
    except OSError:
        return None
    key = (os.path.abspath(video_path), stat.st_mtime_ns, stat.st_size)
    if key not in _recording_settings_cache:
        try:
            if video_path.lower().endswith('.mkv'):
                tag_strings = read_mkv_tag_strings(video_path)
            else:
                tag_strings = read_mp4_tag_strings(video_path)
        except (OSError, ValueError, struct.error, IndexError) as e:
            customlog.log_debug(f'Could not read the tags of {video_path}: {e}')
            tag_strings = []
        settings = parse_recording_settings(tag_strings)
        if settings is not None:
            customlog.log_trace(f'Recording settings of {video_path}: {settings}')
        _recording_settings_cache[key] = settings
    return _recording_settings_cache[key]

### MATROSKA (mkv) TAGS

# EBML element IDs
_EBML = 0x1A45DFA3
_SEGMENT = 0x18538067
_SEEKHEAD = 0x114D9B74
_SEEK = 0x4DBB
_SEEKID = 0x53AB
_SEEKPOSITION = 0x53AC
_TAGS = 0x1254C367
_TAG = 0x7373
_SIMPLETAG = 0x67C8
_TAGSTRING = 0x4487

def _read_ebml_vint(f: BinaryIO, max_length: int, keep_marker: bool) -> Tuple[Optional[int], bool]:
    """Reads an EBML variable-size integer. Returns (value, is all ones) or (None, False) at the end of the file."""
    first = f.read(1)
    if not first:
        return None, False
    length = 1
    mask = 0x80
    while not first[0] & mask:
        mask >>= 1
        length += 1
        if length > max_length:
            raise ValueError('Invalid EBML variable-size integer')
    rest = f.read(length - 1)
    if len(rest) < length - 1:
        return None, False
    value = first[0] if keep_marker else first[0] & (mask - 1)
    all_ones = (first[0] & (mask - 1)) == mask - 1
    for byte in rest:
        value = (value << 8) | byte
        all_ones = all_ones and byte == 0xFF
    return value, all_ones

def _read_ebml_header(f: BinaryIO) -> Tuple[Optional[int], Optional[int]]:
    """Reads the ID and the size of an EBML element (size None: unknown). Returns (None, None) at the end of the file."""
    element_id, _ = _read_ebml_vint(f, 4, keep_marker=True)
    if element_id is None:
        return None, None
    size, unknown = _read_ebml_vint(f, 8, keep_marker=False)
    if size is None:
        return None, None
    return element_id, (None if unknown else size)

def _ebml_children(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """The (ID, content) of the elements in the content of a master element."""
    import io
    f = io.BytesIO(data)
    while f.tell() < len(data):
        element_id, size = _read_ebml_header(f)
        if element_id is None:
            return
        if size is None:
            size = len(data) - f.tell()
        yield element_id, f.read(size)

def _unescape_gstreamer_string(text: bytes) -> bytes:
    """
    The string serialized by GStreamer (as matroskamux writes the tags it does not know, like the comment of Limrendir):
    "Recorded\\ at\\ 25.00\\ fps\\012captureFrequency\\=25.000000" -> Recorded at 25.00 fps<newline>captureFrequency=25.000000
    Characters are escaped with a backslash, and control or non-ASCII bytes with their 3-digit octal code.
    Strings that are not quoted are returned as is.
    """
    if len(text) < 2 or text[:1] != b'"' or text[-1:] != b'"':
        return text
    text = text[1:-1]
    unescaped = bytearray()
    i = 0
    while i < len(text):
        if text[i:i + 1] == b'\\' and i + 1 < len(text):
            octal = text[i + 1:i + 4]
            if len(octal) == 3 and all(48 <= c <= 55 for c in octal):  # '0' to '7'
                unescaped.append(int(octal, 8) & 0xFF)
                i += 4
            else:
                unescaped.append(text[i + 1])
                i += 2
        else:
            unescaped.append(text[i])
            i += 1
    return bytes(unescaped)

def _simple_tag_strings(simple_tag: bytes) -> Iterator[str]:
    for element_id, content in _ebml_children(simple_tag):
        if element_id == _TAGSTRING:
            content = _unescape_gstreamer_string(content.rstrip(b'\x00'))
            yield content.decode('utf-8', errors='replace')
        elif element_id == _SIMPLETAG:  # nested tag
            yield from _simple_tag_strings(content)

def _tags_strings(tags: bytes) -> Iterator[str]:
    for element_id, tag in _ebml_children(tags):
        if element_id == _TAG:
            for child_id, simple_tag in _ebml_children(tag):
                if child_id == _SIMPLETAG:
                    yield from _simple_tag_strings(simple_tag)

def read_mkv_tag_strings(video_path: str) -> List[str]:
    """
    Text of the tags of a Matroska (mkv) file, in the order of the file.

    The top-level elements of the segment are skipped through without reading them (the frames are not read).
    If one has an unknown size (file not finalized), the Tags elements listed in the SeekHead are read instead.

    :param video_path:
    :return:
    """
    strings: List[str] = []
    with open(video_path, 'rb') as f:
        file_size = os.fstat(f.fileno()).st_size
        element_id, size = _read_ebml_header(f)
        if element_id != _EBML or size is None:
            return []
        f.seek(size, os.SEEK_CUR)
        element_id, segment_size = _read_ebml_header(f)
        if element_id != _SEGMENT:
            return []
        segment_start = f.tell()
        segment_end = file_size if segment_size is None else min(file_size, segment_start + segment_size)

        tags_read = set()
        tags_listed = []
        position = segment_start
        while position < segment_end:
            f.seek(position)
            element_id, size = _read_ebml_header(f)
            if element_id is None:
                break
            if element_id == _TAGS and size is not None:
                strings += _tags_strings(f.read(size))
                tags_read.add(position)
            elif element_id == _SEEKHEAD and size is not None:
                for seek_id, seek in _ebml_children(f.read(size)):
                    if seek_id != _SEEK:
                        continue
                    fields = dict(_ebml_children(seek))
                    if int.from_bytes(fields.get(_SEEKID, b''), 'big') == _TAGS and _SEEKPOSITION in fields:
                        tags_listed.append(segment_start + int.from_bytes(fields[_SEEKPOSITION], 'big'))
            if size is None:
                break  # cannot skip an element of unknown size
            position = f.tell() if element_id in (_TAGS, _SEEKHEAD) else f.tell() + size

        for position in tags_listed:
            if position in tags_read or position >= file_size:
                continue
            f.seek(position)
            element_id, size = _read_ebml_header(f)
            if element_id == _TAGS and size is not None:
                strings += _tags_strings(f.read(size))
    return strings

### MP4 TAGS

def _mp4_atoms(f: BinaryIO, start: int, end: int) -> Iterator[Tuple[bytes, int, int]]:
    """The (type, content start, content end) of the atoms between start and end, without reading their content."""
    position = start
    while position + 8 <= end:
        f.seek(position)
        size, kind = struct.unpack('>I4s', f.read(8))
        header_size = 8
        if size == 1:  # 64-bit size
            size = struct.unpack('>Q', f.read(8))[0]
            header_size = 16
        elif size == 0:  # until the end
            size = end - position
        if size < header_size or position + size > end:
            return
        yield kind, position + header_size, position + size
        position += size

def _mp4_tag_strings(f: BinaryIO, start: int, end: int, parent: bytes) -> Iterator[str]:
    for kind, content_start, content_end in _mp4_atoms(f, start, end):
        if kind in (b'moov', b'trak', b'udta', b'ilst'):
            yield from _mp4_tag_strings(f, content_start, content_end, kind)
        elif kind == b'meta':
            # ISO meta atoms start with a version and flags, QuickTime ones directly with their atoms (hdlr)
            f.seek(content_start + 4)
            has_version = f.read(4) != b'hdlr'
            yield from _mp4_tag_strings(f, content_start + (4 if has_version else 0), content_end, kind)
        elif parent == b'ilst':
            # iTunes-style item (e.g. '\xa9cmt'): the value is in a data atom (type, locale, value)
            for child, data_start, data_end in _mp4_atoms(f, content_start, content_end):
                if child == b'data' and data_end - data_start >= 8:
                    f.seek(data_start)
                    data_type = struct.unpack('>I', f.read(4))[0] & 0xFFFFFF
                    f.read(4)
                    if data_type == 1:  # UTF-8 text
                        yield f.read(data_end - data_start - 8).decode('utf-8', errors='replace')
        elif parent == b'udta' and kind.startswith(b'\xa9') and content_end - content_start >= 4:
            # QuickTime-style text atom: text size, language, text
            f.seek(content_start)
            text_size = struct.unpack('>H', f.read(2))[0]
            f.read(2)
            yield f.read(min(text_size, content_end - content_start - 4)).decode('utf-8', errors='replace')

def read_mp4_tag_strings(video_path: str) -> List[str]:
    """
    Text of the tags of a mp4 file, in the order of the file.

    Only the atoms holding tags (moov/udta...) are read, not the frames nor the index tables.

    :param video_path:
    :return:
    """
    with open(video_path, 'rb') as f:
        file_size = os.fstat(f.fileno()).st_size
        return list(_mp4_tag_strings(f, 0, file_size, b''))

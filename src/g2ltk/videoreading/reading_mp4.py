from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from . import video_path_for_type
from .reading_videotags import get_recording_settings_of_file, recorded_capture_frequency

from .. import customlog

###### LOSSY COMPRESSED VIDEO (mp4) READING

def find_available_mp4(dataset_path: str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.mp4')]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_mp4(acquisition_path: str) -> bool:
    """
    Checks if there is a lossy compressed video (mp4, codec H.264) for the given acquisition path.

    :param acquisition_path:
    :return:
    """
    video:Optional[Any] = capture_mp4(acquisition_path)
    if video is None:
        return False
    else:
        video.release()
        return True

def capture_mp4(acquisition_path:str) -> Optional[Any]:
    # This only captures LOSSY COMPRESSED VIDEOS WITH CODEC H264 AND FILETYPE MP4

    # Check for existence
    video_path = video_path_for_type(acquisition_path, 'mp4')
    if video_path is None:
        return None  # the acquisition path has the extension of another video type

    if not os.path.isfile(video_path):
        customlog.log_debug(f'No mov video named {video_path}')
        return None

    # Open video
    video = cv2.VideoCapture(video_path)

    if video.isOpened()== False:
        customlog.log_error(f'Error opening video file {video_path}')
        return None

    #Check codec
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)

    # Depending on the OpenCV / FFmpeg version, H.264 is reported as 'avc1' (container tag) or 'h264' (codec name)
    if codec.lower() not in ('avc1', 'h264'):
        video.release()
        customlog.log_error(f"Codec is {codec}. Cannot read frames from video that is not lossy compressed with codec H.264 (avc1)")
        return None

    return video

def get_number_of_available_frames_mp4(acquisition_path: str) -> Optional[int]:
    video = capture_mp4(acquisition_path)
    n_framenumbers_tot:int = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return n_framenumbers_tot


def get_recording_settings_mp4(acquisition_path: str) -> Optional[Dict[str, str]]:
    """
    Recording settings written by Limrendir in the tags of the mp4 video (see reading_videotags), or None.

    :param acquisition_path:
    :return:
    """
    video_path = video_path_for_type(acquisition_path, 'mp4')
    if video_path is None:
        return None
    return get_recording_settings_of_file(video_path)

def get_acquisition_frequency_mp4(acquisition_path, unit='Hz'):
    # Videos recorded with Limrendir are played at a fixed frame rate, which is not the one of the acquisition:
    # the capture frame rate is in their tags
    capture_frequency = recorded_capture_frequency(get_recording_settings_mp4(acquisition_path))
    if capture_frequency is not None:
        return capture_frequency
    video = capture_mp4(acquisition_path)
    if video is not None:
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    return None
def get_frame_geometry_mp4(acquisition_path):
    video = capture_mp4(acquisition_path)
    if video is not None:
        height:int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width:int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video.release()
        return height, width
    return None

def get_frames_mp4(acquisition_path:str, framenumbers:np.ndarray) -> Optional[np.ndarray]:

    # Capture the video
    mp4video = capture_mp4(acquisition_path)

    height:int = int(mp4video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width:int = int(mp4video.get(cv2.CAP_PROP_FRAME_WIDTH))
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint8)

    # A video that was just opened reads frame 0 next. Seeking is slow (it decodes from an earlier keyframe), and is
    # done in the loop only when needed, so that the first frame is not seeked twice
    current_framenumber:int = -1
    for i_frame, framenumber in enumerate(framenumbers):
        # Since the mp4video is compressed, the reading without moving is optimized and faster
        # so for speed purposes we avoid moving if we don't have to
        if framenumber != current_framenumber + 1:
            mp4video.set(cv2.CAP_PROP_POS_FRAMES, framenumber)

        current_framenumber = framenumber

        ret, frame = mp4video.read()
        if ret == False:
            #Todo:ERROR here
            print('Error opening frame')
        else:
            frames[i_frame] = frame[:,:,0]

    mp4video.release()

    return frames

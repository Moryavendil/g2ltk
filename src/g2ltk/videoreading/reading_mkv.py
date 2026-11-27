from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from . import video_path_for_type
from .reading_videotags import get_recording_settings_of_file, recorded_capture_frequency

# The names g2ltk.videoreading gets from `from .reading_mkv import *`: add new public functions here
__all__ = ['find_available_mkv', 'is_this_a_mkv', 'capture_mkv',
           'get_number_of_available_frames_mkv', 'get_recording_settings_mkv',
           'get_acquisition_frequency_mkv', 'get_frames_mkv']

###### LOSSLESSLY COMPRESSED VIDEO (mkv, codec FFV1) READING

def find_available_mkv(dataset_path: str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.mkv')]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_mkv(acquisition_path: str) -> bool:
    """
    Checks if there is a losslessly compressed video (mkv, codec FFV1) for the given acquisition path.

    Parameters
    ----------
    acquisition_path : str

    Returns
    -------
    bool
    """
    video:Optional[Any] = capture_mkv(acquisition_path)
    if video is None:
        return False
    else:
        video.release()
        return True

def capture_mkv(acquisition_path:str) -> Optional[Any]:
    # This only captures LOSSLESSY COMPRESSED VIDEOS WITH CODEC FFV1 AND FILETYPE MKV

    # Check for existence
    video_path = video_path_for_type(acquisition_path, 'mkv')
    if video_path is None:
        return None  # the acquisition path has the extension of another video type

    if not os.path.isfile(video_path):
        #Todo: ERROR here
        return None

    # Open video
    video = cv2.VideoCapture(video_path)

    if video.isOpened()== False:
        #Todo:ERROR here
        print("Error opening video file")
        return None

    #Check codec
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)

    # Depending on the OpenCV / FFmpeg version, the codec is reported as 'FFV1' or 'ffv1'
    if codec.lower() != 'ffv1':
        video.release()
        #Todo:ERROR here
        print("Cannot read frames from video that is not losslessly compressed with codec FFV1")
        return None

    return video

def get_number_of_available_frames_mkv(acquisition_path: str) -> Optional[int]:
    video = capture_mkv(acquisition_path)
    n_framenumbers_tot:int = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return n_framenumbers_tot

def get_recording_settings_mkv(acquisition_path: str) -> Optional[Dict[str, str]]:
    """
    Recording settings written by Limrendir in the tags of the mkv video (see reading_videotags), or None.

    Parameters
    ----------
    acquisition_path : str

    Returns
    -------
    dict or None
    """
    video_path = video_path_for_type(acquisition_path, 'mkv')
    if video_path is None:
        return None
    return get_recording_settings_of_file(video_path)

def get_acquisition_frequency_mkv(acquisition_path: str, unit='Hz') -> Optional[float]:
    # Videos recorded with Limrendir are played at a fixed frame rate, which is not the one of the acquisition:
    # the capture frame rate is in their tags
    capture_frequency = recorded_capture_frequency(get_recording_settings_mkv(acquisition_path))
    if capture_frequency is not None:
        return capture_frequency
    video = capture_mkv(acquisition_path)
    if video is not None:
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    return None

def get_frames_mkv(acquisition_path:str, framenumbers:np.ndarray) -> Optional[np.ndarray]:

    # Capture the video
    video = capture_mkv(acquisition_path)

    height:int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width:int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint8)

    # A video that was just opened reads frame 0 next. Seeking is slow (it decodes from an earlier keyframe), and is
    # done in the loop only when needed, so that the first frame is not seeked twice
    current_framenumber:int = -1
    for i_frame, framenumber in enumerate(framenumbers):
        # Since the video is compressed, the reading without moving is optimized and faster
        # so for speed purposes we avoid moving if we don't have to
        if framenumber != current_framenumber + 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, framenumber)

        current_framenumber = framenumber

        ret, frame = video.read()
        if ret == False:
            #Todo:ERROR here
            print('Error opening frame')
        else:
            frames[i_frame] = frame[:,:,0]

    video.release()

    return frames


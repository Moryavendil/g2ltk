from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace, log_subtrace
from .. import utility, datasaving

###### LOSSY COMPRESSED VIDEO (mp4) READING

def find_available_mov(dataset_path: str) ->List[str]:

    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.MOV')]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_mov(acquisition_path: str) -> bool:
    log_subtrace(f'func:is_this_a_mov')
    video:Optional[Any] = capture_mov(acquisition_path)
    if video is None:
        return False
    else:
        video.release()
        return True

def capture_mov(acquisition_path:str) -> Optional[Any]:
    log_subtrace(f'func:capture_mov')
    # This only captures LOSSY COMPRESSED VIDEOS WITH CODEC H264 AND FILETYPE MP4

    # Check for existence
    video_path = acquisition_path + '.MOV'

    if not os.path.isfile(video_path):
        log_trace(f'No mov video named {video_path}')
        return None

    # Open video
    video = cv2.VideoCapture(video_path)

    if video.isOpened()== False:
        log_error(f'Error opening video file {video_path}')
        return None

    #Check codec
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)

    # if codec != 'avc1':
    #     video.release()
    #     #Todo:ERROR here
    #     log_error(f"Codec is {codec}. Cannot read frames from video that is not lossy compressed with codec avc1")
    #     return None

    return video

def get_number_of_available_frames_mov(acquisition_path: str) -> Optional[int]:
    video = capture_mov(acquisition_path)
    if video is not None:
        n_framenumbers_tot:int = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return n_framenumbers_tot
    return None

def get_acquisition_frequency_mov(acquisition_path, unit='Hz', verbose=1):
    video = capture_mov(acquisition_path)
    if video is not None:
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    return None

def get_frame_geometry_mov(acquisition_path):
    video = capture_mov(acquisition_path)
    if video is not None:
        height:int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width:int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video.release()
        return height, width
    return None

def get_frames_mov(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

    # Capture the video
    video = capture_mov(acquisition_path)

    height:int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width:int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint8)

    current_framenumber:int = framenumbers[0]
    video.set(cv2.CAP_PROP_POS_FRAMES, current_framenumber)
    for i_frame, framenumber in enumerate(framenumbers):
        # Since the mp4video is compressed, the reading without moving is optimized and faster
        # so for speed purposes we avoid moving if we don't have to
        if framenumber != current_framenumber + 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, framenumber)

        current_framenumber = framenumber

        ret, frame = video.read()
        if ret == False:
            log_error(f'Error opening frame {framenumber} of video at {acquisition_path}')
        else:
            frames[i_frame] = frame[:,:,0]

    video.release()

    return frames
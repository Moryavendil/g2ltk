from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import logging

###### LOSSY COMPRESSED VIDEO (mp4) READING

def find_available_mp4(dataset_path: str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.mp4')]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_mp4(acquisition_path: str) -> bool:
    """
    Checks if there is a Losslessly Compressed Video (lcv) for the given acquisition path.

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
    video_path = acquisition_path + '.mp4'

    if not os.path.isfile(video_path):
        logging.log_debug(f'No mov video named {video_path}')
        return None

    # Open video
    video = cv2.VideoCapture(video_path)

    if video.isOpened()== False:
        logging.log_error(f'Error opening video file {video_path}')
        return None

    #Check codec
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)

    if codec != 'avc1':
        video.release()
        logging.log_error(f"Codec is {codec}. Cannot read frames from video that is not lossy compressed with codec avc1")
        return None

    return video

def get_number_of_available_frames_mp4(acquisition_path: str) -> Optional[int]:
    lcv = capture_mp4(acquisition_path)
    n_framenumbers_tot:int = int(lcv.get(cv2.CAP_PROP_FRAME_COUNT))
    lcv.release()
    return n_framenumbers_tot


def get_acquisition_frequency_mp4(acquisition_path, unit='Hz'):
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

    current_framenumber:int = framenumbers[0]
    mp4video.set(cv2.CAP_PROP_POS_FRAMES, current_framenumber)
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

def get_chronos_acquisition_frequency_from_mp4_video(**parameters) -> Optional[float]:
    t = get_chronos_timestamps_from_mp4_video(**parameters)
    if t is None:
        return None
    else:
        if len(t) < 2:
            return None
        else:
            return float(1/np.mean(t[1:]-t[:-1]))


from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace
from .. import utility, datasaving

###### LOSSLESSLY COMPRESSED VIDEO (lcv) READING

def find_available_lcv(dataset_path: str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.mkv')]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_lcv(acquisition_path: str) -> bool:
    """
    Checks if there is a Losslessly Compressed Video (lcv) for the given dataset.

    :param acquisition_path:
    :return:
    """
    video:Optional[Any] = capture_lcv(acquisition_path)
    if video is None:
        return False
    else:
        video.release()
        return True

def capture_lcv(acquisition_path:str) -> Optional[Any]:
    # This only captures LOSSLESSY COMPRESSED VIDEOS WITH CODEC FFV1 AND FILETYPE MKV

    # Check for existence
    video_path = acquisition_path + ".mkv"

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

    if codec != 'FFV1':
        video.release()
        #Todo:ERROR here
        print("Cannot read frames from video that is not losslessly compressed with codec FFV1")
        return None

    return video

def get_number_of_available_frames_lcv(acquisition_path: str) -> Optional[int]:
    lcv = capture_lcv(acquisition_path)
    n_framenumbers_tot:int = int(lcv.get(cv2.CAP_PROP_FRAME_COUNT))
    lcv.release()
    return n_framenumbers_tot

def get_frames_lcv(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

    # Capture the video
    lcv = capture_lcv(acquisition_path)

    height:int = int(lcv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width:int = int(lcv.get(cv2.CAP_PROP_FRAME_WIDTH))
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint8)

    current_framenumber:int = framenumbers[0]
    lcv.set(cv2.CAP_PROP_POS_FRAMES, current_framenumber)
    for i_frame, framenumber in enumerate(framenumbers):
        # Since the lcv is compressed, the reading without moving is optimized and faster
        # so for speed purposes we avoid moving if we don't have to
        if framenumber != current_framenumber + 1:
            lcv.set(cv2.CAP_PROP_POS_FRAMES, framenumber)

        current_framenumber = framenumber

        ret, frame = lcv.read()
        if ret == False:
            #Todo:ERROR here
            print('Error opening frame')
        else:
            frames[i_frame] = frame[:,:,0]

    lcv.release()

    return frames


from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace
from .. import utility, datasaving

from . import are_there_missing_frames, resize_frames, find_available_videos, get_frames, get_times
from . import is_this_a_gcv, is_this_a_t16

# save videos for easyvisualisation

def save_frames_to_video(video_rawpath:str, frames:np.ndarray, fps:float = 25., filetype:str = 'mkv', codec:Optional[str] = None, resize_factor:int = 1):
    # codec ok : DIVX, XVID (trow error with mp4)
    # couple ok : codec mp4v, filetype mp4
    # couple ok : codec H264, filetype avi
    # couple ok : codec FFV1, filetype mkv (LOSSLESS)
    if codec is None:
        if filetype == 'mkv':
            codec = 'FFV1'
        elif filetype == 'avi':
            codec = 'H264'
        elif filetype == 'mp4':
            codec = 'mp4v'
        else:
            throw_G2L_warning(f'Did not find appropriate codec for filetype {filetype}.')
            codec = 'XVID'

    frames = frames.astype(np.uint8, copy=False)
    frames = resize_frames(frames, resize_factor=resize_factor)

    # frames doit être 3-dimensionnel [length, height, width]
    length, height, width = frames.shape

    video_path = video_rawpath + '.' + filetype

    # THE NEW WAY

    display(f'Saving video {video_path}...', end='\r')

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer= cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for framenumber in range(length):
        writer.write( np.repeat(frames[framenumber], 3).reshape((height, width, 3)).astype(np.uint8, copy=False) )

    writer.release()

    display(f'Video {video_path} saved', end='\n')

def save_acquisition_to_video(acquisition_path:str, do_timestamp:bool = True, fps:float = 25., filetype:str = 'mkv', codec:Optional[str] = None, resize_factor:int = 1):
    if not(is_this_a_gcv(acquisition_path)) and not(is_this_a_t16(acquisition_path)):
        # todo error ERROR here
        log_error(f'WAW')
        return

    frames = get_frames(acquisition_path, framenumbers = None, subregion=None)
    frames = np.copy(frames)
    length, height, width = frames.shape

    if do_timestamp and not(are_there_missing_frames(acquisition_path)):
        t = get_times(acquisition_path, framenumbers = None, unit='s')

        do_white_bckgnd = True

        text_offset = 10 if height >= 45 else 1

        thickness = 1 if height < 35 else 2

        antialiasing = height >= 20

        scale = 0.15 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
        fontScale = max(min(width,height)/(25/scale), 0.8)

        for i_frame in range(length):
            text = f't = {"{:05.3f}".format(t[i_frame])} s'

            testSize = cv2.getTextSize(text,
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=fontScale,
                                       thickness=thickness)[0][0]

            if do_white_bckgnd:
                cv2.putText(frames[i_frame], text,
                            (width - 1 - text_offset-testSize, height-1 - text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, # font
                            fontScale, # font scale
                            255, # color
                            thickness*3, # writing thickness
                            cv2.LINE_AA if antialiasing else cv2.FILLED,
                            False # Use the bottom left as origin (invert image)
                            )

            cv2.putText(frames[i_frame], text,
                        (width - 1 - text_offset - testSize, height-1 - text_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        fontScale, # font scale
                        0, # color
                        thickness, # writing thickness
                        cv2.LINE_AA if antialiasing else cv2.FILLED,
                        False # Use the bottom left as origin (invert image)
                        )

    save_frames_to_video(acquisition_path, frames,
                         fps = fps, filetype=filetype, codec=codec, resize_factor=resize_factor)

def save_all_gcv_videos(dataset:str, do_timestamp:bool = True, fps:float = 25., filetype:str = 'mkv', codec:Optional[str] = None, resize_factor:int = 1):
    log_info(f'Saving all the gcv acquisition in the dataset: {dataset}')

    dataset_path = '../' + dataset

    available_acquisitions =  find_available_videos(dataset_path, type='gcv')
    log_info(f'The following acquisition will be saved: {available_acquisitions}')

    for acquisition in available_acquisitions:
        acquisition_path = os.path.join(dataset_path, acquisition)

        if is_this_a_gcv(acquisition_path):
            save_acquisition_to_video(acquisition_path, do_timestamp=do_timestamp, fps=fps, filetype=filetype, codec=codec, resize_factor=resize_factor)
        else:
            throw_G2L_warning(f'GCV acquisition {acquisition} is found but does not exist ?')

from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace
from .. import utility, datasaving

###### TIFF 16-BITS VIDEO (t16) READING
from PIL import Image
from PIL.TiffTags import TAGS

def find_available_tiffs(dataset_path: str) ->List[str]:
    available_acquisitions = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and
                              np.prod([tifffile.endswith('.tiff') for tifffile in os.listdir(os.path.join(dataset_path, f))])]
    available_acquisitions.sort()
    return available_acquisitions

def find_available_t16(dataset_path: str) ->List[str]:
    available_acquisitions = [f for f in os.listdir(dataset_path) if is_this_a_t16(os.path.join(dataset_path, f))]
    available_acquisitions.sort()
    return available_acquisitions

def find_available_t8(dataset_path: str) ->List[str]:
    available_acquisitions = [f for f in os.listdir(dataset_path) if is_this_a_t8(os.path.join(dataset_path, f))]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_t16(acquisition_path: str) -> bool:
    """
    Checks if there is a Tiff 16-bits video (t16) for the given dataset.

    :param acquisition_path:
    :return:
    """
    if not os.path.isdir(acquisition_path):
        return False

    try:
        all_files_in_folder_are_tiff = np.prod([tifffile.endswith('.tiff') for tifffile in os.listdir(acquisition_path)])
        if not(all_files_in_folder_are_tiff):
            return False
    except:
        return False

    img_metaprobe = Image.open(os.path.join(acquisition_path, os.listdir(acquisition_path)[0]))
    meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}
    tiff_is_16_bits:bool = meta_dict['BitsPerSample'][0] == 16
    return tiff_is_16_bits

def is_this_a_t8(acquisition_path: str) -> bool:
    """
    Checks if there is a Tiff 16-bits video (t16) for the given dataset.

    :param acquisition_path:
    :return:
    """
    if not os.path.isdir(acquisition_path):
        return False

    try:
        all_files_in_folder_are_tiff = np.prod([tifffile.endswith('.tiff') for tifffile in os.listdir(acquisition_path)])
        if not(all_files_in_folder_are_tiff):
            return False
    except:
        return False

    img_metaprobe = Image.open(os.path.join(acquisition_path, os.listdir(acquisition_path)[0]))
    meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}
    tiff_is_8_bits:bool = meta_dict['BitsPerSample'][0] == 8
    return tiff_is_8_bits

def get_number_of_available_frames_t16(acquisition_path: str) -> int:
    if is_this_a_t16(acquisition_path):
        return len(os.listdir(acquisition_path))
    else:
        return 0

def get_number_of_available_frames_t8(acquisition_path: str) -> int:
    if is_this_a_t8(acquisition_path):
        return len(os.listdir(acquisition_path))
    else:
        return 0

def get_frames_t16(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

    all_images = os.listdir(acquisition_path)
    all_images.sort()

    img_metaprobe = Image.open(os.path.join(acquisition_path, all_images[0]))
    meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}

    width:int = meta_dict['ImageWidth'][0]
    height:int = meta_dict['ImageLength'][0]
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint16)

    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED)

    import time
    from imagecodecs import imread
    import matplotlib.pyplot as plt

    frames = np.empty([length, height, width], np.uint16)


    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = imread(os.path.join(acquisition_path, all_images[framenumber]), codec='tiff') # this is the fastest
        # frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED) # this is the 2nd fastest
        # frames[i_frame] = np.array(Image.open(os.path.join(acquisition_path, all_images[framenumber]))) # this is the 3rd fastest

    frames = (frames // 2**8).astype(np.uint8, copy=False)
    if verbose >= 2:
        # todo INFO here
        print('INFO: quality was degraded from 12-bits to 8-bits depth')

    return frames

def get_frames_t16_conservequality(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

    all_images = os.listdir(acquisition_path)
    all_images.sort()

    img_metaprobe = Image.open(os.path.join(acquisition_path, all_images[0]))
    meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}

    width:int = meta_dict['ImageWidth'][0]
    height:int = meta_dict['ImageLength'][0]
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint16)

    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED)

    import time
    from imagecodecs import imread
    import matplotlib.pyplot as plt

    frames = np.empty([length, height, width], np.uint16)


    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = imread(os.path.join(acquisition_path, all_images[framenumber]), codec='tiff') # this is the fastest
        # frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED) # this is the 2nd fastest
        # frames[i_frame] = np.array(Image.open(os.path.join(acquisition_path, all_images[framenumber]))) # this is the 3rd fastest

    return frames


def get_frames_t8(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

    all_images = os.listdir(acquisition_path)
    all_images.sort()

    img_metaprobe = Image.open(os.path.join(acquisition_path, all_images[0]))
    meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}

    width:int = meta_dict['ImageWidth'][0]
    height:int = meta_dict['ImageLength'][0]
    length:int = len(framenumbers)

    frames = np.empty([length, height, width], np.uint16)

    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED)

    import time
    from imagecodecs import imread
    import matplotlib.pyplot as plt

    frames = np.empty([length, height, width], np.uint8)


    for i_frame, framenumber in enumerate(framenumbers):
        frames[i_frame] = imread(os.path.join(acquisition_path, all_images[framenumber]), codec='tiff') # this is the fastest
        # frames[i_frame] = cv2.imread(os.path.join(acquisition_path, all_images[framenumber]), cv2.IMREAD_UNCHANGED) # this is the 2nd fastest
        # frames[i_frame] = np.array(Image.open(os.path.join(acquisition_path, all_images[framenumber]))) # this is the 3rd fastest

    return frames

def get_acquisition_frequency_t16(acquisition_path: str, unit = None, verbose:Optional[int]=None) -> float:
    if '20230309_chronos_b' in acquisition_path:
        if 'f30tiff' in acquisition_path:
            return 1200
        if 'f40tiff' in acquisition_path:
            return 1600
        if 'f50tiff' in acquisition_path:
            return 2000
        if 'f60tiff' in acquisition_path:
            return 600
        if 'f80tiff' in acquisition_path:
            return 800
        if 'f100tiff' in acquisition_path:
            return 1000
        if 'f150tiff' in acquisition_path:
            return 1500
        if 'f250tiff' in acquisition_path:
            return 2500
        if 'f300tiff' in acquisition_path:
            return 3000
        if 'f400tiff' in acquisition_path:
            return 4000
        if 'f400stiff' in acquisition_path:
            return 4000
        if 'f600tiff' in acquisition_path:
            return 6000
        if 'f1000tiff' in acquisition_path:
            return 10000
        if 'f1500tiff' in acquisition_path:
            return 10000
    if '20230309_chronos_a' in acquisition_path:
        if 'f30tiff' in acquisition_path:
            return 1200
        if 'f40tiff' in acquisition_path:
            return 1600
        if 'f50tiff' in acquisition_path:
            return 2000
        if 'f60tiff' in acquisition_path:
            return 600
        if 'f80tiff' in acquisition_path:
            return 800
        if 'f100tiff' in acquisition_path:
            return 1000
        if 'f150tiff' in acquisition_path:
            return 1500
        if 'f250tiff' in acquisition_path:
            return 3000
        if 'f300tiff' in acquisition_path:
            return 3000
        if 'f400tiff' in acquisition_path:
            return 3000
        if 'f400breakuptiff' in acquisition_path:
            return 3000
    return 1.

def get_acquisition_duration_t16(acquisition_path: str, framenumbers:np.ndarray, unit = None, verbose:Optional[int]=None) -> float:
    acqu_freq: float = get_acquisition_frequency_t16(acquisition_path, unit =  'Hz', verbose = verbose)
    acqu_frames: float = len(framenumbers)

    return (acqu_frames - 1) / acqu_freq

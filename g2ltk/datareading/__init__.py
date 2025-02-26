from typing import Optional, Tuple, List, Union
import numpy as np
# import cv2 # to manipulate images and videos
# import os # to navigate in the directories
# import shutil # to remove directories

# from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace, log_subtrace
# from .. import utility, datasaving

# Custom typing
Framenumbers = Optional[Union[np.ndarray, List[int]]]
Subregion = Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]#  start_x, start_y, end_x, end_y

###### GENERAL VIDEO READING

def alert_me_if_there_is_no_video(acquisition_path: str) -> None:
    if not(is_this_a_video(acquisition_path)):
        for i in range(10):
            log_warn(f'No video named {acquisition_path} exists !')

def is_this_a_video(acquisition_path:str) -> bool:
    """
    Checks if there is a video of any kind for the given acquisition path.

    :param acquisition_path:
    :return:
    """
    if is_this_a_gcv(acquisition_path):
        return True
    elif is_this_a_t16(acquisition_path):
        return True
    elif is_this_a_t8(acquisition_path):
        return True
    elif is_this_a_lcv(acquisition_path):
        return True
    elif is_this_a_mp4(acquisition_path):
        return True
    elif is_this_a_mov(acquisition_path):
        return True
    else:
        log_warn(f"There is no video at {acquisition_path}.")
        return False

def find_available_videos(dataset_path: str, type:Optional[str]=None) ->List[str]:
    all_types = ['gcv', 't16', 't8', 'flv', 'mp4', 'mov']
    log_subtrace(f'Finding available videos of type {type or all_types} at {dataset_path}')

    if type is None or type=='all':
        available_acquisitions = []
        for subtype in all_types:
            available_acquisitions += find_available_videos(dataset_path, type=subtype)
    elif type=='gcv':
        available_acquisitions = find_available_gcv(dataset_path)
    elif type=='t16':
        available_acquisitions = find_available_t16(dataset_path)
    elif type=='t8':
        available_acquisitions = find_available_t8(dataset_path)
    elif type=='flv':
        available_acquisitions = find_available_lcv(dataset_path)
    elif type=='mp4':
        available_acquisitions = find_available_mp4(dataset_path)
    elif type=='mov':
        available_acquisitions = find_available_mov(dataset_path)
    else:
        log_error(f'Unsupported data type: {type}')
        return []
    available_acquisitions.sort()
    return available_acquisitions

def describe_dataset(dataset_path: str, type:Optional[str]=None, makeitshort=False) -> None:
    available_acquisitions = find_available_videos(dataset_path, type=type)
    if len(available_acquisitions) == 0:
        log_warn(f'No videos {"" if type is None else f"of type {type} "}found in dataset {dataset_path}!')
        return
    if makeitshort:
        display(f'Available acquisitions: {available_acquisitions}')
    else:
        display('Available acquisitions:')
        for acquisition in available_acquisitions:
            display(f"- '{acquisition}' (TODO type(s))")

def find_available_datasets(root_path: str) ->List[str]:

    log_debug(f'Possible datasets in {root_path}: {os.listdir(root_path)}')
    available_datasets = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and len(find_available_videos(os.path.join(root_path, d))) > 0]
    return available_datasets

def get_number_of_available_frames(acquisition_path: str) -> Optional[int]:
    if not is_this_a_video(acquisition_path):
        log_error(f"There is no video at {acquisition_path}. Could not get number of available frames.")
        return None
    if is_this_a_gcv(acquisition_path):
        return get_number_of_available_frames_gcv(acquisition_path)
    elif is_this_a_t16(acquisition_path):
        return get_number_of_available_frames_t16(acquisition_path)
    elif is_this_a_t8(acquisition_path):
        return get_number_of_available_frames_t8(acquisition_path)
    elif is_this_a_lcv(acquisition_path):
        return get_number_of_available_frames_lcv(acquisition_path)
    elif is_this_a_mp4(acquisition_path):
        return get_number_of_available_frames_mp4(acquisition_path)
    elif is_this_a_mov(acquisition_path):
        return get_number_of_available_frames_mov(acquisition_path)
    else:
        log_error(f'Cannot get number of available frames for {acquisition_path}')
    return None

def get_acquisition_frequency(acquisition_path:str, unit = None, verbose:Optional[int]=None) -> float:
    if not is_this_a_video(acquisition_path):
        log_error(f"There is no video at {acquisition_path}. Could not get acquisition frequency.")
        return -1.
    if is_this_a_gcv(acquisition_path):
        return get_acquisition_frequency_gcv(acquisition_path, unit=unit, verbose=verbose)
    if is_this_a_t16(acquisition_path):
        return get_acquisition_frequency_t16(acquisition_path, unit=unit)
    if is_this_a_t8(acquisition_path):
        return get_acquisition_frequency_t8(acquisition_path, unit=unit)
    if is_this_a_mov(acquisition_path):
        return get_acquisition_frequency_mov(acquisition_path, unit=unit, verbose=verbose)
    if is_this_a_mp4(acquisition_path):
        return get_acquisition_frequency_mp4(acquisition_path, unit=unit, verbose=verbose)
    else:
        log_error(f"Could not get acquisition frequency for {acquisition_path}: returning -1.")
        return -1.

def get_acquisition_duration(acquisition_path:str, framenumbers:Optional[np.ndarray], unit = None, verbose:Optional[int]=None) -> Optional[float]:
    log_subtrace('func:get_acquisition_duration')
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        log_error("ERROR Wrong framenumber, couldnt format it")
        return None

    if is_this_a_gcv(acquisition_path):
        return get_acquisition_duration_gcv(acquisition_path, framenumbers=framenumbers, unit=unit, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        return get_acquisition_duration_t16(acquisition_path, framenumbers=framenumbers, unit=unit)
    else:
        fns = len(framenumbers)
        freq_hz = get_acquisition_frequency(acquisition_path, unit='Hz', verbose=verbose)
        try:
            duration_s = fns / freq_hz
            return duration_s
        except:
            pass
        log_error('Could not get acquisition duration')
        return None

def format_framenumbers(acquisition_path:str, framenumbers:Framenumbers=None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    if not is_this_a_video(acquisition_path):
        log_debug(f"There is no video at {acquisition_path}. Could not format the framenumbers {framenumbers}.")
        return None
    number_of_available_frames = get_number_of_available_frames(acquisition_path)
    if number_of_available_frames is None:
        log_debug('No available frames. Formatted framenumber is None.')
        return None
    if framenumbers is None: # Default value for framenumbers
        framenumbers = np.arange(number_of_available_frames)
    framenumbers = np.array(framenumbers) # handle the array_like case
    #Check that the demand is reasonable
    if framenumbers.min() < 0:
        log_error('Asked for negative framenumber ??')
        return None
    if framenumbers.max() >= get_number_of_available_frames(acquisition_path):
        log_error(f'Requested framenumber {framenumbers.max()} while there are only {get_number_of_available_frames(acquisition_path)} frames for this dataset.')
        return None
    return framenumbers

def get_geometry(acquisition_path:str, framenumbers:Framenumbers = None, subregion:Subregion = None, verbose:Optional[int]=None) -> Optional[Tuple]:
    log_subtrace('func:get_geometry')
    if not is_this_a_video(acquisition_path):
        log_error(f"There is no video at {acquisition_path}. Could not get geometry.")
        return None
    formatted_fns = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if formatted_fns is None: return None

    length = formatted_fns.size

    frame = get_frame(acquisition_path, int(formatted_fns[0]), subregion=subregion, verbose=verbose)

    if frame is None: return None

    height, width = frame.shape

    return length, height, width

def get_frames(acquisition_path:str, framenumbers:Framenumbers=None, subregion:Subregion = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    if not is_this_a_video(acquisition_path):
        log_error(f"There is no video at {acquisition_path}. Could not get frames")
        return None
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        log_warn(f"No framenumbers specified: could not get frames for {acquisition_path}")
        return None

    # Retrieve the frames
    frames = None
    if is_this_a_gcv(acquisition_path):
        frames = get_frames_gcv(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        frames = get_frames_t16(acquisition_path, framenumbers)
    elif is_this_a_t8(acquisition_path):
        frames = get_frames_t8(acquisition_path, framenumbers)
    elif is_this_a_lcv(acquisition_path):
        frames = get_frames_lcv(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_mp4(acquisition_path):
        frames = get_frames_mp4(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_mov(acquisition_path):
        frames = get_frames_mov(acquisition_path, framenumbers, verbose=verbose)
    else:
        log_subtrace('ERROR INCOMING func:get_frames')
        log_error(f'Cannot get frames: there is no video at {acquisition_path}')
        return None

    # crop the subregion
    frames = crop_frames(frames, subregion=subregion)

    # # reverse the y direction
    # # so that we can visualize with plt.imshow and origin='lower'
    # # and still have the right lines numbers
    # if frames is not None:
    #     frames = frames[:, ::-1, :]

    return frames

def get_frame(acquisition_path:str, framenumber:Optional[int], subregion:Subregion = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    if not is_this_a_video(acquisition_path):
        log_error(f"There is no video at {acquisition_path}. Could not get frame {framenumber}.")
        return None
    if framenumber is None:
        log_warn(f"No framenumber specified: could not get frame for {acquisition_path}")
        return None
    frames = get_frames(acquisition_path, np.array([framenumber]), subregion=subregion, verbose=verbose)
    if frames is None:
        log_warn(f"Could not get frame {framenumber} for {acquisition_path}")
        return None
    return frames[0]

def get_times(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, unit = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        log_error("ERROR Wrong framenumber, couldnt format it", verbose=verbose)
        return None

    # Retrieve the frames
    times = None
    if is_this_a_gcv(acquisition_path):
        times = get_times_gcv(acquisition_path, framenumbers, unit=unit, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        times = (np.arange(framenumbers.max()+1) / get_acquisition_frequency_t16(acquisition_path))[framenumbers]
    else:
        freq_hz = get_acquisition_frequency(acquisition_path, unit='Hz', verbose=verbose)
        try:
            times = framenumbers / freq_hz
            return times
        except:
            pass
        log_error('Could not get times')
        return None

    return times

def missing_frames(acquisition_path: str, verbose:Optional[int]=None) -> List:
    """
    Identifies missing frame in a video.

    :param acquisition_path:
    :param verbose:
    :return:
    """
    if is_this_a_gcv(acquisition_path):
        return missing_framenumbers_gcv(acquisition_path, verbose=verbose)
    else:
        log_warn(f'Could not deduce the number of missing frames for video {acquisition_path}', verbose=verbose)
    return []

def missing_frames_in_framenumbers(acquisition_path: str, framenumbers:Optional[np.ndarray]=None, verbose:Optional[int]=None) -> List:
    log_subtrace('func:get_acquisition_duration')
    all_missing_chunks = missing_frames(acquisition_path, verbose=verbose)
    explicit_framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)

    # get the missing frames which are in the requested framenumbers
    missing_chunks_in_framenumbers = []
    for chunk in all_missing_chunks:
        chunk_missing = []
        for frame in chunk:
            if frame in explicit_framenumbers:
                chunk_missing.append(frame)
        if len(chunk_missing) > 0:
            missing_chunks_in_framenumbers.append(chunk_missing)

    return missing_chunks_in_framenumbers

def are_there_missing_frames(acquisition_path: str, framenumbers:Optional[np.ndarray]=None, verbose:Optional[int]=None) -> bool:
    missing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=framenumbers, verbose=verbose)

    nbr_of_missing_chunks = len(missing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in missing_chunks])

    if  nbr_of_missing_chunks > 0:
        log_trace(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_trace(f'Missing chunks: {missing_chunks}', verbose=verbose)
        return True

    log_trace('No missing frames', verbose=verbose)
    return False

def describe_acquisition(dataset:str, acquisition:str, framenumbers:Optional[np.ndarray]=None, subregion:Subregion=None, verbose:Optional[int]=None):
    log_subtrace('func:describe_acquisition')
    log_info(f'Acquisition: "{acquisition}" ({dataset})')

    dataset_path = os.path.join('../', dataset)
    acquisition_path = os.path.join(dataset_path, acquisition)
    if not(is_this_a_video(acquisition_path)):
        log_debug(f'Videos in {dataset} are {find_available_videos(dataset_path)}', verbose=verbose)
        log_error(f'No video named {acquisition} in dataset {dataset}', verbose=verbose)
        return

    # general
    frequency = get_acquisition_frequency(acquisition_path, unit="Hz", verbose=verbose)

    log_info(f'Acquisition frequency: {round(frequency, 2)} Hz', verbose=verbose)

    # raw video file
    maxlength, maxheight, maxwidth = get_geometry(acquisition_path, framenumbers=None, subregion=None)
    maxsize = maxlength*maxheight*maxwidth
    maxduration = get_acquisition_duration(acquisition_path, framenumbers=None, unit="s")
    maxmissing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=None, verbose=verbose)
    nbr_of_missing_chunks = len(maxmissing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in maxmissing_chunks])

    log_debug(f'Acquisition information:', verbose=verbose)
    log_debug(f'Frames dimension: {maxheight}x{maxwidth}', verbose=verbose)
    log_debug(f'Length: {maxlength} frames ({round(maxduration, 2)} s - {round(maxsize / 10 ** 6, 0)} MB)', verbose=verbose)
    if  nbr_of_missing_chunks > 0:
        log_debug(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_debug(f'Missing chunks: {maxmissing_chunks}', verbose=verbose)
    else:
        log_debug('No missing frames for this acquisition', verbose=verbose)

    # chosen data chunk
    length, height, width = get_geometry(acquisition_path, framenumbers = framenumbers, subregion=subregion)
    framesize = height*width
    framesize_kB = int(np.rint(framesize/10**3))
    framessize = length*framesize
    framessize_MB = int(np.rint(framessize/10**6))
    duration = get_acquisition_duration(acquisition_path, framenumbers=framenumbers, unit="s")
    missing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=framenumbers, verbose=verbose)
    nbr_of_missing_chunks = len(missing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in missing_chunks])

    log_info(f'Frames dimension: {height}x{width} ({framesize_kB} kB each)', verbose=verbose)
    log_info(f'Length: {length} frames ({round(duration, 2)} s - {f"{framessize_MB} MB" if framessize_MB < 1000 else f"{round(framessize_MB/1000, 3)} GB"})', verbose=verbose)
    if  nbr_of_missing_chunks > 0:
        log_info(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_info(f'Missing chunks: {missing_chunks}', verbose=verbose)
    else:
        log_info('No missing frames in chosen framenumbers', verbose=verbose)

###### SPECIFIC VIDEO READING

### GCV VIDEO READING

from .reading_gcv import *

### TIFF 16-BITS VIDEO (t16) READING

from .reading_tiffs import *

### LOSSLESSLY COMPRESSED VIDEO (lcv) READING

from .reading_lcv import *

### LOSSY COMPRESSED VIDEO (mp4) READING

from .reading_mp4 import *

### LOSSY COMPRESSED VIDEO (mov) READING

from .reading_mov import *

###### FRAMES EDITING

# Remove a background
def remove_bckgnd_from_frames(frames:np.ndarray, bckgnd:np.ndarray=None):
    if bckgnd is None:
        return frames
    # this takes time bck of the typecasting
    frames_bckgnd_removed = frames.astype(float) - np.expand_dims(bckgnd, axis=0) # remove bckgnd
    frames_bckgnd_removed -= np.min(frames_bckgnd_removed) # minimum is zero
    frames_bckgnd_removed *= 255/np.max(frames_bckgnd_removed) # maximum is 255
    return frames_bckgnd_removed

def remove_bckgnd_from_frame(frame, bckgnd:np.ndarray=None):
    return remove_bckgnd_from_frames(np.array([frame]), bckgnd=bckgnd)[0]

# Crop to a subregion

def crop_frames(frames, subregion:Subregion = None):
    if frames is None: return None
    if subregion is not None:
        start_x, start_y, end_x, end_y = subregion
        return frames[:, start_y:end_y, start_x:end_x]
    return frames

def crop_frame(frame, subregion:Subregion = None):
    return crop_frames(np.array([frame]), subregion=subregion)[0]

# Resize to
def resize_frames(frames:np.ndarray, resize_factor:int = 1):
    """
    Resizes an array with the given resize factor, using opencv resize function (the fastest).
    IT WILL CONVERT THE FRAMES TO UINT8 SINCE OPENCV ONLY SUPPORTS THAT.

    :param frames:
    :param resize_factor:
    :return:
    """
    frames = frames.astype(np.uint8, copy=False)
    if resize_factor == 1:
        return frames

    length, height, width = frames.shape

    new_height, new_width = height * resize_factor, width * resize_factor

    frames_enhanced = np.zeros((length, new_height, new_width), dtype=np.uint8)

    for framenumber in range(length):
        # Other possibilites were skimage.resize but it is more than 10 times slower.
        frames_enhanced[framenumber] = cv2.resize(frames[framenumber], (new_width, new_height))

    return frames_enhanced

def resize_frame(frame, resize_factor:int = 1):
    return resize_frames(np.array([frame]), resize_factor=resize_factor)[0]

###### VIDEO SAVING

from .saving_videos import *

###### GET INFO
def get_t_frames(acquisition_path:str, framenumbers:Framenumbers = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    """Returns the time, in frames (integers)"""
    return format_framenumbers(acquisition_path, framenumbers = framenumbers, verbose = verbose).astype(float)

def get_t_s( acquisition_path:str, framenumbers:Framenumbers = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    return get_times(acquisition_path, framenumbers=framenumbers, unit='s', verbose = verbose)
def get_t_ms(acquisition_path:str, framenumbers:Framenumbers = None, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    return get_times(acquisition_path, framenumbers=framenumbers, unit='ms', verbose = verbose)
def get_x_px(acquisition_path:str, framenumbers:Framenumbers = None, subregion:Subregion = None, resize_factor:int = 1, verbose:Optional[int]=None) -> Optional[np.ndarray]:
    length, height, width = get_geometry(acquisition_path, framenumbers=framenumbers, subregion=subregion, verbose = verbose)
    return np.arange(width * resize_factor) / resize_factor
def get_x_mm(acquisition_path:str, framenumbers:Framenumbers = None, subregion:Subregion = None, resize_factor:int = 1, px_per_mm:float = 1., verbose:Optional[int]=None) -> Optional[np.ndarray]:
    return get_x_px(acquisition_path, framenumbers, subregion, resize_factor, verbose) / px_per_mm
def get_x_m( acquisition_path:str, framenumbers:Framenumbers = None, subregion:Subregion = None, resize_factor:int = 1, px_per_mm:float = 1., verbose:Optional[int]=None) -> Optional[np.ndarray]:
    return get_x_mm(acquisition_path, framenumbers, subregion, resize_factor, px_per_mm, verbose) / 1e3
def get_x_um(acquisition_path:str, framenumbers:Framenumbers = None, subregion:Subregion = None, resize_factor:int = 1, px_per_mm:float = 1., verbose:Optional[int]=None) -> Optional[np.ndarray]:
    return get_x_mm(acquisition_path, framenumbers, subregion, resize_factor, px_per_mm, verbose) * 1e3


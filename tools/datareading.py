from typing import Optional, Any, Tuple, Dict, List
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories
import shutil # to remove directories

from tools import display, throw_G2L_warning, log_error, log_warn, log_info, log_dbug, log_trace
from tools import utility, datasaving

# Custom typing
Meta = Dict[str, str]
Stamps = Dict[str, np.ndarray]
Subregion = Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]#  start_x, start_y, end_x, end_y

### path routines


###### GEVCAPTURE VIDEO (gcv) READING

def find_available_gcv(dataset_path: str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if is_this_a_gcv(os.path.join(dataset_path, f[:-4]))]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_gcv(acquisition_path:str) -> bool:
    """
    Checks if there is a GevCapture Video (GCV) for the given dataset.

    :param acquisition_path:
    :return:
    """
    gcv_path = acquisition_path + '.gcv'
    # check that we indeed have a folder containing the right files
    if not os.path.isdir(gcv_path): return False
    # check that the folder contain the right number of files
    files = os.listdir(gcv_path)
    if len(files) != 3: return False
    stampsfiles = [f for f in files if f.endswith('.stamps')]
    metafiles = [f for f in files if f.endswith('.meta')]
    rawvideofiles = [f for f in files if f.endswith('.raw')]
    # check that we have one stamp file
    if len(stampsfiles) != 1: return False
    if len(metafiles) != 1: return False
    if len(rawvideofiles) != 1: return False
    return True

def get_number_of_available_frames_gcv(acquisition_path:str) -> int:
    n_frames_stamps = get_number_of_available_frames_stamps(acquisition_path)
    n_frames_rawvideo = get_number_of_available_frames_rawvideo(acquisition_path)

    if n_frames_stamps != n_frames_rawvideo:
        # todo warning here
        print(f'WARNING: The stamps file mentions {n_frames_stamps} frames while there are {n_frames_rawvideo} frames availables in the raw video file.')
    return n_frames_rawvideo

def get_acquisition_frequency_gcv(acquisition_path:str, unit = None, verbose:int = 1) -> float:
    if unit is None: # default unit
        unit = 'Hz'
    factor:np.int64 = 1 # Multiplication factor for the time unit
    # raw unit is in ns
    if unit == 'Hz':
        factor  = 1
    else:
        print(f'Unrecognized frequency unit : {unit}')

    meta_info = retrieve_meta(acquisition_path)
    freq_meta = float(meta_info['captureFrequency']) # Hz

    if are_there_missing_frames(acquisition_path, verbose=verbose):
        return freq_meta

    full_stamps = retrieve_stamps(acquisition_path, verbose=verbose)
    camera_timestamps = get_camera_timestamps(full_stamps, unit='s', verbose=verbose)

    freq_camera_timestamps = 1/np.mean(camera_timestamps[1:]-camera_timestamps[:-1])

    # # These are too imprecise to be useful
    # computer_timestamps = get_computer_timestamps(full_stamps, unit='s')
    #
    # freq_computer_timestamps = 1/np.mean(computer_timestamps[1:]-computer_timestamps[:-1])

    return freq_camera_timestamps

def get_acquisition_duration_gcv(acquisition_path:str, framenumbers:np.ndarray, unit = None, verbose:int = 1) -> float:
    full_stamps = retrieve_stamps(acquisition_path)
    camera_timestamps = get_camera_timestamps(full_stamps, unit=unit, verbose=verbose)

    camera_timestamps = camera_timestamps[framenumbers]

    return np.max(camera_timestamps) - np.min(camera_timestamps)

def get_times_gcv(acquisition_path:str, framenumbers:np.ndarray, unit=None, verbose:int = 1) -> Optional[np.ndarray]:
    full_stamps = retrieve_stamps(acquisition_path)
    camera_timestamps = get_camera_timestamps(full_stamps, unit=unit)

    times = camera_timestamps[framenumbers]

    return times

def get_frames_gcv(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:
    return get_frames_rawvideo(acquisition_path, framenumbers=framenumbers, verbose=verbose)

### META READING

def retrieve_meta(acquisition_path: str) -> Meta:
    gcv_path = acquisition_path + '.gcv'
    meta_filename = [f for f in os.listdir(gcv_path) if f.endswith('.meta')][0]
    meta_path = os.path.join(gcv_path, meta_filename)
    if not(os.path.isfile(meta_path)): raise(Exception(f'ERROR: Problem with the {acquisition_path} meta file (it does not exist).'))
    meta:Meta = {}
    try:
        with open(meta_path, 'r') as meta_file:
            for line in meta_file.readlines():
                if '=' in line:
                    if line.count('=') > 1:
                        #Todo: add a real warning here
                        print(f'WARNING: Ambiguity when parsing the {acquisition_path} meta file: is there a rogue "=" somewhere in it?.')
                    variable = line[:-1].split('=')[0]
                    value = '='.join( line[:-1].split('=')[1:] )
                    meta[variable] = value
                else:
                    #Todo: add a real warning here
                    print(f'WARNING: Could not parse correctly the {acquisition_path} meta file.')
    except:
        raise(Exception(f'ERROR: Problem with the {acquisition_path} meta file (probably it could not be opened).'))

    return meta

### STAMPS READING

def retrieve_stamps(acquisition_path:str, verbose:int = 1) -> Stamps:
    """
    Gets all the stamps of a video

    :param video_path:
    :param verbose:
    :return:
    """
    # get the info in the .stamps file
    # camera time is in ns and computer time in ms
    gcv_path = acquisition_path + '.gcv'
    stamps_filename = [f for f in os.listdir(gcv_path) if f.endswith('.stamps')][0]
    stamps_path = os.path.join(gcv_path, stamps_filename)
    if not(os.path.isfile(stamps_path)): raise(Exception(f'ERROR: Problem with the {acquisition_path} stamps file (it does not exist).'))

    framenumber, camera_time, computer_time = [], [], []
    try:
        with open(stamps_path, 'r') as stamps_file:
            for line in stamps_file.readlines():
                if line.count('\t') == 2:
                    framenumber.append(line[:-1].split('\t')[0])     # The frame number, an integer from 0 to N-1 (N number of frames) (int)
                    camera_time.append(line[:-1].split('\t')[1])     # The time given by the camera, in ns (int)
                    computer_time.append(line[:-1].split('\t')[2])   # The time given by the computer, in ms (int)
                else:
                    #Todo: add a real warning here
                    print(f'WARNING: Could not parse correctly the {acquisition_path} stamps file.')
    except:
        raise(Exception(f'ERROR: Problem with the {acquisition_path} stamps file (probably it could not be opened).'))
    #Todo: check if a typecasting error can happen here ?
    stamps:Stamps = {'framenumber': np.array(framenumber, dtype=int),
                     'camera_time': np.array(camera_time, dtype=np.int64),
                     'computer_time': np.array(computer_time, dtype=np.int64)}
    return stamps

def get_number_of_available_frames_stamps(acquisition_path: str) -> int:
    full_stamps:Stamps = retrieve_stamps(acquisition_path)
    n_frames_tot:int = len(full_stamps['framenumber'])
    return n_frames_tot

def missing_framenumbers_gcv(acquisition_path: str, verbose:int=1) -> List:
    """
    Identifies missing frame in a GCV video using the timestamps.

    :param acquisition_path:
    :param verbose:
    :return:
    """
    full_stamps = retrieve_stamps(acquisition_path)
    framenumbers = full_stamps['framenumber']
    gaps = framenumbers[1:] - framenumbers[:-1] - 1
    missing_gaps = gaps[np.where(gaps != 0)[0]]
    first_missing_frames = framenumbers[np.where(gaps != 0)[0]]+1
    first_missing_frames -= framenumbers[0] # relative numerotation of framenumbers
    all_missing_chunks = []
    for i in range(len(missing_gaps)):
        all_missing_chunks.append([])
        for j in range(missing_gaps[i]):
            all_missing_chunks[i].append(first_missing_frames[i] + j)
    log_trace(f'Missing frames for {acquisition_path}:', verbose=verbose)
    log_trace(f'{all_missing_chunks}', verbose=verbose)
    return all_missing_chunks

def identify_missing_framenumbers(framenumbers:np.ndarray, verbose:int = 1) -> np.ndarray:
    """
    Identifies missing frame using the timestamps.

    For example, if framenumbers is [1, 2, 3, 7, 8, 9, 11]
    Then the output is:
        'Missing frames numbered from 4 to 6 (3 missing frames)'
        'Missing frame number 10'
        'Total: 4 missing frames'
    and the function returns [4 5 6 10]

    :param framenumbers:
    :param verbose:
    :return:
    """
    # Here MF means missing frame (abbreviated for better code lisibility)
    arg_MF = np.where((framenumbers[1:] - framenumbers[:-1]) != 1)[0]
    MF = np.array([], dtype=int)
    for i_MF in arg_MF.astype(int):
        # for each bunch of MFs
        first_MF = framenumbers[i_MF] + 1
        last_MF = framenumbers[i_MF + 1] - 1
        number_of_MFs = last_MF - first_MF + 1
        if verbose >= 2:
            if number_of_MFs > 1:
                print(f'Missing frames numbered from {first_MF} to {last_MF} ({number_of_MFs} missing frames)')
            elif number_of_MFs == 1:
                print(f'Missing frame number {first_MF}')
            else:
                # todo error here
                print("ERROR: this should not be reached")
        MF = np.concatenate((MF, np.arange(first_MF, last_MF + 1, dtype=int)))
    if len(MF) > 0 and verbose >= 1:
        print(f'Total: {len(MF)} missing frames')

    return MF

def get_camera_timestamps(stamps:Stamps, unit:str = None, verbose:int = 1) -> np.ndarray:
    if unit is None: # default unit
        unit = 's'
    factor:np.int64 = 1 # Multiplication factor for the time unit
    # raw unit is in ns
    if unit == 'ns':
        factor  = 1
    elif unit == 'us':
        factor = 10**3
    elif unit == 'ms':
        factor = 10**6
    elif unit == 's':
        factor = 10**9
    elif unit == 'min':
        factor = 10**9 * 60
    else:
        print(f'Unrecognized time unit : {unit}')

    camera_timestamps:np.ndarray = stamps['camera_time'].copy()
    camera_timestamps -= camera_timestamps[0] # start of the video is time 0

    return camera_timestamps / factor

def get_computer_timestamps(stamps:Stamps, unit:str = 's', verbose:int = 1) -> np.ndarray:
    # Multiplication factor for the time unit
    factor:np.int64 = 1
    # raw unit is ns
    if unit == 'ns':
        factor  = 10**6
    elif unit == 'us':
        factor = 10**3
    elif unit == 'ms':
        factor = 1
    elif unit == 's':
        factor = 10**3
    elif unit == 'min':
        factor = 10**3 * 60
    else:
        print(f'Unrecognized time unit : {unit}')

    computer_timestamps:np.ndarray = stamps['computer_time'].copy()

    return computer_timestamps / factor

def get_monotonic_timebound(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, unit = None, verbose:int = 1) -> Tuple[float, float]:
    if framenumbers is None: # Default value for framenumbers
        framenumbers = np.arange(get_number_of_available_frames(acquisition_path))
    #Check that the demand is reasonable
    if framenumbers.min() < 0:
        #Todo: ERROR here
        print('Asked for negative framenumber ??')
        return None
    if framenumbers.max() >= get_number_of_available_frames_stamps(acquisition_path):
        #Todo: WARNING here
        print(f'Requested framenumber {framenumbers.max()} while there are only {get_number_of_available_frames(acquisition_path)} frames for this dataset.')
        framenumbers = framenumbers[framenumbers < get_number_of_available_frames_stamps(acquisition_path)]

    # Retrieve the frames
    times = get_computer_timestamps(retrieve_stamps(acquisition_path, verbose), unit=unit, verbose=verbose)

    good_times = times[framenumbers]

    return good_times[0], good_times[-1]

def get_regularly_spaced_stamps(full_stamps:Stamps, start_framenumber:int = 0, end_framenumber:int = -1, interval:int  = 1, verbose:int = 1) -> Optional[Stamps]:
    """
    This functions gives a selection of stamps that the user wants, from all the stamps available according to the stamps file.


    :param full_stamps: All the stamps available
    :param start_framenumber:
    :param end_framenumber:
    :param interval:
    :param verbose:
    :return:
    """
    all_framenumbers = full_stamps['framenumber']
    last_framenumber = all_framenumbers[-1]
    # Putting end_frame = -1 means the end frame is the last frame
    if end_framenumber == -1:
        end_framenumber = last_framenumber

    ### Check that the frames we are asking for makes sense
    if start_framenumber < 0:
        print('Start frame < 0')
        return None
    if start_framenumber > last_framenumber:
        print(f'The requested start frame ({start_framenumber}) is after the last recorded frame ({last_framenumber})')
        return None
    if start_framenumber > end_framenumber:
        print(f'The requested start frame ({start_framenumber}) is after the requested end frame ({end_framenumber})')
        return None
    if end_framenumber < 0:
        print('End frame < 0')
        return None
    if end_framenumber > last_framenumber:
        #Todo: WARNING here
        print(f'The requested end frame ({end_framenumber}) is after the last recorded frame ({last_framenumber}).')
        print(f'Changing the end frame to {last_framenumber}. You might get less frames than expected.')
        end_framenumber = last_framenumber

    ### Check for missing frames
    missing_framenumbers = identify_missing_framenumbers(all_framenumbers, verbose=verbose)
    if not len(missing_framenumbers) == 0:
        print(f'There are missing frames ! Frames {list(missing_framenumbers)} are missing.')
    # here MF means missing frame (abbreviated for code lisibility)
    MFs_after_start = missing_framenumbers >= start_framenumber
    MFs_before_end = missing_framenumbers <= end_framenumber
    MFs_in_requested_interval = MFs_after_start * MFs_before_end
    there_are_MFs_in_the_request_interval:bool = 1 in MFs_in_requested_interval
    if there_are_MFs_in_the_request_interval:
        #Todo: WARNING here
        print(f'There are missing frames in the requested interval. The returned frames will not be evenly spaced.')

    frames_correctly_spaced = (all_framenumbers - start_framenumber) % interval == 0
    frames_after_start = all_framenumbers >= start_framenumber
    frames_before_end = all_framenumbers <= end_framenumber
    valid_frames = frames_correctly_spaced * frames_after_start * frames_before_end

    stamps_wanted = {**full_stamps}
    for k in stamps_wanted:
        stamps_wanted[k] = stamps_wanted[k][valid_frames]

    # Last checkup !
    #Todo: Understand why this shit seems to work
    ## THIS PART IS A BY SHADY AND THE COMPUTING IS BIZARRE< REWRITE IT PLZ
    n_frames_asked_for = (end_framenumber - start_framenumber + 1) // interval + (1 if (end_framenumber - start_framenumber + 1) % interval != 0 else 0) # The number we should have
    n_frames_returned = len(stamps_wanted['framenumber']) # The frames we send back
    if n_frames_returned != n_frames_asked_for:
        #Todo: WARNING here
        # This happens either if
        # (a) there were missing frames (should be a warning before telling that from the index function) or
        # (b) something elsed foired in this function, a supplementary check should not hurt !
        print('WARNING, STAMPS ARE NOT COHERENT WITH VIDEO')

    return stamps_wanted

## RAWVIDEO READING

def get_number_of_available_frames_rawvideo(acquisition_path: str) -> int:
    gcv_path = acquisition_path + '.gcv'
    rawvideo_filename = [f for f in os.listdir(gcv_path) if f.endswith('.raw')][0]
    rawvideo_path = os.path.join(gcv_path, rawvideo_filename)
    if not(os.path.isfile(rawvideo_path)): raise(Exception(f'ERROR: Problem with the {acquisition_path} rawvideo file (it does not exist).'))
    f = open(rawvideo_path, "rb")
    f.seek(0, 2)
    file_size:int = f.tell()

    meta:Meta = retrieve_meta(acquisition_path)
    img_w:int = int(meta.get('subRegionWidth', '0'))
    img_h:int = int(meta.get('subRegionHeight', '0'))
    img_s:int = img_w * img_h
    n_frames_tot:int = file_size // img_s
    if img_s == 0 or file_size % img_s != 0:
        #Todo: WARNING here
        print('WARNING bad formatting')
    return n_frames_tot

def get_frames_rawvideo(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:
    meta = retrieve_meta(acquisition_path)

    width:int = int(meta.get('subRegionWidth', '0'))
    height:int = int(meta.get('subRegionHeight', '0'))
    length:int = len(framenumbers)

    bytes_per_image:int = width * height
    filesize:int = bytes_per_image * length
    if filesize > 10**9:
        pass
        #todo: warn here is filesize > 1 GB

    frames = np.empty([length, height, width], np.uint8)

    gcv_path = acquisition_path + '.gcv'
    rawvideo_filename = [f for f in os.listdir(gcv_path) if f.endswith('.raw')][0]
    rawvideo_path = os.path.join(gcv_path, rawvideo_filename)
    if not(os.path.isfile(rawvideo_path)): raise(Exception(f'ERROR: Problem with the {acquisition_path} rawvideo file (it does not exist).'))

    if length > 1 and (framenumbers[1:] - framenumbers[:-1]).max() == 1:
        # for some reason np.frombuffer is faster than np.fromfile if not in a loop... don't ask me why.
        with open(rawvideo_path, 'rb') as file:
            frames = np.frombuffer(file.read((framenumbers[-1]+1) * bytes_per_image),
                                   dtype = np.uint8,
                                   offset = framenumbers[0] * bytes_per_image,
                                   count = bytes_per_image * length).reshape((length, height, width))
    else:
        for i_framenumber, framenumber in enumerate(framenumbers):
            frames[i_framenumber] = np.fromfile(rawvideo_path,
                                                dtype=np.uint8,
                                                offset = framenumber * bytes_per_image ,
                                                count = bytes_per_image).reshape((height, width))

    return frames


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

def get_frames_t16(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:

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

def get_frames_t16_conservequality(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:

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


def get_frames_t8(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:

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

def get_acquisition_frequency_t16(acquisition_path: str, unit = None, verbose:int = 1) -> float:
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

def get_acquisition_duration_t16(acquisition_path: str, framenumbers:np.ndarray, unit = None, verbose:int = 1) -> float:
    acqu_freq: float = get_acquisition_frequency_t16(acquisition_path, unit =  'Hz', verbose = verbose)
    acqu_frames: float = len(framenumbers)

    return (acqu_frames - 1) / acqu_freq


###### LOSSLESSLY COMPRESSED VIDEO (lcv) READING

def find_available_flv(dataset_path: str) ->List[str]:
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

def get_frames_lcv(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:

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
        #Todo: ERROR here
        # print('ERROR: No MP4 video')
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

    if codec != 'avc1':
        video.release()
        #Todo:ERROR here
        print(f"Codec is {codec}. Cannot read frames from video that is not lossy compressed with codec avc1")
        return None

    return video

def get_number_of_available_frames_mp4(acquisition_path: str) -> Optional[int]:
    lcv = capture_mp4(acquisition_path)
    n_framenumbers_tot:int = int(lcv.get(cv2.CAP_PROP_FRAME_COUNT))
    lcv.release()
    return n_framenumbers_tot

def get_frames_mp4(acquisition_path:str, framenumbers:np.ndarray, verbose:int = 1) -> Optional[np.ndarray]:

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

def get_chronos_timestamps_from_mp4_video(**parameters) -> Optional[np.ndarray]:
    # Dataset selection
    dataset = parameters.get('dataset', 'unspecified-dataset')
    dataset_path = '../' + dataset
    if not(os.path.isdir(dataset_path)):
        print(f'ERROR: There is no dataset {dataset}.')
        return None

    # Acquisition selection
    acquisition = parameters.get('acquisition', 'unspecified-acquisition')
    acquisition_path = os.path.join(dataset_path, acquisition)
    if not (is_this_a_mp4(acquisition_path)):
        acquisition_path = acquisition_path.replace('tiff', 'mp4')
        if not (is_this_a_mp4(acquisition_path)):
            print(f'ERROR: There is no acquisition named {acquisition} for the dataset {dataset}.')
            return None

    # Parameters getting
    roi = parameters.get('roi', None)
    framenumbers = parameters.get('framenumbers', None)

    # Data fetching
    frames = get_frames(acquisition_path, framenumbers=framenumbers, subregion=roi)
    length, height, width = frames.shape

    raw_tstamps = frames[:, -40:, :]

    threshold_luminosity = 185

    tstamps = raw_tstamps[:, 6:29, :] > threshold_luminosity

    # Separate the letters
    offset = 9
    letter_width = 16
    n_letters = (tstamps.shape[2] - offset) // letter_width

    letters = np.empty((tstamps.shape[0], n_letters, tstamps.shape[1], letter_width))

    for i_frame in range(tstamps.shape[0]):
        for i_letter in range(n_letters):
            letters[i_frame][i_letter] = tstamps[i_frame, :,
                                         offset + i_letter * letter_width:offset + (i_letter + 1) * letter_width]

    # Check that the letters are well formatted
    residue = np.sum([np.sum([np.sum(letter) % 4 for letter in letter_sequence]) for letter_sequence in letters])
    if residue > 0:
        # todo error here
        print("ERROR: non zero residue")
        return None

    # Fetch the dict
    letters_dict_parameters = {'datatype': 'h264_timestamp_letters_dict'}
    try:
        letters_dict = datasaving.fetch_saved_data(letters_dict_parameters)
    except:
        # todo error here
        print("ERROR: no timestamp letters dict saved")
        return None

    placeholder_letter = 'X'

    # Transform to letters
    letters_formatted = np.full((letters.shape[0], letters.shape[1]), placeholder_letter, dtype=str)

    for i_key, key in enumerate(letters_dict.keys()):
        print(f'{i_key+1}/{len(letters_dict.keys())}\r', end='')
        for i_frame in range(tstamps.shape[0]):
            for i_letter in range(n_letters):
                if letters_formatted[i_frame][i_letter] == placeholder_letter and (letters[i_frame][i_letter] == letters_dict[key]).all():
                    letters_formatted[i_frame][i_letter] = key

    # Check that it works
    if placeholder_letter in letters_formatted:
        # todo error
        print("ERROR: Unrecognized letter")
        return None

    timestamps = [''.join(letters_sequence) for letters_sequence in letters_formatted]

    all_times = np.zeros(len(timestamps), dtype=float)
    all_framenumbers = np.zeros(len(timestamps), dtype=int)

    for i_timestamp, timestamp in enumerate(timestamps):
        framenumber = int(timestamp.split('/')[0]) - 1
        time = float(timestamp.split('T=')[1].split('s')[0])
        all_framenumbers[i_timestamp] = framenumber
        all_times[i_timestamp] = time

    print(f'Nb of framenumbers: {len(all_framenumbers)}')
    print(f'Max fnb: {all_framenumbers.max()}')
    print(f'Min fnb: {all_framenumbers.min()}')
    print(f'First fnb: {all_framenumbers[0]}')
    print(f'Last fnb: {all_framenumbers[-1]}')
    print(f'ones: {np.sum((all_framenumbers[1:]-all_framenumbers[:-1]) == 1)}')
    print(f'doublons: {np.sum((all_framenumbers[1:]-all_framenumbers[:-1]) == 0)}')
    print(f'timeshifts: {np.sum((all_framenumbers[1:]-all_framenumbers[:-1]) < 0)}')
    print(f'gaps: {np.sum((all_framenumbers[1:]-all_framenumbers[:-1]) > 1)}')

    are_ok = all_framenumbers >= all_framenumbers[0]
    all_framenumbers_ok = all_framenumbers[are_ok]
    all_times_ok = all_times[are_ok]

    t = np.zeros(len(all_times_ok), dtype=float)
    framenumbers = all_framenumbers_ok - all_framenumbers_ok.min()
    t[framenumbers] = all_times_ok

    return t

def get_chronos_acquisition_frequency_from_mp4_video(**parameters) -> Optional[float]:
    t = get_chronos_timestamps_from_mp4_video(**parameters)
    if t is None:
        return None
    else:
        if len(t) < 2:
            return None
        else:
            return float(1/np.mean(t[1:]-t[:-1]))



###### GENERAL VIDEO READING

def alert_me_if_there_is_no_video(acquisition_path: str) -> None:
    if not(is_this_a_video(acquisition_path)):
        for i in range(10):
            print(f'WARNING No video named {acquisition_path} exists !')

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
    elif is_this_a_lcv(acquisition_path):
        return True
    elif is_this_a_mp4(acquisition_path):
        return True
    else:
        throw_G2L_warning(f'There is no video at the acquisition {acquisition_path}')
        return False

def find_available_videos(dataset_path: str) ->List[str]:
    available_acquisitions = []
    available_acquisitions += find_available_gcv(dataset_path)
    available_acquisitions += find_available_t16(dataset_path)
    available_acquisitions += find_available_flv(dataset_path)
    available_acquisitions += find_available_mp4(dataset_path)
    available_acquisitions.sort()
    return available_acquisitions

def get_number_of_available_frames(acquisition_path: str) -> Optional[int]:
    if is_this_a_gcv(acquisition_path):
        return get_number_of_available_frames_gcv(acquisition_path)
    elif is_this_a_t16(acquisition_path):
        return get_number_of_available_frames_t16(acquisition_path)
    elif is_this_a_lcv(acquisition_path):
        return get_number_of_available_frames_lcv(acquisition_path)
    elif is_this_a_mp4(acquisition_path):
        return get_number_of_available_frames_mp4(acquisition_path)
    else:
        #Todo: ERROR HERE
        print('No video file.')
    return None

def get_acquisition_frequency(acquisition_path:str, unit = None, verbose:int = 1) -> float:
    if is_this_a_gcv(acquisition_path):
        return get_acquisition_frequency_gcv(acquisition_path, unit=unit, verbose=verbose)
    if is_this_a_t16(acquisition_path):
        return get_acquisition_frequency_t16(acquisition_path, unit=unit, verbose=verbose)
    else:
        #Todo: ERROR HERE
        print('No video file.')
        return -1.

def get_acquisition_duration(acquisition_path:str, framenumbers:Optional[np.ndarray], unit = None, verbose:int = 1) -> float:
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        # todo ERROR here
        print("ERROR Wrong framenumber, couldnt format it")
        return None

    if is_this_a_gcv(acquisition_path):
        return get_acquisition_duration_gcv(acquisition_path, framenumbers=framenumbers, unit=unit, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        return get_acquisition_duration_t16(acquisition_path, framenumbers=framenumbers, unit=unit, verbose=verbose)
    else:
        #Todo: ERROR HERE
        print('ERROR: Couldnt get acquisition duration.')
        return -1.

def format_framenumbers(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, verbose:int = 1) -> Optional[np.ndarray]:
    if framenumbers is None: # Default value for framenumbers
        framenumbers = np.arange(get_number_of_available_frames(acquisition_path))
    framenumbers = np.array(framenumbers)
    #Check that the demand is reasonable
    if framenumbers.min() < 0:
        #Todo: ERROR here
        print('Asked for negative framenumber ??')
        return None
    if framenumbers.max() >= get_number_of_available_frames(acquisition_path):
        #Todo: ERROR here
        print(f'Requested framenumber {framenumbers.max()} while there are only {get_number_of_available_frames(acquisition_path)} frames for this dataset.')
        return None
    return framenumbers

def get_geometry(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, subregion:Subregion = None, verbose:int = 1) -> Optional[Tuple]:
    formatted_fns = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if formatted_fns is None: return None

    length = formatted_fns.size

    frame = get_frame(acquisition_path, formatted_fns[0], subregion=subregion, verbose=verbose)

    if frame is None: return None

    height, width = frame.shape

    return length, height, width

def get_frames(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, subregion:Subregion = None, verbose:int = 1) -> Optional[np.ndarray]:
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        # todo ERROR here
        print("ERROR Wrong framenumber, couldnt format it")
        return None

    # Retrieve the frames
    frames = None
    if is_this_a_gcv(acquisition_path):
        frames = get_frames_gcv(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        frames = get_frames_t16(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_lcv(acquisition_path):
        frames = get_frames_lcv(acquisition_path, framenumbers, verbose=verbose)
    elif is_this_a_mp4(acquisition_path):
        frames = get_frames_mp4(acquisition_path, framenumbers, verbose=verbose)
    else:
        #Todo: ERROR HERE
        print('No video file.')
        return None

    # crop the subregion
    frames = crop_frames(frames, subregion=subregion)

    # # reverse the y direction
    # # so that we can visualize with plt.imshow and origin='lower'
    # # and still have the right lines numbers
    # if frames is not None:
    #     frames = frames[:, ::-1, :]

    return frames

def get_frame(acquisition_path:str, framenumber:int, subregion:Subregion = None, verbose:int = 1) -> Optional[np.ndarray]:
    return get_frames(acquisition_path, np.array([framenumber]), subregion=subregion, verbose=verbose)[0].astype(int)

def get_times(acquisition_path:str, framenumbers:Optional[np.ndarray] = None, unit = None, verbose:int=1) -> Optional[np.ndarray]:
    framenumbers = format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    if framenumbers is None:
        # todo ERROR here
        print("ERROR Wrong framenumber, couldnt format it")
        return None

    # Retrieve the frames
    times = None
    if is_this_a_gcv(acquisition_path):
        times = get_times_gcv(acquisition_path, framenumbers, unit=unit, verbose=verbose)
    elif is_this_a_t16(acquisition_path):
        times = (np.arange(framenumbers.max()+1) / get_acquisition_frequency_t16(acquisition_path))[framenumbers]
    else:
        #Todo: ERROR HERE
        print('No video for this dataset.')
        return None

    return times

def missing_frames(acquisition_path: str, verbose:int=1) -> List:
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

def missing_frames_in_framenumbers(acquisition_path: str, framenumbers:Optional[np.ndarray]=None, verbose:int=1) -> List:
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

def are_there_missing_frames(acquisition_path: str, framenumbers:Optional[np.ndarray]=None, verbose:int=1) -> bool:
    missing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=framenumbers, verbose=verbose)

    nbr_of_missing_chunks = len(missing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in missing_chunks])

    if  nbr_of_missing_chunks > 0:
        log_trace(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_trace(f'Missing chunks: {missing_chunks}', verbose=verbose)
        return True

    log_trace('No missing frames', verbose=verbose)
    return False


def describe(dataset:str, acquisition:str, framenumbers:Optional[np.ndarray]=None, subregion:Subregion=None, verbose:int=1):
    display(f'Acquisition: "{acquisition}" ({dataset})')

    dataset_path = os.path.join('../', dataset)
    acquisition_path = os.path.join(dataset_path, acquisition)
    if not(is_this_a_video(acquisition_path)):
        log_dbug(f'Videos in {dataset} are {find_available_videos(dataset_path)}', verbose=verbose)
        log_error(f'No video named {acquisition} in dataset {dataset}', verbose=verbose)

    # genberal
    frequency = get_acquisition_frequency(acquisition_path, unit="Hz", verbose=verbose)

    log_info(f'Acquisition frequency: {round(frequency, 2)} Hz', verbose=verbose)

    # raw video file
    maxlength, maxheight, maxwidth = get_geometry(acquisition_path, framenumbers=None, subregion=None)
    maxsize = maxlength*maxheight*maxwidth
    maxduration = get_acquisition_duration(acquisition_path, framenumbers=None, unit="s")
    maxmissing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=None, verbose=verbose)
    nbr_of_missing_chunks = len(maxmissing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in maxmissing_chunks])

    log_dbug(f'Acquisition information:', verbose=verbose)
    log_dbug(f'Frames dimension: {maxheight}x{maxwidth}', verbose=verbose)
    log_dbug(f'Length: {maxlength} frames ({round(maxduration, 2)} s - {round(maxsize/10**6, 0)} MB)', verbose=verbose)
    if  nbr_of_missing_chunks > 0:
        log_dbug(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_dbug(f'Missing chunks: {maxmissing_chunks}', verbose=verbose)
    else:
        log_dbug('No missing frames for this acquisition', verbose=verbose)

    # chosen data chunk
    length, height, width = get_geometry(acquisition_path, framenumbers = framenumbers, subregion=subregion)
    size = length*height*width
    duration = get_acquisition_duration(acquisition_path, framenumbers=framenumbers, unit="s")
    missing_chunks = missing_frames_in_framenumbers(acquisition_path, framenumbers=framenumbers, verbose=verbose)
    nbr_of_missing_chunks = len(missing_chunks)
    nbr_of_missing_frames = np.sum([len(chunk) for chunk in missing_chunks])

    log_info(f'Chosen data', verbose=verbose)
    log_info(f'Frames dimension: {height}x{width} ({round(size/10**3, 0)} kB each)', verbose=verbose)
    log_info(f'Length: {length} frames ({round(duration, 2)} s - {round(size/10**6, 0)} MB)', verbose=verbose)
    if  nbr_of_missing_chunks > 0:
        log_info(f'There are {nbr_of_missing_chunks} missing chunks ({nbr_of_missing_frames} frames total)', verbose=verbose)
        log_info(f'Missing chunks: {missing_chunks}', verbose=verbose)
    else:
        log_info('No missing frames in chosen framenumbers', verbose=verbose)

### FRAMES EDITING

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
            #Todo:Warning here
            print(f'Warning: did not find appropriate codec for filetype {filetype}.')
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

    display(f'Video {video_path} saved')

def save_acquisition_to_video(acquisition_path:str, do_timestamp:bool = True, fps:float = 25., filetype:str = 'mkv', codec:Optional[str] = None, resize_factor:int = 1):
    if not(is_this_a_gcv(acquisition_path)) and not(is_this_a_t16(acquisition_path)):
        # todo error ERROR here
        print(f'WAW')
        return

    frames = get_frames(acquisition_path, framenumbers = None, subregion=None)
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
    display(f'Saving all the gcv acquisition in the dataset: {dataset}')

    dataset_path = '../' + dataset

    available_acquisitions =  find_available_gcv(dataset_path)
    display(f'The following acquisition will be saved: {available_acquisitions}')

    for acquisition in available_acquisitions:
        acquisition_path = os.path.join(dataset_path, acquisition)

        if is_this_a_gcv(acquisition_path):
            save_acquisition_to_video(acquisition_path, do_timestamp=do_timestamp, fps=fps, filetype=filetype, codec=codec, resize_factor=resize_factor)
        else:
            throw_G2L_warning(f'GCV acquisition {acquisition} is found but does not exist ?')

from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import os # to navigate in the directories

from .. import utility

from . import are_there_missing_frames, format_framenumbers

Meta = Dict[str, str]
Stamps = Dict[str, np.ndarray]

###### GEVCAPTURE VIDEO (gcv) READING

def find_available_gcv(dataset_path:str) ->List[str]:
    available_acquisitions = [f[:-4] for f in os.listdir(dataset_path) if f.endswith('.gcv') and is_this_a_gcv(os.path.join(dataset_path, f[:-4]))]
    available_acquisitions.sort()
    return available_acquisitions

def is_this_a_gcv(acquisition_path:str) -> bool:
    """
    Checks if there is a GevCapture Video (GCV) for the given dataset.

    :param acquisition_path:
    :return:
    """
    utility.log_subtrace(f'func: is_this_a_gcv ({acquisition_path})')
    gcv_path = acquisition_path + '.gcv'
    # check that we indeed have a folder containing the right files
    if not os.path.isdir(gcv_path): return False
    # check that the folder contain the right number of files
    files = os.listdir(gcv_path)
    stampsfiles = [f for f in files if f.endswith('.stamps')]
    metafiles = [f for f in files if f.endswith('.meta')]
    # check that we have one stamp file
    if len(stampsfiles) != 1: return False
    if len(metafiles) != 1: return False
    if len(files) == 3:
        rawvideofiles = [f for f in files if f.endswith('.raw')]
        if len(rawvideofiles) != 1:
            utility.log_trace(f'Video {acquisition_path}: {3} files: stamps, meta but no rawvideo?')
            return False
    elif len(files) == 2:
        # utility.log_warning(f'Video {acquisition_path}: No rawvideo file!')
        utility.log_trace(f'Video {acquisition_path}: It is a GCV without a rawvideo file!')
        return True
    else:
        utility.log_trace(f'Video {acquisition_path}: {len(files)} files?! Too much to be a gcv.')
        return False
    return True

def get_number_of_available_frames_gcv(acquisition_path:str) -> int:
    n_frames_stamps = get_number_of_available_frames_stamps(acquisition_path)
    n_frames_rawvideo = get_number_of_available_frames_rawvideo(acquisition_path)

    if n_frames_stamps != n_frames_rawvideo:
        utility.log_warning(f'Video {acquisition_path}: The stamps file mentions {n_frames_stamps} frames while there are {n_frames_rawvideo} frames availables in the raw video file.')
    if n_frames_rawvideo > 0:
        return n_frames_rawvideo
    return n_frames_stamps

def get_acquisition_frequency_gcv(acquisition_path:str, unit = None) -> float:
    if unit is None: # default unit
        unit = 'Hz'
    factor:np.int64 = 1 # Multiplication factor for the time unit
    # raw unit is in ns
    if unit == 'Hz':
        factor  = 1
    else:
        utility.log_warning(f'Unrecognized frequency unit : {unit}')

    meta_info = retrieve_meta(acquisition_path)
    freq_meta = float(meta_info['captureFrequency']) # Hz

    if are_there_missing_frames(acquisition_path):
        return freq_meta

    full_stamps = retrieve_stamps(acquisition_path)
    camera_timestamps = get_camera_timestamps(full_stamps, unit='s')

    freq_camera_timestamps = 1/np.mean(camera_timestamps[1:]-camera_timestamps[:-1])

    # # These are too imprecise to be useful
    # computer_timestamps = get_computer_timestamps(full_stamps, unit='s')
    #
    # freq_computer_timestamps = 1/np.mean(computer_timestamps[1:]-computer_timestamps[:-1])

    return freq_camera_timestamps

def get_acquisition_duration_gcv(acquisition_path:str, framenumbers:np.ndarray, unit = None) -> float:
    full_stamps = retrieve_stamps(acquisition_path)
    camera_timestamps = get_camera_timestamps(full_stamps, unit=unit)

    camera_timestamps = camera_timestamps[framenumbers]

    return np.max(camera_timestamps) - np.min(camera_timestamps)

def get_times_gcv(acquisition_path:str, framenumbers:np.ndarray, unit=None) -> Optional[np.ndarray]:
    full_stamps = retrieve_stamps(acquisition_path)
    camera_timestamps = get_camera_timestamps(full_stamps, unit=unit)

    times = camera_timestamps[framenumbers]

    return times

def get_frames_gcv(acquisition_path:str, framenumbers:np.ndarray) -> Optional[np.ndarray]:
    return get_frames_rawvideo(acquisition_path, framenumbers=framenumbers)

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
                        utility.throw_G2L_warning(f'Ambiguity when parsing the {acquisition_path} meta file: is there a rogue "=" somewhere in it?.')
                    variable = line[:-1].split('=')[0]
                    value = '='.join( line[:-1].split('=')[1:] )
                    meta[variable] = value
                else:
                    utility.throw_G2L_warning(f'Could not parse correctly the {acquisition_path} meta file.')
    except:
        raise(Exception(f'ERROR: Problem with the {acquisition_path} meta file (probably it could not be opened).'))

    utility.log_subtrace(f'Video {acquisition_path}: Metafile content: meta={meta}')

    return meta

def get_frame_geometry_gcv(acquisition_path: str) -> Tuple[int, int]:
    utility.log_subtrace('func:get_frame_geometry_gcv')
    # returns the geometry from the metafile, in the format width, height
    meta:Meta = retrieve_meta(acquisition_path)

    usingROI = meta.get('usingROI', 'false') == 'true'
    camera_region_width  = int(meta.get('cameraRegionWidth', '0'))
    camera_region_height = int(meta.get('cameraRegionHeight', '0'))
    viewer_region_width  = int(meta.get('viewerRegionWidth', '0'))
    viewer_region_height = int(meta.get('viewerRegionHeight', '0'))

    legacy_width  = int(meta.get('subRegionWidth', '0'))
    legacy_height = int(meta.get('subRegionHeight', '0'))

    width:int  =  camera_region_width if not usingROI else viewer_region_width
    height:int =  camera_region_height if not usingROI else viewer_region_height
    if width  == 0: width = legacy_width
    if height == 0: height = legacy_height

    # OLD CODE
    # width:int = int(meta.get('subRegionWidth', '0'))
    # height:int = int(meta.get('subRegionHeight', '0'))

    utility.log_debug(f'Video {acquisition_path}: Retrieved frame geometry from metafile.')
    utility.log_trace(f'Video {acquisition_path}: Frame geometry: (w,h)=({width},{height})')

    return width, height

### STAMPS READING

def retrieve_stamps(acquisition_path:str) -> Stamps:
    """Gets all the stamps of a video
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
                    utility.throw_G2L_warning(f'Could not parse correctly the {acquisition_path} stamps file.')
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

def missing_framenumbers_gcv(acquisition_path: str) -> List:
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
    utility.log_trace(f'Missing frames for {acquisition_path}:')
    utility.log_trace(f'{all_missing_chunks}')
    return all_missing_chunks

def identify_missing_framenumbers(framenumbers:np.ndarray, verbose:Optional[int]=None) -> np.ndarray:
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

def get_camera_timestamps(stamps:Stamps, unit:str = None) -> np.ndarray:
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

def get_computer_timestamps(stamps:Stamps, unit:str='s') -> np.ndarray:
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

def get_monotonic_timebound(acquisition_path:str, framenumbers:Optional[np.ndarray]=None, unit=None) -> Tuple[float, float]:
    framenumbers = format_framenumbers(acquisition_path, framenumbers)

    # Retrieve the frames
    times = get_computer_timestamps(retrieve_stamps(acquisition_path), unit=unit)

    good_times:np.ndarray = times[framenumbers]

    return float(good_times[0]), float(good_times[-1])

def get_regularly_spaced_stamps(full_stamps:Stamps, start_framenumber:int = 0, end_framenumber:int = -1, interval:int  = 1, verbose:Optional[int]=None) -> Optional[Stamps]:
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
        utility.log_warning(f'The requested end frame ({end_framenumber}) is after the last recorded frame ({last_framenumber}).', verbose=verbose)
        utility.log_warning(f'Changing the end frame to {last_framenumber}. You might get less frames than expected.', verbose=verbose)
        end_framenumber = last_framenumber

    ### Check for missing frames
    missing_framenumbers = identify_missing_framenumbers(all_framenumbers, verbose=verbose)
    if not len(missing_framenumbers) == 0:
        utility.log_warning(f'There are missing frames ! Frames {list(missing_framenumbers)} are missing.', verbose=verbose)
    # here MF means missing frame (abbreviated for code lisibility)
    MFs_after_start = missing_framenumbers >= start_framenumber
    MFs_before_end = missing_framenumbers <= end_framenumber
    MFs_in_requested_interval = MFs_after_start * MFs_before_end
    there_are_MFs_in_the_request_interval:bool = 1 in MFs_in_requested_interval
    if there_are_MFs_in_the_request_interval:
        utility.log_warning(f'There are missing frames in the requested interval. The returned frames will not be evenly spaced.', verbose=verbose)

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
        utility.log_warning('STAMPS ARE NOT COHERENT WITH VIDEO', verbose=verbose)

    return stamps_wanted


## RAWVIDEO READING

def get_number_of_available_frames_rawvideo(acquisition_path: str) -> int:
    gcv_path = acquisition_path + '.gcv'

    rawvideofiles = [f for f in os.listdir(gcv_path) if f.endswith('.raw')]
    if len(rawvideofiles) == 0:
        # At this point, we were already warned a dozen times by is_this_a_gcv, so no need to spam the user
        # utility.log_warning(f'Video {acquisition_path}: No rawvideo frames available.')
        return 0
    elif len(rawvideofiles) > 2:
        utility.log_warning(f'Video {acquisition_path}: More than one rawvideo?!')
        return 0

    rawvideo_filename = rawvideofiles[0]
    rawvideo_path = os.path.join(gcv_path, rawvideo_filename)
    if not(os.path.isfile(rawvideo_path)):
        utility.log_warning(f'Video {acquisition_path}: No rawvideo file.')
        return 0
    f = open(rawvideo_path, "rb")
    f.seek(0, 2)
    file_size:int = f.tell()

    img_w, img_h = get_frame_geometry_gcv(acquisition_path)
    img_s:int = img_w * img_h
    n_frames_tot:int = file_size // img_s
    if img_s == 0 or file_size % img_s != 0:
        utility.log_warning(f'Video {acquisition_path}: Bad formatting of rawvideo file')
    return n_frames_tot

def get_frames_rawvideo(acquisition_path:str, framenumbers:np.ndarray) -> Optional[np.ndarray]:

    width, height = get_frame_geometry_gcv(acquisition_path)
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

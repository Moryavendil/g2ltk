from typing import Optional, Any, Tuple, Dict, List, Union
import numpy as np
import cv2 # to manipulate images and videos
import os # to navigate in the directories

from .. import display, throw_G2L_warning, log_error, log_warn, log_info, log_debug, log_trace
from .. import utility, datasaving

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
        log_debug(f'No mov video named {video_path}')
        return None

    # Open video
    video = cv2.VideoCapture(video_path)

    if video.isOpened()== False:
        log_error(f'Error opening video file {video_path}')
        return None

    #Check codec
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)

    if codec != 'avc1':
        video.release()
        log_error(f"Codec is {codec}. Cannot read frames from video that is not lossy compressed with codec avc1")
        return None

    return video

def get_number_of_available_frames_mp4(acquisition_path: str) -> Optional[int]:
    lcv = capture_mp4(acquisition_path)
    n_framenumbers_tot:int = int(lcv.get(cv2.CAP_PROP_FRAME_COUNT))
    lcv.release()
    return n_framenumbers_tot


def get_acquisition_frequency_mp4(acquisition_path, unit='Hz', verbose=1):
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

def get_frames_mp4(acquisition_path:str, framenumbers:np.ndarray, verbose:Optional[int]=None) -> Optional[np.ndarray]:

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
    frames = get_frames_mp4(acquisition_path, framenumbers=framenumbers, subregion=roi)
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


# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import numpy as np
import matplotlib.pyplot as plt

from g2ltk import set_verbose, datareading, utility
utility.configure_mpl()

import pandas as pd

from g2ltk import datareading


# <codecell>

def convert_tiff_to_gcv(acquisition_path, acquisition_frequency, exposure_time, framenumbers=None, subregion=None, verbose=1):
    acquisition_gcv = acquisition_path + '_gcv'

    gcv_path = acquisition_gcv + '.gcv'
    if os.path.isdir(gcv_path):
        utility.log_warn(f'FILE "{gcv_path}" ALREADY EXISTS. Aborting.')
        return
    os.makedirs(gcv_path)
    metafilepath = os.path.join(gcv_path, 'metainfo.meta')
    rawvideofilepath = os.path.join(gcv_path, 'rawvideo.raw')
    stampsfilepath = os.path.join(gcv_path, 'timestamps.stamps')

    ### META FILE

    # The dictonary that will contain the info
    metainfo = {}

    # retreive the TIFF metadata
    from PIL import Image
    from PIL.TiffTags import TAGS

    all_images = os.listdir(acquisition_path)
    all_images.sort()
    img_metaprobe = Image.open(os.path.join(acquisition_path, all_images[0]))
    tiff_meta_dict = {TAGS[key] : img_metaprobe.tag[key] for key in img_metaprobe.tag_v2}

    # convert the tiff metadata to our kind of metadata
    for key in tiff_meta_dict:
        metainfo[key] = str(tiff_meta_dict[key][0])

    # add our own info, formatted in our own style
    metainfo['usingROI'] = 'false'
    metainfo['subRegionX'] = '0'
    metainfo['subRegionY'] = '0'
    metainfo['subRegionWidth'] = metainfo['ImageWidth']
    metainfo['subRegionHeight'] = metainfo['ImageLength']
    metainfo['captureCameraName'] = metainfo['UniqueCameraModel']
    metainfo['captureFrequency'] = str(acquisition_frequency) # cahier de manip
    metainfo['captureExposureTime'] = str(exposure_time) # in us. RIGHT CLIC ON A .TIFF AND GO TO PROPERTIES -> IMAGE
    metainfo['captureProg'] = metainfo['UniqueCameraModel']

    # write that in the meta file
    with open(metafilepath, 'w') as metafile:
        for key in metainfo:
            metafile.write(key+'='+metainfo[key]+'\n')

    ### STAMPS FILE
    # make up for the stamps data
    framenumbers = datareading.format_framenumbers(acquisition_path, framenumbers, verbose=verbose)
    fn = framenumbers.astype(int)
    camera_time = np.rint(framenumbers / acquisition_frequency * 1e9).astype(np.int64) # mock camera time
    computer_time = np.rint(framenumbers / acquisition_frequency * 1e6).astype(np.int64) # mock computer time

    # write that in the meta file
    with open(stampsfilepath, 'w') as stampsfile:
        for i_pt in range(len(fn)):
            stampsfile.write(str(fn[i_pt])+'\t'+str(camera_time[i_pt])+'\t'+str(computer_time[i_pt])+'\n')

    ### DATA FILE

    # Data fetching
    frames = datareading.get_frames(acquisition_path, framenumbers=framenumbers, subregion=subregion)
    length, height, width = frames.shape

    #bytesToWrite = frames.flatten().tobytes()
    frames.flatten().tofile(rawvideofilepath)


# <codecell>

# Datasets display
root_path = '../'
datasets = datareading.find_available_datasets(root_path)
print('Available datasets:', datareading.find_available_datasets(root_path))


# <codecell>

# Dataset selection & acquisitions display
dataset = '-'
if len(datasets) == 1:
    dataset = datasets[0]
    datareading.log_info(f'Auto-selected dataset {dataset}')
dataset_path = os.path.join(root_path, dataset)
datareading.describe_dataset(dataset_path, type='t8', makeitshort=True)


# <codecell>

# # Acquisition selection
# acquisition = 'ha_n140f000a000'
# acquisition_path = os.path.join(dataset_path, acquisition)
# datareading.is_this_a_video(acquisition_path)


# <codecell>

ext = '.xlsx' # '.ods'
spreadsheets = [f[:-len(ext)] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(ext)]
spreadsheets.sort()
if len(spreadsheets) != 1:
    utility.log_warn(f'No spreadsheet or too many of them: {spreadsheets}')
else:
    spreadsheet = spreadsheets[0]
    infotable = pd.read_excel(os.path.join(dataset_path, spreadsheet + ext), skiprows=2)


# <codecell>

for acquisition in datareading.find_available_tiffs(dataset_path):

    acquisition_path = os.path.join(dataset_path, acquisition)
    datareading.is_this_a_video(acquisition_path)

    acquisition_frequency = (float(infotable['acquisition_frequency'][infotable['title'] == acquisition]))
    exposure_time = (float(infotable['exposure_time'][infotable['title'] == acquisition]))
    print(f'acquisition: {acquisition} | fps={acquisition_frequency} | texp={exposure_time}')

    # Parameters definition
    framenumbers = np.arange(datareading.get_number_of_available_frames(acquisition_path))
    roi = None, None, None, None  #start_x, start_y, end_x, end_y

    convert_tiff_to_gcv(acquisition_path, acquisition_frequency, exposure_time, framenumbers=framenumbers, subregion=roi)


# <markdowncell>

# ## TESTING
# We display the first, middle and last images of thegenerated gcv to check that eveything is OK


# <codecell>

### TEST NOW
acquisition_gcv = 'ha_n140f000a000' + '_gcv'

acquisition_path = os.path.join(dataset_path, acquisition_gcv)
datareading.is_this_a_video(acquisition_path)


# <codecell>

# Parameters definition
mxfn = datareading.get_number_of_available_frames(acquisition_path)
framenumbers = np.array([0, mxfn//2, mxfn-1])
roi = None, None, None, None  #start_x, start_y, end_x, end_y


# <codecell>

frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
length, height, width = frames.shape


# <codecell>

fig, axes = plt.subplots(3, 1)
axes[0].imshow(frames[0])
axes[1].imshow(frames[1])
axes[2].imshow(frames[2])


# <codecell>

    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    


    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    


    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    


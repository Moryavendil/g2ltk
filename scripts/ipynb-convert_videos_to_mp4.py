# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

import os

from tools import datareading


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
datareading.describe_dataset(dataset_path, type='gcv', makeitshort=True)


# <codecell>

datareading.save_all_gcv_videos(dataset, do_timestamp = True, fps = 20., filetype = 'mp4')


    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    


    # <markdowncell>
    
    # If you can read this, reads_py() is no longer broken! 
    


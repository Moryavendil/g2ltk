# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from g2ltk import datareading


# <codecell>

### Datasets display
datareading.set_default_root_path('../')
datareading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = datareading.find_dataset(None)
datareading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)


# <codecell>

datareading.save_all_gcv_videos(dataset=dataset, do_timestamp = True, fps = 20., filetype = 'mp4')


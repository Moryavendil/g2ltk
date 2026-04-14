# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

from g2ltk import videoreading


# <codecell>

### Datasets display
videoreading.set_default_root_path('../')
videoreading.describe_root_path()


# <codecell>

### Dataset selection & acquisitions display
dataset = videoreading.find_dataset(None)
videoreading.describe_dataset(dataset=dataset, videotype='gcv', makeitshort=True)


# <codecell>

videoreading.save_all_gcv_videos(dataset=dataset, do_timestamp = True, fps = 20., filetype = 'mp4')


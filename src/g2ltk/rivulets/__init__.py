from g2ltk._optional import require_extra
# rivulets uses both g2ltk.plotting and g2ltk.videoreading
require_extra('g2ltk.rivulets', 'full', {'matplotlib': 'matplotlib', 'IPython': 'ipython',
                                         'cv2': 'opencv-python-headless', 'PIL': 'pillow', 'pandas': 'pandas'})

# The verbosity of g2ltk can also be set from here (rivulets.set_verbose('debug')): it applies to all of g2ltk
from g2ltk.customlog import set_verbose, verbosity

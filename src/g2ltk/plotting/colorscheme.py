import matplotlib.colors as col
from matplotlib.colors import Normalize, LogNorm

# default settings
errorbar_kw_default = {'capsize':3, 'ls':''}
fill_between_kw_default = {'lw':0.0}


gray = '#808080'
red = '#ff0000'
blue = '#0000ff'
green = '#00ff00'
anglecmap = col.LinearSegmentedColormap.from_list('anglecmap', [green, blue, gray, red, green], N=256, gamma=1)
anglecmap_r = col.LinearSegmentedColormap.from_list('anglecmap_r', [green, blue, gray, red, green][::-1], N=256, gamma=1)
anglecmap_shifted = col.LinearSegmentedColormap.from_list('anglecmap_shifted', [gray, blue, green, red, gray], N=256, gamma=1)
anglecmap_shifted_r = col.LinearSegmentedColormap.from_list('anglecmap_shifted_r', [gray, blue, green, red, gray][::-1], N=256, gamma=1)

# rivulet colors
color_w = '#3d5da9'
color_w_rgb = (61, 93, 169)
cmap_w_dict = {'red':   [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[0]/255, color_w_rgb[0]/255]],
               'green': [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[1]/255, color_w_rgb[1]/255]],
               'blue':  [[0.0,  1.0, 1.0],
                         [1.0,  color_w_rgb[2]/255, color_w_rgb[2]/255]]}
cmap_w = col.LinearSegmentedColormap('white_to_w', segmentdata=cmap_w_dict, N=256)


color_z = '#ff1a1a'
color_z_rgb = (255, 26, 26)
cmap_z_dict = {'red':   [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[0]/255, color_z_rgb[0]/255]],
               'green': [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[1]/255, color_z_rgb[1]/255]],
               'blue':  [[0.0,  1.0, 1.0],
                         [1.0,  color_z_rgb[2]/255, color_z_rgb[2]/255]]}
cmap_z = col.LinearSegmentedColormap('white_to_z', segmentdata=cmap_z_dict, N=256)

color_q = '#9c1ab2' # (dark version : #9c1ab2 | light version : #c320df

# condition colors
color_smallQ = '#008C00'
color_bigQ = '#C29A49'

cmap_zonly = 'PuOr_r'
cmap_wonly = 'viridis_r'
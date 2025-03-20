

# default settings
errorbar_kw_default = {'capsize': 3, 'ls': '', 'marker':'o', 'mfc': 'w'}
fill_between_kw_default = {'alpha':.1, 'lw':0.0}


# rivulet colors
color_q = '#9c1ab2'
color_w = '#3d5da9'
color_z = '#ff1a1a' # (dark version : #9c1ab2 | light version : #c320df

def force_aspect_ratio(ax, aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

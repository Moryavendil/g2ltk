from typing import Optional, Tuple, Dict
import numpy as np
import os
# import cv2 # to manipulate images and videos
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit # to fit functions

from g2ltk import datareading, datasaving, utility
from g2ltk import display

# Custom typing
Meta = Dict[str, str]
Stamps = Dict[str, np.ndarray]
Subregion = Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]#  start_x, start_y, end_x, end_y

### RIVULET FINDING

"""
Default value for finding functions parameters
 - resize_factor        (int, 1 - 3):       resizing image pour une meilleure precision
 - remove_median_bckgnd (bool):             Remove the median image for all the frames. only use when the whole rivulet is moving. Should be unnecessary on clean videos
 - white_tolerance      (float, 0 - 255):   Difference between the channel white background and the black borders
 - rivulet_size_factor  (float, 1. - 5.):   How much wider is the rivulet compared to the size occupied by low luminosity extremapoints
 - std_factor           (float, 1. - 5.):   How much of the noise to remove
 - borders_min_distance (float, 1. - 100.):  The distance, in px / resize_factor, between two consecutive maximums in the function find_extrema used to find the borders
 - borders_prominence (float, 1. - 100.):  prominence of menisci, given to scipy's find_peaks
 - borders_width (float, 1. - 100.):  width of menisci in px, given to scipy's find_peaks
 - max_rivulet_width    (float, 1. - 1000.): Maximum authorized rivulet width, in pixels 
 - max_borders_luminosity_difference (float, 0 - 255): Maximum authorized luminosity difference between the rivulet borders
 - verbose (int, 0 - 5):                    Debug level
"""
default_kwargs = {

    'resize_factor': 2,
    ### median removing
    'remove_median_bckgnd': False, # removes the t- median background for each pixel.
    # This is recommended if there are no pixel where the rivulet is often, i.e. the rivulet moves a lot over a large span and is rarely straight
    'remove_median_bckgnd_zwise': False, # removes the t-median for each z-line, independently.
    # This is smarter because at a given x we rarely have more than half the pixel spanned by the rivulet so it is often useful.
    # it allows us to mitigate the effects of uneven lighting in the x direction
    'gaussian_blur_kernel_size': None,
    'white_tolerance': 70.,
    'rivulet_size_factor': 2.,
    'std_factor': 3.,
    'borders_min_distance': None,
    'borders_prominence': None,
    'borders_width': None,
    'max_rivulet_width': 20.,
    'max_borders_luminosity_difference': 50.,
    'verbose': None
}


# COS : Center of Shadow, center of mass of the shadows -> Rename BOS, Barycentre of Shadow
# BOL : Barycentre of Light, center of mass of the light zone
pass
### CENTER OF RIVULET FINDING
pass

### LINEWISE METHODS
# COS
def cos_linewise(x:np.ndarray, y:np.ndarray, **kwargs)-> float:
    """
    This function locates the rivulet by computing the center of mass of the shadow of the rivulet.

    :param x:
    :param y:
    :param kwargs: white_tolerance (whiteness of the rivulet, 0-256) ; rivulet_size_factor (width of the rivulet, 1.-5.)
    :return:
    """
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    # Step 1: get the roi i.e. the channel (white-ish zone)
    white_threshold = np.max(y) - kwargs['white_tolerance']
    is_white = y >= white_threshold

    left = x[np.argmax(is_white)]
    right = x[len(x) - np.argmax(is_white[::-1]) - 1]

    x_roi = x[(x > left) & (x < right)]
    y_roi = 255 - y[(x > left) & (x < right)]

    # gets the max intensity (approx rivulet centre) and estimate the width of the rivulet
    x_center = x_roi[np.argmax(y_roi)]
    y_max = y_roi[np.argmax(y_roi)]
    y_median = np.median(y_roi)
    y_threshold: float = (y_max + y_median) / 2

    approx_size = np.sum(y_roi >= y_threshold) * np.mean(x_roi[1:] - x_roi[:-1])

    # get the rivulet zone border
    x_left, x_right = x_center - kwargs['rivulet_size_factor'] * approx_size, x_center + kwargs['rivulet_size_factor'] * approx_size

    # get the zone around the rivulet
    criterion = (x_roi >= x_left) & (x_roi <= x_right)
    x_ponderate = x_roi[criterion]
    y_ponderate = y_roi[criterion]

    # put the smallest weight when no rivulet. HOW TO DETERMINE THE WEIGHTS ?
    # weights_offset = np.min(y_ponderate)
    weights_offset = np.median(y_roi)
    weights = np.maximum(y_ponderate - weights_offset, 0)

    # get the COM
    position = np.sum(x_ponderate * weights) / np.sum(weights)

    # pos1 = np.sum(x_ponderate * np.maximum(y_ponderate - np.min(y_ponderate), 0)) / np.sum(np.maximum(y_ponderate - np.min(y_ponderate), 0))
    # pos2 = np.sum(x_ponderate * np.maximum(y_ponderate - np.median(y_roi), 0)) / np.sum(np.maximum(y_ponderate - np.median(y_roi), 0))
    # print(f'DEBUG: deltapos: {np.abs(pos2-pos1)} px')

    return position

def borders_linewise(z_line:np.ndarray, l_line:np.ndarray, **kwargs):
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    ### Step 2: Obtain the shadow representation
    height, = l_line.shape

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l_line, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l_line >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=0).max()
    bot = height - np.argmax(is_channel[::-1], axis=0).min()

    # 2.4 Obtain the z and shadow image for the channel
    s_channel = 255 - l_line[top:bot] # shadowisity (255 - luminosity)
    z_channel = z_line[top:bot]           # z coordinate

    ### Step 3: Find the borders
    bmax = bimax_by_peakfinder(z_channel, s_channel,
                               distance=kwargs['borders_min_distance'],
                               width=kwargs['borders_width'],
                               prominence=kwargs['borders_prominence'])

    z1, y1, z2, y2 = bmax

    zdiff:float = np.abs(z1 - z2)
    space_ok = zdiff < kwargs['max_rivulet_width']
    utility.log_trace(f"space_ok (Dz < {kwargs['max_rivulet_width']}): {space_ok}")

    ydiff:float = np.abs(y1 - y2)
    ydiff_ok = ydiff < kwargs['max_borders_luminosity_difference']
    utility.log_trace(f"ydiff_ok (Ds < {kwargs['max_borders_luminosity_difference']}): {ydiff_ok}")

    if space_ok * ydiff_ok: # There are 2 peaks
        if z1 < z2:
            zinf, zsup = z1, z2
        else:
            zinf, zsup = z2, z1
    else: # Il y a qu'un seul max...
        if y1 > y2:
            zinf, zsup = z1, z1
        else:
            zinf, zsup = z2, z2
    return np.array([zinf, zsup])

def bol_linewise(z_line:np.ndarray, l_line:np.ndarray, borders_for_this_line=None, **kwargs)-> float:
    """
    This function locates the rivulet by computing the center of mass of the light part of the rivulet.

    :param z_line:
    :param l_line:
    :param kwargs: white_tolerance (whiteness of the rivulet, 0-256) ; rivulet_size_factor (width of the rivulet, 1.-5.)
    :return:
    """
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    ### Step 3: Find the borders
    if borders_for_this_line is None:
        borders_for_this_line = borders_linewise(z_line, l_line, **kwargs)

    ### Step 2: Obtain the shadow representation
    height, = l_line.shape

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l_line, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l_line >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=0).max()
    bot = height - np.argmax(is_channel[::-1], axis=0).min()

    # 2.4 Obtain the z and shadow image for the channel
    s_channel = 255 - l_line[top:bot] # shadowisity (255 - luminosity)
    z_channel = z_line[top:bot]           # z coordinate

    ### Step 4: Compute the BOL
    zbot, ztop = borders_for_this_line

    rivulet_zone = (z_channel >= zbot) * (z_channel <= ztop)

    if np.sum(rivulet_zone) == 0:
        return zbot

    z_rivulet_zone = z_channel[rivulet_zone]
    l_rivulet_zone = 255 - s_channel[rivulet_zone]

    # Center of mass
    weights_offset = np.min(l_rivulet_zone) - 1e-5
    weights = np.maximum(l_rivulet_zone - weights_offset, 0)
    position = np.sum(z_rivulet_zone * weights) / np.sum(weights)

    return position


### FRAMEWISE METHODS
def cos_framewise(frame:np.ndarray, **kwargs)->np.ndarray:
    """

    white_tolerance :
    the channel is deemed there if the light is superior to (max l)  - white_tolerance
    for example if the max value of light in image is 253 and white_tolerance = 70, everything where  l > (253-70) is deemed the channel

    kwargs
    resize_factor (resizing of the frame, 1-4) ;
    white_tolerance (whiteness of the channel, 0-256) ;
    rivulet_size_factor (width of the rivulet, 1.-5.)
    'remove_median_bckgnd_zwise': False, # removes the t-median for each z-line, independently.
    'gaussian_blur_kernel_size'
    # This is smarter because at a given x we rarely have more than half the pixel spanned by the rivulet so it is often powerful.

    Parameters
    ----------
    frame
    kwargs

    Returns
    -------

    """
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]


    ### Step 1: Alter the image
    # 1.1 : resize (if needed)
    l = datareading.resize_frame(frame, resize_factor=kwargs['resize_factor'])
    l_origin = l.copy()
    l_origin -= l_origin.min()

    # 1.2 remove z median
    if kwargs['remove_median_bckgnd_zwise']:
        l = l - np.median(l, axis=(0), keepdims=True)
    l_bckgnd_rmved = l.copy()
    l_bckgnd_rmved -= l_bckgnd_rmved.min()

    # 1.3 gaussian blur
    if kwargs['gaussian_blur_kernel_size'] is not None:
        sigma = kwargs['gaussian_blur_kernel_size']
        l = gaussian_filter(l, sigma=sigma)
    l_filtered = l.copy()
    l_filtered -= l_filtered.min()

    # 1.4 remove minimum
    l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255

    ### Step 2: Obtain the shadow representation
    height, width = l.shape

    # 2.1 build the z coordinate ('horizontal' in real life)
    x1D = np.arange(width)
    z1D = np.arange(height)
    z2D = np.repeat(z1D, width).reshape((height, width))

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=0).max()
    bot = height - np.argmax(is_channel[::-1], axis=0).min()

    # 2.4 Obtain the z and shadow image for the channel
    s_channel = 255 - l[top:bot, :] # shadowisity (255 - luminosity)
    z_channel = z2D[top:bot, :]           # z coordinate

    ### Step 3: Obtain the border of the rivulet
    # 3.1 get a threshold to know where is the reivulet
    s_channel_max = np.amax(s_channel, axis=0, keepdims=True)
    s_channel_median = np.median(s_channel, axis=0, keepdims=True)
    # The threshold above which we count the rivulet
    # This should be ap[prox. the half-max of the rivulet shadow
    s_channel_threshold = (s_channel_max + s_channel_median) / 2

    # 3.2 compute the upper-approximate the half-width of the rivulet
    approx_rivulet_size = np.sum(s_channel >= s_channel_threshold, axis=0) * kwargs['rivulet_size_factor']

    # 3.3 the approximate position (resolution = size of the rivulet, a minima 1 pixel)
    riv_pos_approx = np.argmax(s_channel, axis=0, keepdims=True) + z_channel[0, :]

    # 3.4 identify the zone around the rivulet
    z_top = np.maximum(riv_pos_approx - approx_rivulet_size, np.zeros_like(riv_pos_approx))
    z_bot = np.minimum(riv_pos_approx + approx_rivulet_size, s_channel.shape[0] * np.ones_like(riv_pos_approx))
    around_the_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

    ### Step 4: compute the center fo mass
    # the background near the rivulet
    s_bckgnd_near_rivulet = np.amin(s_channel, axis=0, where=around_the_rivulet, initial=255, keepdims=True) * (1-1e-5)

    # the weights to compute the COM
    weights = (s_channel - s_bckgnd_near_rivulet) * around_the_rivulet

    # The COM rivulet with sub-pixel resolution
    rivulet = np.sum(z_channel * weights, axis=0) / np.sum(weights, axis=0)

    # take into account the resizing
    rivulet /= kwargs['resize_factor']

    return rivulet

def borders_framewise(frame:np.ndarray, prominence:float = 1, do_fit:bool=False, **kwargs) -> np.ndarray:
    """

    kwargs resize_factor (resizing of the frame, 1-4) ; max_rivulet_width (maximum authorized rivulet with, in pixels, 1-100)  ; max_borders_luminosity_difference (maximum authorized luminosity difference between the rivulet borders, 0-255)


    Parameters
    ----------
    frame
    prominence
    do_fit
    kwargs

    Returns
    -------

    """
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    ### Step 1: Alter the image
    # 1.1 : resize (if needed)
    l = datareading.resize_frame(frame, resize_factor=kwargs['resize_factor'])
    l_origin = l.copy()
    l_origin -= l_origin.min()

    # 1.2 remove z median
    if kwargs['remove_median_bckgnd_zwise']:
        l = l - np.median(l, axis=(0), keepdims=True)
    l_bckgnd_rmved = l.copy()
    l_bckgnd_rmved -= l_bckgnd_rmved.min()

    # 1.3 gaussian blur
    if kwargs['gaussian_blur_kernel_size'] is not None:
        sigma = kwargs['gaussian_blur_kernel_size']
        l = gaussian_filter(l, sigma=sigma)
    l_filtered = l.copy()
    l_filtered -= l_filtered.min()

    # 1.4 remove minimum
    l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255

    ### Step 2: Obtain the shadow representation
    height, width = l.shape

    # 2.1 build the z coordinate ('horizontal' in real life)
    x1D = np.arange(width) / kwargs['resize_factor']
    z1D = np.arange(height) / kwargs['resize_factor']
    z2D = np.repeat(z1D, width).reshape((height, width))

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=0).max()
    bot = height - np.argmax(is_channel[::-1], axis=0).min()

    # 2.4 Obtain the z and shadow image for the channel
    s_channel = 255 - l[top:bot, :] # shadowisity (255 - luminosity)
    z_channel = z2D[top:bot, :]           # z coordinate

    ### Step 3 : find borders
    bmax:np.ndarray = np.zeros((width, 4), dtype=float)

    for i_l in range(width):
        try:
            if do_fit:
                bmax[i_l] = bimax_fit_by_peakfinder(z_channel[:, i_l], s_channel[:, i_l],
                                                    distance=kwargs['borders_min_distance'],
                                                    width=kwargs['borders_width'],
                                                    prominence=kwargs['borders_prominence'])
            else:
                bmax[i_l] = bimax_by_peakfinder(z_channel[:, i_l], s_channel[:, i_l],
                                                distance=kwargs['borders_min_distance'],
                                                width=kwargs['borders_width'],
                                                prominence=kwargs['borders_prominence'])
        except:
            utility.log_error(f'Error in bimax line {i_l}')

    z1, y1, z2, y2 = bmax[:,0], bmax[:,1], bmax[:,2], bmax[:,3]
    x1, x2 = x1D.copy(), x1D.copy()

    # remove too spaced away
    zdiff:np.ndarray = np.abs(z1 - z2)
    space_ok = zdiff < kwargs['max_rivulet_width']
    if (1 - space_ok).sum() > 0:
        utility.log_debug(f'Too spaced away (> {kwargs["max_rivulet_width"]} resized px): {(1 - space_ok).sum()} pts', verbose=kwargs['verbose'])

    # remove too different peaks
    ydiff:np.ndarray = np.abs(y1 - y2)
    ydiff_ok = ydiff < kwargs['max_borders_luminosity_difference']
    if (1 - ydiff_ok).sum() > 0:
        utility.log_debug(f'Too different (> {kwargs["max_borders_luminosity_difference"]} lum): {(1 - ydiff_ok).sum()} pts', verbose=kwargs['verbose'])

    # There are 2 peaks
    deuxmax = space_ok * ydiff_ok

    # si il y a qu'un seul max...
    unmax = np.bitwise_not(deuxmax)
    # On garde le plus grand
    desacord = y1 > y2
    ndesacord = np.bitwise_not(desacord)

    zsup = np.concatenate((np.maximum(z1[deuxmax], z2[deuxmax]), z1[unmax * desacord], z2[unmax * ndesacord]))
    x_zsup = np.concatenate((np.maximum(x1[deuxmax], x2[deuxmax]), x1[unmax * desacord], x2[unmax * ndesacord]))
    suprightorder = x_zsup.argsort()
    x_zsup, zsup = x_zsup[suprightorder], zsup[suprightorder]

    zinf = np.concatenate((np.minimum(z1[deuxmax], z2[deuxmax]), z1[unmax * desacord], z2[unmax * ndesacord]))
    x_zinf = np.concatenate((np.minimum(x1[deuxmax], x2[deuxmax]), x1[unmax * desacord], x2[unmax * ndesacord]))
    infrightorder = x_zinf.argsort()
    x_zinf, zinf = x_zinf[infrightorder], zinf[infrightorder]

    return np.array([zinf, zsup])

def bol_framewise(frame:np.ndarray, borders_for_this_frame = None, **kwargs)-> np.ndarray:
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    ### Step 3 : find borders
    if borders_for_this_frame is None:
        borders_for_this_frame:np.ndarray = borders_framewise(frame, **kwargs)

    ### Step 1: Alter the image
    # 1.1 : resize (if needed)
    l = datareading.resize_frame(frame, resize_factor=kwargs['resize_factor'])
    l_origin = l.copy()
    l_origin -= l_origin.min()

    # 1.2 remove z median
    if kwargs['remove_median_bckgnd_zwise']:
        l = l - np.median(l, axis=(0), keepdims=True)
    l_bckgnd_rmved = l.copy()
    l_bckgnd_rmved -= l_bckgnd_rmved.min()

    # 1.3 gaussian blur
    if kwargs['gaussian_blur_kernel_size'] is not None:
        sigma = kwargs['gaussian_blur_kernel_size']
        l = gaussian_filter(l, sigma=sigma)
    l_filtered = l.copy()
    l_filtered -= l_filtered.min()

    # 1.4 remove minimum
    l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255

    ### Step 2: Obtain the shadow representation
    height, width = l.shape

    # 2.1 build the z coordinate ('horizontal' in real life)
    x1D = np.arange(width) / kwargs['resize_factor']
    z1D = np.arange(height) / kwargs['resize_factor']
    z2D = np.repeat(z1D, width).reshape((height, width))

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l, 95, axis=0, keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=0).max()
    bot = height - np.argmax(is_channel[::-1], axis=0).min()

    # 2.4 Obtain the z and shadow image for the channel
    l_channel = l[top:bot, :] # luminosity
    z_channel = z2D[top:bot, :]           # z coordinate

    ### Step 4: Compute the BOL
    # the zone inside the rivulet
    z_top = borders_for_this_frame[0,:]
    z_bot = borders_for_this_frame[1,:]
    inside_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

    bckgnd_inside_rivulet = np.amin(l_channel, axis=0, where=inside_rivulet, initial=255, keepdims=True) - 1e-4

    # the weights to compute the COM
    weights = (l_channel - bckgnd_inside_rivulet) * inside_rivulet

    # The BOL rivulet with sub-pixel resolution
    # this handles the tricky size of 0-width rivulet (when the two borders are at the same point, it happens for some shitty videos
    nonzerosum = np.sum(weights, axis=0) > 0

    rivulet = np.empty(width, dtype=float)
    rivulet[nonzerosum] = np.sum(z_channel * weights, axis=0)[nonzerosum] / np.sum(weights, axis=0)[nonzerosum]
    rivulet[np.bitwise_not(nonzerosum)] = ((z_bot+z_top)/2)[np.bitwise_not(nonzerosum)]

    return rivulet

###

### VIDEOWISE METHODS

def cos_videowise(frames:np.ndarray, **kwargs)->np.ndarray: # WORK IN PROGRESS
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    ### Step 1: Alter the image
    # 1.1 : resize (if needed)
    l = datareading.resize_frames(frames, resize_factor=kwargs['resize_factor'])
    l_origin = l.copy() - l.min() # DEBUG

    # 1.2 remove z median
    if kwargs['remove_median_bckgnd_zwise']:
        l = l - np.median(l, axis=(0, 1), keepdims=True)
    l_bckgnd_rmved = l.copy() - l.min() # DEBUG

    # 1.3 gaussian blur
    if kwargs['gaussian_blur_kernel_size'] is not None:
        sigma = kwargs['gaussian_blur_kernel_size']
        for i_t in range(len(l)):
            l[i_t] = gaussian_filter(l[i_t], sigma=sigma)
    l_filtered = l.copy() - l.min() # DEBUG

    # 1.4 remove minimum
    l -= l.min() # so that the min is at 0, i.e. the max of shadow is a 255

    ### Step 2: Obtain the shadow representation
    length, height, width = l.shape

    # 2.1 build the z coordinate ('horizontal' in real life)
    x1D = np.arange(width)
    z1D = np.arange(height)
    z2D = np.repeat(z1D, width).reshape((height, width))
    z3D = np.repeat(z2D[None, :, :], length, axis=0)

    # 2.2 discriminate the channel (white zone in the image)
    max_l = np.percentile(l, 95, axis=(0, 1), keepdims=True) # the max luminosity (except outliers). we take the 95th percentile
    threshold_l = max_l - kwargs['white_tolerance']
    is_channel = l >= threshold_l

    # 2.3 identify the channel borders
    top = np.argmax(is_channel, axis=1).max()
    bot = height - np.argmax(is_channel[::-1], axis=1).min()

    # 2.4 Obtain the z and shadow image for the channel
    s_channel = 255 - l[:, top:bot, :] # shadowisity (255 - luminosity)
    z_channel = z3D[:, top:bot, :]           # z coordinate

    ### Step 3: Obtain the border of the rivulet
    # 3.1 get a threshold to know where is the reivulet
    s_channel_max = np.amax(s_channel, axis=(1), keepdims=True)
    s_channel_median = np.median(s_channel, axis=(1), keepdims=True)
    # The threshold above which we count the rivulet
    # This should be ap[prox. the half-max of the rivulet shadow
    s_channel_threshold = (s_channel_max + s_channel_median) / 2

    # 3.2 compute the upper-approximate the half-width of the rivulet
    approx_rivulet_size = np.sum(s_channel >= s_channel_threshold, axis=1, keepdims=True) * kwargs['rivulet_size_factor']

    # 3.3 the approximate position (resolution = size of the rivulet, a minima 1 pixel)
    riv_pos_approx = np.argmax(s_channel, axis=1, keepdims=True) + z_channel[:, 0:1:, :]

    # utility.log_info(f's {s_channel.shape} --(argmax)-> {np.argmax(s_channel, axis=1, keepdims=True).shape}')
    # utility.log_info(f'+ z {z_channel[:, 0:1:, :].shape}')
    # utility.log_info(f'= riv_pos_approx {riv_pos_approx.shape}')

    # 3.4 identify the zone around the rivulet
    z_top = np.maximum(riv_pos_approx - approx_rivulet_size, np.zeros_like(riv_pos_approx))
    z_bot = np.minimum(riv_pos_approx + approx_rivulet_size, s_channel.shape[0] * np.ones_like(riv_pos_approx))
    around_the_rivulet = (z_channel >= z_top) & (z_channel <= z_bot)

    # utility.log_info(f'riv_pos_approx {riv_pos_approx.shape} + approx_rivulet_size {approx_rivulet_size.shape}')
    # utility.log_info(f'? np.zeros_like(riv_pos_approx) {np.zeros_like(riv_pos_approx).shape}')
    # utility.log_info(f'= z_top {z_top.shape}')


    ### Step 4: compute the center fo mass
    # 4.1 remove the background near the rivulet
    s_bckgnd_near_rivulet = np.amin(s_channel, axis=1, where=around_the_rivulet, initial=255, keepdims=True) * (1-1e-5)

    # 4.2 compute the weights to compute the COM
    weights = (s_channel - s_bckgnd_near_rivulet) * around_the_rivulet

    # 4.3 The COM rivulet with sub-pixel resolution
    rivulet = np.sum(z_channel * weights, axis=1) / np.sum(weights, axis=1)

    # take into account the resizing
    rivulet /= kwargs['resize_factor']

    return rivulet

### GLOBAL METHOD

# def find_mbp(**parameters):
#     dataset = parameters.get('dataset', 'unspecified-dataset')
#     acquisition = parameters.get('acquisition', 'unspecified-acquisition')
#     fetch_or_generate_data(**parameters):
#     borders =

### TOP - BOTTOM OF RIVULET FINDING

### LINEWISE METHODS

def bimax_naive(x, y):
    # check that there is float: important for use with find peaks
    # position of the maxs
    xmax = utility.find_extrema(x, y.astype(float, copy=False), peak_category='max')
    ymax = np.interp(xmax, x, y)

    if len(xmax) == 0:
        return 0, 0, 0, 0
    elif len(xmax) == 1:
        return xmax[0], ymax[0], xmax[0], ymax[0]

    # take the 2 bigger maxs
    maxs_sorted = ymax.argsort()
    x1, x2 = xmax[maxs_sorted][-1], xmax[maxs_sorted][-2]

    # y of the 2 bigger maxs
    y1, y2 = ymax[maxs_sorted][-1], ymax[maxs_sorted][-2]

    return x1, y1, x2, y2

def bimax_supernaive(x, y, **kwargs):
    # check that there is float: important for use with find peaks
    # position of the maxs
    xmax = x[find_peaks(y.astype(float, copy=False), distance=kwargs.get('distance', None))[0]]
    ymax = np.interp(xmax, x, y)

    if len(xmax) == 0:
        return 0, 0, 0, 0
    elif len(xmax) == 1:
        return xmax[0], ymax[0], xmax[0], ymax[0]

    # take the 2 bigger maxs
    maxs_sorted = ymax.argsort()
    x1, x2 = xmax[maxs_sorted][-1], xmax[maxs_sorted][-2]

    # y of the 2 bigger maxs
    y1, y2 = ymax[maxs_sorted][-1], ymax[maxs_sorted][-2]

    return x1, y1, x2, y2

def bimax_fit(x, y, w0:float = 1.):
    # INITIAL guess
    # position and y of the maxs
    x10, y1, x20, y2 = bimax_naive(x, y)

    # amplitude of the maxs
    deltax = x10-x20
    g = utility.gaussian_unnormalized(deltax, 0, w0)
    # We have
    # y1 =   a10 + g a20
    # y2 = g a10 +   a20
    # So we invert the matrix
    a10 = (y1 - g*y2)/(1-g**2)
    a20 = (y2 - g*y1)/(1-g**2)

    # noise
    bckgnd_noise0:float = max(0., y.min())

    p0 = (x10, a10, w0, x20, a20, w0, bckgnd_noise0)
    lbounds = (x.min(), 0., 0., x.min(), 0., 0., 0.)
    ubounds = (x.max(), 255, x.max()-x.min(), x.max(), 255, x.max()-x.min(), 255)
    bounds = (lbounds, ubounds)

    popt, pcov = curve_fit(utility.double_gauss, x, y, p0=p0, bounds=bounds)

    return popt

def bimax(x, y, do_fit:bool = False, w0:float = 1., **kwargs):
    if do_fit:
        return bimax_fit(x, y, w0)
    return bimax_supernaive(x, y, **kwargs)
    # return bimax_naive(x, y)

from scipy.signal import find_peaks

def bimax_by_peakfinder(z, y, distance:Optional[float]=None, width:Optional[float]=None, prominence:Optional[float]=None):
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence, width=width)

    # take the 2 bigger maxs
    maxs_sorted = y[peaks].argsort()
    x1, x2 = z[peaks][maxs_sorted][-1], z[peaks][maxs_sorted][-2]

    # y of the 2 bigger maxs
    y1, y2 = y[peaks][maxs_sorted][-1], y[peaks][maxs_sorted][-2]

    return x1, y1, x2, y2

def bimax_fit_by_peakfinder(z, y, distance:Optional[float]=None, width:Optional[float]=None, prominence:Optional[float]=None):
    x10, y10, x20, y20 = bimax_by_peakfinder(z, y, distance=distance, width=width, prominence=prominence)

    w0 = np.abs(x20-x10)/4

    # amplitude of the maxs
    deltax = x10-x20
    g = utility.gaussian_unnormalized(deltax, 0, w0)
    # We have
    # y1 =   a10 + g a20
    # y2 = g a10 +   a20
    # So we invert the matrix
    a10 = (y10 - g*y20)/(1-g**2)
    a20 = (y20 - g*y10)/(1-g**2)

    # noise
    bckgnd_noise0:float = max(0., y.min())

    p0 = (x10, a10, w0, x20, a20, w0, bckgnd_noise0)
    lbounds = (z.min(), 0., 0., z.min(), 0., 0., 0.)
    ubounds = (z.max(), 255, z.max()-z.min(), z.max(), 255, z.max()-z.min(), 255)
    bounds = (lbounds, ubounds)

    popt, pcov = curve_fit(utility.double_gauss, z, y, p0=p0, bounds=bounds)

    x1, x2 = popt[0], popt[3]
    y1, y2 = utility.double_gauss(x1, *popt), utility.double_gauss(x2, *popt)

    return x1, y1, x2, y2

def borders(frame:np.ndarray, do_fit:bool = False, w0:float = 1., **kwargs) -> np.ndarray:
    """

    :param frame:
    :param do_fit:
    :param w0:
    :param kwargs: resize_factor (resizing of the frame, 1-4) ; max_rivulet_width (maximum authorized rivulet with, in pixels, 1-100)  ; max_borders_luminosity_difference (maximum authorized luminosity difference between the rivulet borders, 0-255)
    :return:
    """
    for key in default_kwargs.keys():
        if not key in kwargs.keys():
            kwargs[key] = default_kwargs[key]

    frame_resized = datareading.resize_frame(frame, resize_factor=kwargs['resize_factor'])
    height, width = frame_resized.shape

    zz = np.zeros((width, 4), dtype=float)

    z = np.arange(height) / kwargs['resize_factor']

    for l in range(width):

        y = 255 - frame_resized[:, l].astype(float)

        zz[l] = bimax(z, y, do_fit=do_fit, w0=w0, **kwargs)

    z1, y1, z2, y2 = zz[:,0], zz[:,1], zz[:,2], zz[:,3]

    x = np.linspace(0, width / kwargs['resize_factor'], width, endpoint=False)

    x1, x2 = x.copy(), x.copy()

    # remove too spaced away
    zdiff = np.abs(z1 - z2)
    space_ok = zdiff < kwargs['max_rivulet_width']

    # remove too different peaks
    ydiff = np.abs(y1 - y2)
    ydiff_ok = ydiff < kwargs['max_borders_luminosity_difference']

    # There are 2 peaks
    deuxmax = space_ok * ydiff_ok

    # si il y a qu'un seul max...
    unmax = np.bitwise_not(deuxmax)
    # On garde le plus grand
    desacord = y1 > y2
    ndesacord = np.bitwise_not(desacord)

    zsup = np.concatenate((np.maximum(z1[deuxmax], z2[deuxmax]), z1[unmax * desacord], z2[unmax * ndesacord]))
    x_zsup = np.concatenate((np.maximum(x1[deuxmax], x2[deuxmax]), x1[unmax * desacord], x2[unmax * ndesacord]))
    suprightorder = x_zsup.argsort()
    x_zsup, zsup = x_zsup[suprightorder], zsup[suprightorder]

    zinf = np.concatenate((np.minimum(z1[deuxmax], z2[deuxmax]), z1[unmax * desacord], z2[unmax * ndesacord]))
    x_zinf = np.concatenate((np.minimum(x1[deuxmax], x2[deuxmax]), x1[unmax * desacord], x2[unmax * ndesacord]))
    infrightorder = x_zinf.argsort()
    x_zinf, zinf = x_zinf[infrightorder], zinf[infrightorder]

    return np.array([zinf, zsup])

### GLOBAL METHOD

def get_acquisition_path_from_parameters(**parameters) -> str:
    # Dataset selection
    dataset = parameters.get('dataset', 'unspecified-dataset')
    dataset_path = '../' + dataset
    if not(os.path.isdir(dataset_path)):
        print(f'WARNING (RVFD): There is no dataset named {dataset}.')

    # Acquisition selection
    acquisition = parameters.get('acquisition', 'unspecified-acquisition')
    acquisition_path = os.path.join(dataset_path, acquisition)
    if not(datareading.is_this_a_video(acquisition_path)):
        print(f'WARNING (RVFD): There is no acquisition named {acquisition} for the dataset {dataset}.')

    return acquisition_path

def get_frames_from_parameters(**parameters):
    acquisition_path = get_acquisition_path_from_parameters(**parameters)

    # Parameters getting
    roi = parameters.get('roi', None)
    framenumbers = parameters.get('framenumbers', None)

    # Data fetching
    frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
    # length, height, width = frames.shape

    frames = frames.astype(float, copy=False)

    if parameters.get('remove_median_bckgnd', default_kwargs['remove_median_bckgnd']):
        frames = frames - np.median(frames, axis=0, keepdims=True)
        frames -= frames.min()

    return frames

def find_borders(**parameters):
    # Get the frames
    frames = get_frames_from_parameters(**parameters)
    length, height, width = frames.shape

    for key in default_kwargs.keys():
        if not key in parameters.keys():
            parameters[key] = default_kwargs[key]

    brds = np.zeros((length, 2, width * parameters['resize_factor']), float)

    np.seterr(all='raise')
    for framenumber in range(length):
        try:
            brds[framenumber] = borders_framewise(frames[framenumber], **parameters)
        except:
            print(f'Error frame {framenumber}')
        if (framenumber+1)%5 == 0:
            display(f'Borders finding ({round(100*(framenumber+1)/length, 2)} %)', end = '\r')
    display(f'', end = '\r')
    utility.log_debug(f'Borders found', verbose=parameters['verbose'])

    return brds

def find_cos(**parameters):
    # Dataset selection
    dataset = parameters.get('dataset', 'unspecified-dataset')
    dataset_path = '../' + dataset
    if not(os.path.isdir(dataset_path)):
        utility.log_warning(f'(RVFD): There is no dataset named {dataset}.')

    # Acquisition selection
    acquisition = parameters.get('acquisition', 'unspecified-acquisition')
    acquisition_path = os.path.join(dataset_path, acquisition)
    if not(datareading.is_this_a_video(acquisition_path)):
        utility.log_warning(f'(RVFD): There is no acquisition named {acquisition} for the dataset {dataset}.')

    # Parameters getting
    roi = parameters.get('roi', None)
    framenumbers = parameters.get('framenumbers', None)

    # Data fetching
    frames = datareading.get_frames(acquisition_path, framenumbers = framenumbers, subregion=roi)
    length, height, width = frames.shape

    global default_kwargs
    if parameters.get('remove_median_bckgnd', default_kwargs['remove_median_bckgnd']):
        frames = frames - np.median(frames, axis=0, keepdims=True)

    for key in default_kwargs.keys():
        if not key in parameters.keys():
            parameters[key] = default_kwargs[key]

    rivs = np.zeros((length, width * parameters['resize_factor']), dtype=float)

    np.seterr(all='raise') # we are extra-careful because we do not want trash data
    try:
        # we first try the faster videowise
        rivs = cos_videowise(frames, **parameters)
    except:
        utility.log_error('cos_videowise failed. trying framewise to identify the problematic frame.')
        for framenumber in range(length):
            try:
                rivs[framenumber] = cos_framewise(frames[framenumber], **parameters)
            except:
                utility.log_error(f'Error doint cos_framewise on frame {framenumber}')

    return rivs

def find_bol(verbose:int = 1, **parameters):
    # First we need the borders
    borders_for_this_video = datasaving.fetch_or_generate_data_from_parameters('borders', parameters, verbose=verbose)

    # Then the frames
    frames = get_frames_from_parameters(**parameters)
    length, height, width = frames.shape

    for key in default_kwargs.keys():
        if not key in parameters.keys():
            parameters[key] = default_kwargs[key]

    rivs = np.zeros((length, width * parameters['resize_factor']), float)

    np.seterr(all='raise')
    for framenumber in range(length):
        try:
            rivs[framenumber] = bol_framewise(frames[framenumber], borders_for_this_frame=borders_for_this_video[framenumber], **parameters)
        except:
            print(f'Error frame {framenumber}')
        if (framenumbe+1)%5 == 0:
            display(f'BOL finding ({round(100*(framenumber+1)/length,2)} %)', end='\r')
    display(f'', end = '\r')
    utility.log_debug(f'BOL computed', verbose=parameters['verbose'])

    return rivs


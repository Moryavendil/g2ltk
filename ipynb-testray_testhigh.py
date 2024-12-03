# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>

%matplotlib notebook

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import logfacility as lg


# <codecell>

lg.set_verbose(2)


# <codecell>

class Halfcircle():
    def __init__(self, xc=None, yc=None, R=None, topbot=None):
        self.xc = xc
        self.yc = yc
        self.R = R
        if xc is None or yc is None or R is None:
            raise Exception('Invalid circle')
        self.topbot = topbot

    def y(self, x):
        if isinstance(x, np.ndarray):
            if ((x - self.xc)**2 > self.R**2).any():
                print(f'Out of circle (x)')
                return None
        else:
            if (x - self.xc)**2 > self.R**2:
                print(f'Out of circle (x)')
                return None

        if self.topbot=='top':
            return self.yc + np.sqrt(self.R**2 - (x-self.xc)**2)
        elif self.topbot=='bot':
            return self.yc - np.sqrt(self.R**2 - (x-self.xc)**2)
        else:
            print(f'Unrecognized topbot: {self.topbot}')
            return None

    def t_vector(self, xt, yt=None):
        yt = yt or self.y(xt)
        yv = yt-self.yc
        xv = xt-self.xc
        return np.array([yv, -xv])/math.sqrt(xv**2+yv**2)
    def t_angle(self, xt, yt=None):
        yt = yt or self.y(xt)
        n_v = self.t_vector(xt, yt)
        return np.arctan2(n_v[1], n_v[0])

    def n_vector(self, xt, yt=None):
        yt = yt or self.y(xt)
        yv = yt-self.yc
        xv = xt-self.xc
        return np.array([xv, yv])/math.sqrt(xv**2+yv**2)
    def n_angle(self, xt, yt=None):
        yt = yt or self.y(xt)
        n_v = self.n_vector(xt, yt)
        return np.arctan2(n_v[1], n_v[0])

    def drawxy(self):
        x_rep = np.linspace(self.xc-self.R, self.xc+self.R, 2001, endpoint=True)
        y_rep = self.y(x_rep)
        return x_rep, y_rep

class Line:
    def __init__(self, xref=None, yref=None, angle=None, slope=None, vector=None, media=None, intensity=None):
        self.xref = xref
        self.yref = yref
        self.media = media
        self.intensity = intensity

        # path
        self.slope = None
        if angle is not None:
            self.angle = angle
            self.vector = (np.cos(self.angle), np.sin(self.angle))
            if not np.isclose(self.vector[0], 0.):
                self.slope = self.vector[1] / self.vector[0]
        elif slope is not None:
            self.slope = slope
            self.angle = np.arctan(self.slope)
            self.vector = (np.cos(self.angle), np.sin(self.angle))
        elif vector is not None:
            self.vector = vector
            self.angle = np.arctan2(self.vector[1], self.vector[0])
            if not np.isclose(self.vector[0], 0.):
                self.slope = self.vector[1] / self.vector[0]
        else:
            raise Exception('Invalid line')

        self.x0 = None
        self.y0 = None
        if self.slope is None: # vertical line
            self.x0 = self.xref
        elif np.isclose(self.slope, 0): # horizontal line
            self.y0 = self.yref
        else:
            self.x0 = self.xref - self.yref / self.slope
            self.y0 = self.yref - self.xref * self.slope

    def y(self, x):
        return self.yref + (x - self.xref) * self.slope

def contactpoints(line:Line, circle:Halfcircle):
    """Returns the x coordinates of the contact points between a line and a circle (in order of x rising).

    Parameters
    ----------
    alpha_
    y0_
    n_circle
    xc
    yc
    R

    Returns
    -------

    """
    A = 1+line.slope**2
    Bsur2 = line.slope*(line.y0-circle.yc)-circle.xc
    C = (line.y0-circle.yc)**2 - circle.xc**2 - circle.R**2
    Dsur4 = Bsur2**2 - A*C
    if Dsur4 >= 0 :
        x_possiblecontact1 = (-Bsur2 - np.sqrt(Dsur4))/A
        x_possiblecontact2 = (-Bsur2 + np.sqrt(Dsur4))/A
        return [x_possiblecontact1, x_possiblecontact2]
    lg.trace('contactpoints: NO CONTACT')
    return None

def isvalid(xtest, line, circle):
    # if the contact point is backwards
    if xtest < line.xref:
        return False
    # if the contact point is the departure point
    if np.isclose(xtest, line.xref):
        return False
    # if the contact point is not on the right half of the circle
    if not np.isclose(line.y(xtest), circle.y(xtest)):
        return False
    return True

def find_valid_contactpoints(line):
    global circle1, circle2

    c1pt = contactpoints(line, circle1)
    c2pt = contactpoints(line, circle2)

    lg.debug(f'contactpoints RAW 1: {c1pt} | 2: {c2pt}')
    if c1pt is not None:
        if not isvalid(c1pt[0], line, circle1):
            c1pt[0] = None
        if not isvalid(c1pt[1], line, circle1):
            c1pt[1] = None
        if c1pt[0] is None and (c1pt[1]) is None:
            c1pt = None
    if c2pt is not None:
        if not isvalid(c2pt[0], line, circle2):
            c2pt[0] = None
        if not isvalid(c2pt[1], line, circle2):
            c2pt[1] = None
        if c2pt[0] is None and (c2pt[1]) is None:
            c2pt = None
    lg.debug(f'contactpoints VALID 1: {c1pt} | 2: {c2pt}')
    return c1pt, c2pt



# <codecell>

w = .5 # rivulet width

b = 1.
R = b/2

xstart = -2*R # where the simulation starts
xend = 2*R # where the simulation ends

circle1 = Halfcircle(topbot = 'top', R=R, xc=0, yc = -w/2-R)
circle2 = Halfcircle(topbot = 'bot', R=R, xc=0, yc = +w/2+R)



# <codecell>

x1_rep, y1_rep = circle1.drawxy()
x2_rep, y2_rep = circle2.drawxy()

plt.figure()
ax = plt.gca()
ax.plot(x1_rep, y1_rep, c='r')
ax.plot(x2_rep, y2_rep, c='k')

ax.set_aspect('equal')
ax.set_xlim(xstart, xend)
ax.set_ylim(min(circle1.yc, circle2.yc)-max(circle1.R, circle2.R), max(circle1.yc, circle2.yc)+max(circle1.R, circle2.R))


# <codecell>

# One straight line


def addlinetodraw_fromx(line:Line, xstart, xend, **kwargs):
    global lines_to_draw, lines_to_draw_kwargs
    kwargs['alpha'] = kwargs.get('alpha', None) or line.intensity
    lines_to_draw.append([[xstart, xend], [line.y(xstart), line.y(xend)]])
    lines_to_draw_kwargs.append({**kwargs})

def addlinetodraw(xstart, xend, ystart, yend, **kwargs):
    global lines_to_draw, lines_to_draw_kwargs
    lines_to_draw.append([[xstart, xend], [ystart, yend]])
    lines_to_draw_kwargs.append({**kwargs})


# <codecell>

def drawnormale(xt, circle:Halfcircle, length=.1):
    yt = circle.y(xt)
    line = Line(xref=xt, yref = yt, vector=circle.n_vector(xt))

    vec = line.vector
    xstart = xt - length * vec[0]
    xend = xt + length * vec[0]
    ystart = yt - length * vec[1]
    yend = yt + length * vec[1]
    addlinetodraw(xstart=xstart, xend=xend, ystart=ystart, yend=yend, color='r', ls='--')

def drawtangent(xt, circle:Halfcircle, length=.1):
    yt = circle.y(xt)
    line = Line(xref=xt, yref = yt, vector=circle.t_vector(xt))

    vec = line.vector
    xstart = xt - length * vec[0]
    xend = xt + length * vec[0]
    ystart = yt - length * vec[1]
    yend = yt + length * vec[1]
    addlinetodraw(xstart=xstart, xend=xend, ystart=ystart, yend=yend, color='b', ls='--')


# <codecell>

lines_to_draw = []
lines_to_draw_kwargs = []
for l in [.1, .25, 1/np.sqrt(2)/2, np.sqrt(3)/4, .5]:
    drawnormale(l, circle1)
    drawnormale(l, circle2)
    drawnormale(-l, circle1)
    drawnormale(-l, circle2)
    drawtangent(l, circle1)
    drawtangent(l, circle2)
    drawtangent(-l, circle1)
    drawtangent(-l, circle2)
drawnormale(0, circle1)
drawnormale(0, circle2)
drawtangent(0, circle1)
drawtangent(0, circle2)

plt.figure()
ax = plt.gca()
ax.plot(x1_rep, y1_rep, c='k')
ax.plot(x2_rep, y2_rep, c='k')

for i_line, line_to_draw in enumerate(lines_to_draw):
    plt.plot(line_to_draw[0], line_to_draw[1], **lines_to_draw_kwargs[i_line])

ax.set_aspect('equal')
ax.set_xlim(xstart, xend)
ax.set_ylim(min(circle1.yc, circle2.yc)-max(circle1.R, circle2.R), max(circle1.yc, circle2.yc)+max(circle1.R, circle2.R))



# <codecell>

lines_to_draw = []
lines_to_draw_kwargs = []

epsilon = .5


media0 ='oil'

indices = {'oil': 1.3, 'air': 1., 'glass': 1.5}



# <codecell>


def drawonemoreline(line, showtg=False, **kwargs):
    if line is None:
        return (None, None)
    c1pts, c2pts = find_valid_contactpoints(line)
    if (c1pts is None) and (c2pts is None):
        addlinetodraw_fromx(line, line.xref, xend=xend, **kwargs)
        return (None, None)
    elif c1pts is not None:
        # Decide which point is good
        # HERE WE CONSIDER THAT THE THING NEVER GOES BACK
        # We take the closest or, if it is invalid, the farthest
        cpt_x = c1pts[0] or c1pts[1] # this should always work bcz if both are None, c1pt is None
        addlinetodraw_fromx(line, line.xref, xend=cpt_x, **kwargs)
        # now to the angle thingy
        if not np.isclose(line.y(cpt_x), circle1.y(cpt_x)):
            lg.warning(f'cpt_x = {cpt_x} | cpt_y = {line.y(cpt_x)} (line) = {circle1.y(cpt_x)} (circle1) [WHY NOT THE SAME?]')
        cpt_y = circle1.y(cpt_x)

        # compute the thing
        # medium
        medium1 = line.media
        if medium1 == 'oil':
            medium2 = 'air'
        elif medium1 == 'air':
            medium2 = 'oil'
        n1 = indices[medium1]
        n2 = indices[medium2]

        angle_entree = line.angle
        lg.debug(f'angleline: {angle_entree*180/np.pi} deg')
        if not(-np.pi/2 < angle_entree < np.pi/2):
            lg.error('angle line is not ok')

        angle_normale = circle1.n_angle(cpt_x)
        lg.debug(f'anglen: {angle_normale*180/np.pi} deg')
        if not(-np.pi< angle_normale < np.pi):
            lg.error('angle normal is not ok')

        # normal
        if showtg:
            drawnormale(cpt_x, circle1)
            drawtangent(cpt_x, circle1)

        i = np.pi - angle_normale + angle_entree
        otherside = i > np.pi/2
        if otherside:
            lg.debug('otherside: {}')
            i = angle_normale - angle_entree
        lg.debug(f'i (incident): {i*180/np.pi} deg')

        ### REFGRACTION
        line_refract = None
        if np.abs(n1*np.sin(i)/n2) > 1:
            lg.debug('NO REFRACTION')
            t_TE = 0.
            t_TM = 0.

        else:
            t = np.arcsin(n1*np.sin(i)/n2)
            lg.debug(f'i (refracted): {t*180/np.pi} deg')
            angle_sortie = t + angle_normale - np.pi
            if otherside:
                angle_sortie = angle_normale - t
            lg.debug(f'angle_sortie (refracted): {angle_sortie*180/np.pi} deg')
            if not(-np.pi/2 < angle_sortie < np.pi/2):
                lg.error('angle_sortie (refracted) is not ok')
            line_refract = Line(xref=cpt_x, yref=cpt_y, angle=angle_sortie, media=medium2)

        ### REFLEXION
        line_reflex = None
        r = -i
        lg.debug(f'i (reflection): {r*180/np.pi} deg')
        angle_sortie = r + angle_normale
        if otherside:
            angle_sortie = -r + angle_normale - np.pi # highly shady there
        lg.debug(f'angle_sortie (reflection): {angle_sortie*180/np.pi} deg')
        if not(-np.pi/2 < angle_sortie < np.pi/2):
            lg.error('angle_sortie (reflection) is not ok : canceling')
        else:
            line_reflex = Line(xref=cpt_x, yref=cpt_y, angle=angle_sortie, media=medium1)

        return (line_refract, line_reflex)

    elif c2pts is not None:
        # Decide which point is good
        # HERE WE CONSIDER THAT THE THING NEVER GOES BACK
        # We take the closest or, if it is invalid, the farthest
        cpt_x = c2pts[0] or c2pts[1] # this should always work bcz if both are None, c1pt is None
        addlinetodraw_fromx(line, line.xref, xend=cpt_x, **kwargs)
        return None, None



# <codecell>

lines_to_draw = []
lines_to_draw_kwargs = []

angle = -5 * np.pi/180
showtg = False
y0s = np.linspace(-w/2 - 2*R, 0, 101)[1:-1]

# # DEMONSTRATOR FOR MULTIPLE REFLEXIONS, BOTTOM
# y0s = [-0.375]
# showtg = True
# angle = -5 * np.pi/180

# # Demonstrator straight strong, BOTTOM
# y0s = [-0.7225]
# showtg = True
# angle = -5 * np.pi/180

# # DEMONSTRATOR FOR MULTIPLE REFLEXIONS, BOTTOM
y0s = [+0.375]
showtg = True
angle = 5 * np.pi/180

# This is difficult because you have to do real raytracing because  you have to take account reflexions !!
# because near normal incidence, reflexions are more important !!
# DO THE INTENSITY OF REFLEXIONS !!!!!

### NORMLEMENT LA TRIGO MARCHE POUR LE CERCLE 1
### PEUT ETRE DES CHOSES A ADAPTER (otherside inversé ??) POUR LE CERCLE 2

n_steps = 12
colors = ['purple', 'b', 'green', 'yellow', 'orange', 'red', 'm', 'k']
for y0 in y0s:
    line_i = Line(xref = xstart, yref = y0,  angle = angle, media='oil')
    lines = [line_i]

    for i_step in range(n_steps):
        color = colors[i_step%len(colors)]
        # if np.all([line is None for line in lines]):
        #     break
        # else:
        if True:
            lg.info(f'Step {i_step}, drawing {color} lines')
            newlines = []
            for line in lines:
                lt, lr = drawonemoreline(line, showtg=showtg, color=color, alpha=.5)
                newlines.append(lt)
                newlines.append(lr)
            lines = newlines


    if np.any(line is not None for line in lines):
        lg.warning('Lacking a step !!')


# <codecell>

plt.figure()
ax = plt.gca()
ax.plot(x1_rep, y1_rep, c='k')
ax.plot(x2_rep, y2_rep, c='k')

for i_line, line_to_draw in enumerate(lines_to_draw):
    plt.plot(line_to_draw[0], line_to_draw[1], **lines_to_draw_kwargs[i_line])

ax.set_aspect('equal')
ax.set_xlim(xstart, xend)
ax.set_ylim(min(circle1.yc, circle2.yc)-max(circle1.R, circle2.R), max(circle1.yc, circle2.yc)+max(circle1.R, circle2.R))


# <codecell>




# <codecell>

# lines_to_draw = []
# lines_to_draw_kwargs = []
#
#
# X = -0.49
#
# anglen = circle2.n_angle(X)
# lg.debug(f'anglen: {anglen * 180 / np.pi} deg')
# drawnormale(X, circle2)
#
# print(lines_to_draw)
#
# plt.figure()
# ax = plt.gca()
# ax.plot(x1_rep, y1_rep, c='k')
# ax.plot(x2_rep, y2_rep, c='k')
#
# for i_line, line_to_draw in enumerate(lines_to_draw):
#     print(i_line)
#     plt.plot(line_to_draw[0], line_to_draw[1], **lines_to_draw_kwargs[i_line])
#
# ax.set_aspect('equal')
# ax.set_xlim(xstart, xend)
# ax.set_ylim(min(circle1.yc, circle2.yc)-max(circle1.R, circle2.R), max(circle1.yc, circle2.yc)+max(circle1.R, circle2.R))


# <codecell>




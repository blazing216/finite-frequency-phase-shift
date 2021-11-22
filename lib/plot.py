#-*-coding:utf-8-*-
from __future__ import print_function, division, unicode_literals
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import AnchoredText
from matplotlib.legend_handler import HandlerPatch
#from collections.abc import Iterable

a4 = (8.27, 11.69)
a4ratio = a4[1]/a4[0]
inch2cm = 2.54
cm2inch = 1/2.54
quickfig = dict(bottom=0.7, top=0.9, left=0.1, right=0.4)

# selected colors
Orange= np.array([247,144,61])/255
Green = np.array([89,169,90])/255
Blue = np.array([77,133,189])/255

class Figure:
    def __init__(self, fig):
        self.fig = fig

    def savefig(self, figname, dpi=300):
        self.fig.savefig(figname, dpi=dpi, bbox_inches='tight')

def fill_plot(x, y, *args, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()
    if 'vertical' in kwargs:
        vertical = kwargs.pop('vertical')
    else:
        vertical = False
    if 'baseline' in kwargs:
        baseline = kwargs.pop('baseline')
    else:
        baseline = 0
    if 'fill_pos' in kwargs:
        fill_pos = kwargs.pop('fill_pos')
    else:
        fill_pos = None
    if 'fill_neg' in kwargs:
        fill_neg = kwargs.pop('fill_neg')
    else:
        fill_neg = None

    if vertical:
        ax.plot(y+baseline, x, *args, **kwargs)
        if fill_pos is not None:
            ax.fill_betweenx(x, y+baseline, baseline, where=y>0,
                interpolate=True, facecolor=fill_pos, **kwargs)
        if fill_neg is not None:
            ax.fill_betweenx(x, y+baseline, baseline, where=y<0,
                interpolate=True, facecolor=fill_neg, **kwargs)
    else:
        ax.plot(x, y+baseline, *args, **kwargs)

        if fill_pos is not None:
            ax.fill_between(x, y+baseline, baseline, where=y>0,
                interpolate=True, facecolor=fill_pos, **kwargs)
        if fill_neg is not None:
            ax.fill_between(x, y+baseline, baseline, where=y<0,
                interpolate=True, facecolor=fill_neg, **kwargs)
    return None

def labelax(axs, labels=None, loc='upper left', frameon=False, 
            borderpad=0.1, color=None, fontdict=None, **kwargs):
    try:
        iter(axs)
    except:
        axs_tmp = [axs]
    else:
        axs_tmp = axs

    if labels is None:
        labels_tmp = list(map(chr, range(97, 97+len(axs))))
    elif isinstance(labels, str):
        labels_tmp = [labels]
    else:
        try:
            iter(labels)
        except:
            labels_tmp = [labels]
        else:
            labels_tmp = [v for v in labels]

    if len(labels_tmp) > len(axs_tmp):
        labels_tmp = labels_tmp[:len(axs_tmp)]
    elif len(labels_tmp) < len(axs_tmp):
        labels_tmp += ([''] * (len(axs_tmp) - len(labels_tmp)))
#         assert len(labels)  len(axs_tmp), \
#                "labels (len = %d) should have the same length with axs (len = %d)" % (len(labels), len(axs_tmp))
        
    for ax, label in zip(axs_tmp, labels_tmp):
        # print('try add one')
        
        # prop_tmp must be re-initialized every iteration, otherwise 'AnchoredText'
        # raises 'KeyError'. I don't know the reason.
        prop_tmp = dict(size=8, weight='bold')
        if fontdict is not None:
            # default value of prop cannot be set in function
            # definition, otherwise it will raise a KeyError.
            # Maybe dictionary is not suitable for a default value
            # see: https://docs.python-guide.org/writing/gotchas/
            prop_tmp.update(fontdict)

        if color is not None:
            prop_tmp.update({'color': color})
        
        at = AnchoredText(label, loc=loc, frameon=frameon,
                          borderpad=borderpad, prop=prop_tmp, 
                          bbox_transform=ax.transAxes, **kwargs)
        ax.add_artist(at)
        # print('add one')

def offset_transform(dx=1.0/72, dy=1.0/72, transform=None, ax=None, fig=None):
    if fig is None:
        if ax is None:
            fig = plt.gcf()
        else:
            fig = ax.figure

    if transform is None:
        return mtransforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    else:
        return transform + mtransforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

def auxplot():
    for v in np.linspace(0,1,11):
        plt.plot([0,1], [v,v], 'r', clip_on=False,
                 transform=plt.gcf().transFigure)
        plt.plot([v,v], [0,1], 'r', clip_on=False,
                 transform=plt.gcf().transFigure)

def savefig(name, dpi=300, **kwargs):
    if 'transparent' in kwargs:
        transparent = kwargs.pop('transparent')
    else:
        transparent = False

    plt.savefig(name+'.pdf', dpi=dpi, **kwargs)
    plt.savefig(name+'_tight.pdf', dpi=dpi, bbox_inches='tight', **kwargs)
    plt.savefig(name+'.png', bbox_inches='tight', dpi=dpi, 
                transparent=transparent, **kwargs)

def points2path(x, y):
    valid = (~np.isnan(x) ) & (~np.isnan(y))
    verts = np.column_stack([x[valid],y[valid]])
    verts = np.vstack([verts, np.array([0,0])])
    codes = np.ones(len(verts)) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    codes[-1] = mpath.Path.CLOSEPOLY
    path = mpath.Path(verts, codes)

    return path

def points2patch(x, y, **kwargs):
    path = points2path(x, y)
    patch = mpatches.PathPatch(path, **kwargs)

    return patch

def plot_poly(x, y, ax=None, **kwargs):
    patch = points2patch(np.asarray(x), np.asarray(y), **kwargs)
    if ax is None:
        plt.gca().add_patch(patch)
    else:
        ax.add_patch(patch)
    return patch

def plot_error_region(x, y, x1, y1, ax=None, **kwargs):
    patch = plot_poly(np.hstack([x, x[::-1]]),
                     np.hstack([y, y1[::-1]]),
                      ax=ax,
                      **kwargs)
    return patch

def plot_rect(x0, y0, x1, y1, ax=None, **kwargs):
    patch = plot_poly(np.array([x0, x1, x1, x0, x0]),
            np.array([y0, y0, y1, y1, y0]),
            ax=ax, **kwargs)
    return patch

def bins(x, eps=1e-2):
    '''automatically get bins for either
    linearly or logarithmtically spaced series'''
    if is_linear_spaced(x):
        return linbins(x)
    elif is_log_spaced(x):
        return logbins(x)
    else:
        raise Exception("x is neither linear nor log spaced")

def linbins(x):
    '''get bins for a linearly spaced series x'''
    dx = x[1] - x[0]
    xbins = np.hstack([x, x[-1]+dx]) - 0.5 * dx
    return xbins

def logbins(x):
    '''get bins for a logarithmically spaced series x'''
    dx = x[1] / x[0]
    sqrt_dx = np.sqrt(dx)
    xbins = np.hstack([x, x[-1] * dx]) / sqrt_dx
    return xbins

def is_linear_spaced(x, eps=1e-2):
    '''check if a series is linearly spaced'''
    if len(x) < 2:
        return False
    else:
        return np.abs(((x[1:] - x[:-1]) - (x[1] - x[0]    )) / (x[1] - x[0])).max() < eps

def is_log_spaced(x, eps=1e-2):
    '''check if a series is linearly spaced'''
    logx = np.log(x)
    return is_linear_spaced(logx, eps=eps)


def ticklocator_params(xmajor=None, xminor=None,
                       ymajor=None, yminor=None,
                       xmajorgrid=False, xminorgrid=False,
                       ymajorgrid=False, yminorgrid=False, ax=None):
    if ax is None:
        ax = plt.gca()

    if xmajor is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xmajor))
    if xminor is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(xminor))
    if ymajor is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(ymajor))
    if yminor is not None:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(yminor))

    if xmajorgrid is True:
        ax.grid(axis='x', which='major')
    if xminorgrid is True:
        ax.grid(axis='x', which='minor')
    
    if ymajorgrid is True:
        ax.grid(axis='y', which='major')
    if yminorgrid is True:
        ax.grid(axis='y', which='minor')

def axis_params(xlim=None, xlabel=None, ylim=None, ylabel=None,
        leftcolor=None, rightcolor=None,
        bottomcolor=None, topcolor=None, 
        left=True, right=True,
        bottom=True, top=True,
        ax=None):
    if ax is None:
        ax = plt.gca()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if leftcolor is not None:
        ax.spines['left'].set_color(leftcolor)
        ax.tick_params(axis='y', which='both', color=leftcolor,
                labelcolor=leftcolor)
        label = ax.get_ylabel()
        ax.set_ylabel(label, color=leftcolor)

    if rightcolor is not None:
        ax.spines['right'].set_color(rightcolor)
        ax.tick_params(axis='y', which='both', color=rightcolor,
                labelcolor=rightcolor)
        label = ax.get_ylabel()
        ax.set_ylabel(label, color=rightcolor)

    if topcolor is not None:
        ax.spines['top'].set_color(topcolor)
        ax.tick_params(axis='x', which='both', color=topcolor,
                labelcolor=topcolor)
        label = ax.get_xlabel()
        ax.set_xlabel(label, color=topcolor)

    if bottomcolor is not None:
        ax.spines['bottom'].set_color(bottomcolor)
        ax.tick_params(axis='x', which='both', color=bottomcolor,
                labelcolor=bottomcolor)
        label = ax.get_xlabel()
        ax.set_xlabel(xlabel, color=bottomcolor)

    if left is False:
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft=False, left=False, which='both')

    if right is False:
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelright=False, right=False, which='both')

    if bottom is False:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False, bottom=False, which='both')

    if top is False:
        ax.spines['top'].set_visible(False)
        ax.tick_params(labeltop=False, top=False, which='both')

# legend
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        minradius = min(width + xdescent, height + ydescent)
        p = mpatches.Ellipse(xy=center, width=minradius,
                             height=minradius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def make_mp4(filewildcards, mp4file='movie.mp4', framerate=25, fps=None):
    if fps is None:
        fps = framerate

    if filewildcards.find('*') > 0:
        call(f"ffmpeg -r {framerate} -pattern_type glob -i '{filewildcards}'  -pix_fmt yuv420p -r {fps} {mp4file}",
            shell=True)
    else:
        call(f'ffmpeg -r {framerate} -i {filewildcards}  -pix_fmt yuv420p -r {fps} {mp4file}',
            shell=True)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from tqdm import tqdm

import ygeolib.plot as yplot
import ygeolib.utils as yutils
from ygeolib.layermodel import Model1D
import ygeolib.disp_xu as disp

from . import synthetic, phaseshift

DB = Model1D.from_thk_vs(\
    thk=[0.15, 0.5, 1.35, 8.0, 0.0],
    vs=[2.44, 2.88, 3.25, 3.50, 3.70])

Dublin_basin_sonic_log_img = ('/home/xuyh/YX/Marathon/'
    'Project_Phase_shift/src/phase_shift_lib/data/'
    'vp_model_dublin_basin.png')

def plot_sonic_log_img():
    im = plt.imread(Dublin_basin_sonic_log_img)
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    plt.imshow(im)
    fig.canvas.draw()

    data2fig = ax.transData + fig.transFigure.inverted()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    data_points = [(281.9, 67.3),
                   (281.9, 537.9),
                   (395.2, 537.9),
                   (395.2, 67.3)]
    fig_points = [data2fig.transform(data_point)\
              for data_point in data_points]
    ax1 = fig.add_axes([fig_points[1][0], fig_points[1][1],
                    fig_points[3][0]-fig_points[1][0],
                    fig_points[3][1]-fig_points[1][1]],
                  facecolor='none')
    ax1.set_xlim(3,7)
    ax1.set_ylim(1.5,0)

    yplot.axis_params(leftcolor='r', rightcolor='r',
        topcolor='r', bottomcolor='r', ax=ax1)

    ax.axis('off')

    return fig, ax1

def example_para1():
    fmin = 0.5
    fmax = 30
    df = 0.01
    tmin = -4
    tmax = 10
    dt = 0.01

    f = np.arange(fmin, fmax+0.1*df, df)
    t = np.arange(tmin, tmax+0.1*dt, dt)

    return f, t

def example_para_for_quick_test():
    fmin = 0.1
    fmax = 30
    df = 0.1
    tmin = -30
    tmax = 30
    dt = 0.1

    f = np.arange(fmin, fmax+0.1*df, df)
    t = np.arange(tmin, tmax+0.1*dt, dt)

    return f, t

def example_para1_for_paper():
    fmin = 0.1
    fmax = 30
    df = 0.001
    tmin = -30
    tmax = 30
    dt = 0.004

    f = np.arange(fmin, fmax+0.1*df, df)
    t = np.arange(tmin, tmax+0.1*dt, dt)

    return f, t

def example_para2_for_paper():
    fmin = 0.1
    fmax = 30
    df = 0.01
    tmin = -30
    tmax = 30
    dt = 0.004

    f = np.arange(fmin, fmax+0.1*df, df)
    t = np.arange(tmin, tmax+0.1*dt, dt)

    return f, t

def cR(f):
    c = disp.cR(DB, 1/f, mode=0)
    return c

def surface_wave_for_paper(dist, para_func=None,
        farfield=False, casual_branch_only=True):
    if para_func is None:
        para_func = example_para2_for_paper
    f, t = para_func()
    c = cR(f)

    seis = synthetic.surface_wave(f, c, t, dist=dist,
            farfield=farfield,
            casual_branch_only=casual_branch_only)

    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

def plot_debug_surface_wave_bessel(dist):
    sw = surface_wave_for_paper(dist)
    sw_far = surface_wave_for_paper(dist, farfield=True)
    
    plt.figure(figsize=(4,4))
    
    plt.subplot(211)
    plt.plot(sw['t'], sw['seis'], '-k', label='Bessel')
    plt.plot(sw_far['t'], sw_far['seis'], '-r',
            label='Cosine')
    plt.legend()
    plt.xlim(-2, 2)
    
    plt.subplot(212)
    f = np.linspace(0.1,0.2,10)
    _, t = example_para2_for_paper()
    c = cR(f)
    wvfm = synthetic.surface_wave(f, c, t, dist,
            return_single_frequencies=True)
    wvfm_far = synthetic.surface_wave(f, c, t, dist,
            farfield=True,
            return_single_frequencies=True)
    for i in range(wvfm.shape[1]):
        plt.plot(t, wvfm[:,i]/np.abs(wvfm[:,i]).max()+i, '-k')
        plt.plot(t, wvfm_far[:,i]/np.abs(wvfm_far[:,i]).max()+i, '-r')
    plt.xlim(-5, 5)
    
    plt.tight_layout()

def plot_debug_surface_wave_for_paper(dist, f=None, para_func=None):
    if para_func is None:
        para_func = example_para2_for_paper
    if f is None:
        f, t = para_func()
    elif np.isscalar(f):
        _, t = para_func()
        f = np.array([f])
    else:
        _, t = para_func()
        f = np.array(f)
    c = cR(f)

    nf = len(f)
    if nf >= 2:
        df = f[1] - f[0]
        weight = np.ones(nf)
        weight[0] = 0.5
        weight[-1] = 0.5
    elif nf == 1:
        df = 1
        weight = np.ones(nf)

    omega = 2 * np.pi * f

    nt = len(t)

    t_colvec = t.reshape(-1,1)

    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*f*dist/c) * \
        np.exp(1j * 2*np.pi*f*t_colvec)

    wvfm_sum = np.sum(weight * wvfm, axis=1) * df

    wvfm_cos = 1 / (np.pi) * \
        np.sqrt(c/(f*dist)) * \
        np.cos(2*np.pi*f*t_colvec - 2*np.pi*f*dist/c + np.pi/4)

    wvfm_cos_sum = np.sum(weight * wvfm_cos, axis=1) * df

    def plot_tick(t0, t, x, *args, **kwargs):
        x0 = np.interp(t0, t, x)
        plt.plot(t0, x0, *args, **kwargs)

    def tstar(dist, c):
        return dist/c

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    plt.figure(figsize=(6,2.5))

    plt.subplot(121)
    for i in range(nf):
        x = wvfm[:,i].real / np.abs(wvfm[:,i].real).max()/2 + i
        x_cos = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=4, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=5.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=5.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=4, mfc='salmon',
            alpha=0.7)

    plt.plot(t, wvfm_sum.real / np.abs(wvfm_sum.real).max()/2 + nf, '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['sum']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km (Bessel)')
    
    
    plt.subplot(122)
    for i in range(nf):
        x = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        #x_cos = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=4, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=5.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=5.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=4, mfc='salmon',
            alpha=0.7)
    plt.plot(t, wvfm_cos_sum.real / np.abs(wvfm_cos_sum.real).max()/2 + nf,
        '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['sum']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km (cosine)')
    
    plt.tight_layout()
    
def plot_debug_surface_wave_for_paper_v2(dist, f=None, para_func=None):
    if para_func is None:
        para_func = example_para2_for_paper
    if f is None:
        f, t = para_func()
    elif np.isscalar(f):
        _, t = para_func()
        f = np.array([f])
    else:
        _, t = para_func()
        f = np.array(f)
    c = cR(f)

    nf = len(f)
    if nf >= 2:
        df = f[1] - f[0]
        weight = np.ones(nf)
        weight[0] = 0.5
        weight[-1] = 0.5
    elif nf == 1:
        df = 1
        weight = np.ones(nf)

    omega = 2 * np.pi * f

    nt = len(t)

    t_colvec = t.reshape(-1,1)

#     wvfm = 1 * \
#         - hankel1(0, - 2*np.pi*f*dist/c) * \
#         np.exp(1j * 2*np.pi*f*t_colvec)
    wvfm = synthetic.surface_wave(f, c, t, dist,
            return_single_frequencies=True)

    wvfm_sum = np.sum(weight * wvfm, axis=1) * df

#     wvfm_cos = 1 / (np.pi) * \
#         np.sqrt(c/(f*dist)) * \
#         np.cos(2*np.pi*f*t_colvec - 2*np.pi*f*dist/c + np.pi/4)
    wvfm_cos = synthetic.surface_wave(f, c, t, dist,
            farfield=True,
            return_single_frequencies=True)

    wvfm_cos_sum = np.sum(weight * wvfm_cos, axis=1) * df

    def plot_tick(t0, t, x, *args, **kwargs):
        x0 = np.interp(t0, t, x)
        plt.plot(t0, x0, *args, **kwargs)

    def tstar(dist, c):
        return dist/c

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    plt.figure(figsize=(6,2.5))

    plt.subplot(121)
    for i in range(nf):
        x = wvfm[:,i].real / np.abs(wvfm[:,i].real).max()/2 + i
        x_cos = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=4, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=5.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=5.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=4, mfc='salmon',
            alpha=0.7)

    plt.plot(t, wvfm_sum.real / np.abs(wvfm_sum.real).max()/2 + nf, '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['sum']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km (Bessel)')
    
    
    plt.subplot(122)
    for i in range(nf):
        x = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        #x_cos = wvfm_cos[:,i].real / np.abs(wvfm_cos[:,i].real).max()/2 + i
        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=4, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=5.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=5.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=4, mfc='salmon',
            alpha=0.7)
    plt.plot(t, wvfm_cos_sum.real / np.abs(wvfm_cos_sum.real).max()/2 + nf,
        '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['sum']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km (cosine)')
    
    plt.tight_layout()
    
def plot_debug_phaseshift_bessel(dist, f=1):
    colors = plt.get_cmap('tab10')(np.linspace(0,1,10))
    if np.isscalar(f):
        f = np.array([f])
    else:
        f = np.array(f)

    plt.figure(figsize=(3,2))
    for i, fv in enumerate(f):
        c = cR(fv)
        dphi = np.angle(- hankel1(0, - 2*np.pi*fv*dist/c) / \
                (np.exp(1j*(-2*np.pi*fv*dist/c + np.pi/4))) )
        dphi *= -1
        plt.plot(dist, dphi, lw=0.5, color=colors[i],
            label=f'{fv:g} Hz')

    plt.axhline(0, dashes=(10,10))
    plt.xlim(0.5, 4.0)
    plt.xlabel('Distance (km)')
    plt.gca().set_yticks([-np.pi/30, -np.pi/60, 0,
        np.pi/60, np.pi/30])
    plt.gca().set_yticklabels(['$-\\pi/30$', '$-\\pi/60$',
        '0', '$\\pi/60$', '$\\pi/30$'])
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(np.pi/120))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.ylim(-np.pi/30, np.pi/30*0.1)
    plt.ylabel('Phase shift (radian)')

    plt.legend(loc='lower right')


def plot_debug_surface_wave_for_paper_using_cos(dist, f=None, para_func=None):
    if para_func is None:
        para_func = example_para2_for_paper
    if f is None:
        f, t = para_func()
    elif np.isscalar(f):
        f = np.array([f])
    else:
        _, t = para_func()
        f = np.array(f)
    c = cR(f)

    nf = len(f)
    if nf >= 2:
        df = f[1] - f[0]
        weight = np.ones(nf)
        weight[0] = 0.5
        weight[-1] = 0.5
    elif nf == 1:
        df = 1
        weight = np.ones(nf)

    omega = 2 * np.pi * f

    nt = len(t)

    t_colvec = t.reshape(-1,1)

    #wvfm = 1 * \
    #    - hankel1(0, - 2*np.pi*f*dist/c) * \
    #    np.exp(1j * 2*np.pi*f*t_colvec)

    #wvfm_sum = np.sum(weight * wvfm, axis=1) * df

    wvfm = 1 / (np.pi) * \
        np.sqrt(c/(f*dist)) * \
        np.cos(2*np.pi*f*t_colvec - 2*np.pi*f*dist/c + np.pi/4)

    wvfm_sum = np.sum(weight * wvfm, axis=1) * df

    def plot_tick(t0, t, x, *args, **kwargs):
        x0 = np.interp(t0, t, x)
        plt.plot(t0, x0, *args, **kwargs)

    def tstar(dist, c):
        return dist/c

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    plt.figure(figsize=(4,3))

    #plt.subplot(121)
    for i in range(nf):
        x = wvfm[:,i].real / np.abs(wvfm[:,i].real).max()/2 + i
        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=6, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=7.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=3.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=2, mfc='salmon',
            alpha=0.7)

    plt.plot(t, wvfm_sum.real / np.abs(wvfm_sum.real).max()/2 + nf, '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['sum']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km')


def plot_debug_phaseshift(dist, f=None, gamma=1, para_func=None):
    sw = surface_wave_for_paper(dist)
    c = [np.interp(fv, sw['f'], sw['c']) for fv in f]
    c = np.array(c)
    if np.isscalar(f):
        f = np.array([f])
    else:
        f = np.array(f)

    def plot_tick(t0, t, x, *args, **kwargs):
        x0 = np.interp(t0, t, x)
        plt.plot(t0, x0, *args, **kwargs)

    def tstar(dist, c):
        return dist/c

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    plt.figure(figsize=(4,3))

    nf = len(f)
    t = sw['t']
    for i in range(nf):
        x = phaseshift.gaussian_filter(sw, f[i], gamma=gamma,
                emin=7)
        x = x / np.abs(x).max() / 2 + i

        plt.plot(t, x)
        #plt.plot(t, x_cos, color='r', dashes=(10,10))
        plot_tick(tstar(dist, c[i]), t, x, 'kD', ms=4, mfc='lightgray')
        plot_tick(t0(dist, c[i], f[i]), t, x, 'kH', ms=5.5, mfc='lightgray')

        t0_measure = phaseshift.closest_maximum_accurate(t0(dist, c[i], f[i]),
            t, x)
        plot_tick(t0_measure, t, x, 'kH', ms=5.5, mfc='salmon',
            alpha=0.7)
        plot_tick(t0_measure+1/(8*f[i]), t, x, 'kD', ms=4, mfc='salmon',
            alpha=0.7)

    plt.plot(t, sw['seis'] / np.abs(sw['seis']).max() / 2 + nf, '-r')
    plt.axvline(0, dashes=(10,10))

    plt.xlim(-1.5, 1.5)
    plt.gca().set_yticks(np.arange(nf+1))
    yticklabels = [f'{fv:.3g} Hz' for fv in f] + ['broadband']
    plt.gca().set_yticklabels(yticklabels)
    plt.xlabel('Time (s)')

    from matplotlib.lines import Line2D
    def markerhandle(*args, **kwargs):
        return Line2D([], [], *args, **kwargs)
    plt.legend(handles=[markerhandle(linestyle='none', marker='H', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='salmon'),
        markerhandle(linestyle='none', marker='H', color='k', mfc='lightgray'),
        markerhandle(linestyle='none', marker='D', color='k', mfc='lightgray')],
        labels=['t0$_{obs}$', 't$^*_{obs}$', 't0', 't$^*$'],
        loc='lower left',
        )
    plt.title(f'$\\Delta$ = {dist:.3g} km')


def example_mft_fc_for_paper(sw):
    f = np.logspace(np.log10(sw['f'][0]),
        np.log10(sw['f'][-1]),
        51)
    c = np.interp(f, sw['f'], sw['c'])
    return f, c

def ptt(sw, gamma, **kwargs):
    f, c = example_mft_fc_for_paper(sw)
    ptt = phaseshift.ptt_ag(sw, f, c,
        gamma=gamma, emin=7, **kwargs)
    ptt0 = phaseshift.ptt_theo(sw, f, c,
        iridge=0)
    return {'f':f, 'c':c, 't':ptt,
            't0':ptt0}

def ptts(dist, gamma, **kwargs):
    res = []
    ndist = len(dist)
    for i in tqdm(range(ndist)):
        d = dist[i]
        sw = surface_wave_for_paper(d)
        phase_tt = ptt(sw, gamma=gamma, **kwargs)
        phase_tt.update({'dist': d, 'gamma': gamma})
        res.append(phase_tt)
    return res

def ptts_for_paper(gamma=1, **kwargs):
    dist = np.arange(0.5, 4.001, 0.1)
    return ptts(dist, gamma, **kwargs)

def ptts_for_paper_to_file(outfile, gamma=1, force=False, **kwargs):
    if not force:
        print('Use force=True to run. Notice that it might take a few minutes', flush=True)
        return

    res = ptts_for_paper(gamma, **kwargs)
    pickle.dump(res,
            open(outfile, 'wb'))
    print(outfile, flush=True)

def ptts_for_paper_single_frequency():
    dist = np.arange(0.5, 4.001, 0.1)
    sw_f, t = example_para2_for_paper()
    f = np.logspace(np.log10(sw_f[0]),
        np.log10(sw_f[-1]), 51)
    c = cR(f)
    nt = len(t)
    t_colvec = t.reshape(-1,1)

    ptts = []
    ndist = len(dist)
    for i in tqdm(range(ndist)): 
    #for i, d in enumerate(dist):
        d = dist[i]
        wvfm = 1 * \
            - hankel1(0, - 2*np.pi*f*d/c) * \
            np.exp(1j * 2*np.pi*f*t_colvec)
        wvfm = wvfm.real

        t0 = d / c - 1/(8*f)
        t0_measure = [phaseshift.closest_maximum_accurate(t0[j],
                t, wvfm[:,j]) for j in range(wvfm.shape[1])]
        ptts.append({'f':f, 'c':c, 't':np.array(t0_measure),
            't0':t0, 'dist': d, 'gamma':1})
    return ptts

def ptts_nearfield_finite_frequency(dist, gamma, **kwargs):
    res = []
    ndist = len(dist)
    ptts_theo = []
    ptts_sf = []
    ptts_ag = []
    for i in tqdm(range(ndist)):
        d = dist[i]
        sw = surface_wave_for_paper(d)
        f, c = example_mft_fc_for_paper(sw)
        t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)
        t_sf = phaseshift.ptt_single_frequency(sw, f, c, iridge=0,
                farfield=False, casual_branch_only=True)
        t_ag = phaseshift.ptt_ag(sw, f, c, gamma=gamma,
                emin=7, **kwargs)
        ptts_theo.append(t_theo)
        ptts_sf.append(t_sf)
        ptts_ag.append(t_ag)
    ptts_theo = np.array(ptts_theo)
    ptts_sf = np.array(ptts_sf)
    ptts_ag = np.array(ptts_ag)

    return f, dist, ptts_theo, ptts_sf, ptts_ag

def ptts_nearfield_finite_frequency_for_paper():
    dist = np.arange(0.5, 4.001, 0.1)
    return ptts_nearfield_finite_frequency(dist, gamma=1, ampratio=2)

def plot_debug_ptts_farfield_multifreq(dist=3, f=0.5):
    sw = surface_wave_for_paper(dist, para_func=None,
        farfield=False, casual_branch_only=True)
    t0 = phaseshift.ptt_t0(sw, sw['f'], sw['c'], iridge=0)
    t_theo = phaseshift.ptt_theo(sw, sw['f'], sw['c'], iridge=0)
    t_sf = phaseshift.ptt_single_frequency(sw, sw['f'], sw['c'], iridge=0,
            farfield=False, casual_branch_only=True)
    t_ag = phaseshift.ptt_ag(sw, sw['f'], sw['c'], gamma=1,
            iridge=0, ampratio=2)
    sw_far = surface_wave_for_paper(dist, para_func=None,
        farfield=True, casual_branch_only=True)
    t0_far = phaseshift.ptt_t0(sw_far, sw_far['f'], sw_far['c'], iridge=0)
    t_theo_far = phaseshift.ptt_theo(sw_far, sw_far['f'], sw_far['c'], iridge=0)
    t_sf_far = phaseshift.ptt_single_frequency(sw_far, sw_far['f'], sw_far['c'], iridge=0,
            farfield=True, casual_branch_only=True)
    t_ag_far = phaseshift.ptt_ag(sw_far, sw_far['f'], sw_far['c'], gamma=1,
            iridge=0, ampratio=2)

    plt.figure(figsize=(6,6))

    plt.subplot(321)
    plt.plot(sw['t'], sw['seis'], '-k')
    plt.plot(sw_far['t'], sw_far['seis'], '-.k')
    plt.xlim(0, 3)
    plt.xlabel('Time (s)')
    plt.title('Seis (Bessel)')

    plt.subplot(322)
    plt.plot(sw['f'], t0, '-k', label='t0')
    plt.plot(sw['f'], t_theo, '-r', label='theo')
    plt.plot(sw['f'], t_sf, '-g', label='sf')
    plt.plot(sw['f'], t_ag, '-b', label='ag')
    plt.semilogx()
    plt.xlim(1, 25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')
    plt.title('Travel time')
    plt.legend()

    plt.subplot(323)
    plt.plot(sw['f'], t0-t0, '-k', label='t0')
    plt.plot(sw['f'], t_theo-t0, '-r', label='theo')
    plt.plot(sw['f'], t_sf-t0, '-g', label='sf')
    plt.plot(sw['f'], t_ag-t0, '-b', label='ag')
    plt.semilogx()
    plt.xlim(1,25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')
    plt.title('Travel time residual')
    plt.legend()

    plt.subplot(324)
    #plt.plot(sw['f'], (t0-t0)*(2*np.pi*sw['f'])/np.pi, '-k', label='t0')
    plt.plot(sw['f'], (t_theo-t0)*(2*np.pi*sw['f'])/np.pi, '-r', label='theo', lw=0.5)
    plt.plot(sw['f'], (t_sf-t0)*(2*np.pi*sw['f'])/np.pi, '-g', label='sf', lw=0.5)
    plt.plot(sw['f'], (t_ag-t0)*(2*np.pi*sw['f'])/np.pi, '-b', label='ag', lw=0.5)
    plt.plot(sw['f'], (t_theo_far-t0)*(2*np.pi*sw['f'])/np.pi, '-.r', label='theo_far', lw=0.5)
    plt.plot(sw['f'], (t_sf_far-t0)*(2*np.pi*sw['f'])/np.pi, '-.g', label='sf_far', lw=0.5)
    plt.plot(sw['f'], (t_ag_far-t0)*(2*np.pi*sw['f'])/np.pi, '-.b', label='ag_far', lw=0.5)
    plt.semilogx()
    plt.xlim(1,25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase shift ($\\times$ $\pi$)')
    plt.title('Phase shift')
    plt.legend()

    plt.subplot(325)
    plt.plot(sw['f'], (t_ag - t_sf)*(2*np.pi*sw['f'])/np.pi, color='k', label='bessel', lw=0.5)
    plt.plot(sw['f'], (t_ag_far - t_sf_far)*(2*np.pi*sw['f'])/np.pi, color='r', label='farfield', lw=0.5)
    plt.semilogx()
    plt.xlim(1,25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase shift ($\\times$ $\pi$)')
    plt.title('finite frequency phase shift')
    plt.legend()

    plt.subplot(326)
    plt.plot(sw['f'], (t_ag - t_ag_far)*(2*np.pi*sw['f'])/np.pi, color='k', label='narrowband', lw=0.5)
    plt.plot(sw['f'], (t_sf - t_sf_far)*(2*np.pi*sw['f'])/np.pi, color='r', label='single frequency', lw=0.5)
    plt.semilogx()
    plt.xlim(1,25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase shift ($\\times$ $\pi$)')
    plt.title('near-field phase shift')
    plt.legend()
    
    plt.tight_layout()
    
    

def plot_debug_ptts_for_paper_single_frequency(dist=3, f=None):
    sw_f, t = example_para2_for_paper()
    if f is None:
        f = np.logspace(np.log10(sw_f[0]),
            np.log10(sw_f[-1]), 51)
    elif np.isscalar(f):
        f = np.array([f])
    else:
        f = np.array(f)

    c = cR(f)
    nt = len(t)
    t_colvec = t.reshape(-1,1)

    ptts = []
    #ndist = len(dist)
    #for i in tqdm(range(ndist)): 
    #for i, d in enumerate(dist):
    d = dist
    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*f*d/c) * \
        np.exp(1j * 2*np.pi*f*t_colvec)
    wvfm = wvfm.real

    t0 = d / c - 1/(8*f)
    t0_measure = [phaseshift.closest_maximum_accurate(t0[j],
            t, wvfm[:,j]) for j in range(wvfm.shape[1])]

    plt.figure(figsize=(3,5))
    for i in range(len(f)):
        plt.plot(t, wvfm[:,i]/np.abs(wvfm[:,i]).max()/2 +i)
    plt.gca().set_yticks(np.arange(len(f)))
    plt.gca().set_yticklabels([f'{fv:.3g} Hz' for fv in f])
    plt.xlabel('Time (s)')
    #plt.xlim(0, 4)

def ptts2ps(phase_tts):
    dist = []
    dphase = []
    for phase_tt in phase_tts:
        dphase.append(\
            (phase_tt['t'] - phase_tt['t0'])\
            * 2 * np.pi * phase_tt['f'])
        dist.append(phase_tt['dist'])
    dphase = np.array(dphase)
    dist = np.array(dist)
    f = phase_tt['f']

    return f, dist, dphase

class Figure:
    def __init__(self, fig):
        self.fig = fig

    def savefig(self, figname, dpi=300):
        self.fig.savefig(figname, dpi=dpi, bbox_inches='tight')

def plot_example_for_paper():
    f0 = 5
    gamma = 0.5
    sw = surface_wave_for_paper(dist=2.5)
    c0 = np.interp(f0, sw['f'], sw['c'])
    t0_theo = phaseshift.ptt_theo(sw, f0, c0)
    seis_filtered = phaseshift.gaussian_filter(sw,
        f=f0, gamma=gamma)
    t0 = phaseshift.closest_maximum_accurate(t0_theo,
        sw['t'], seis_filtered)

    fig = plt.figure(figsize=yplot.a4)
    gs = fig.add_gridspec(3,2,left=0.15, right=0.45, bottom=0.6, top=0.95,
        width_ratios=(0.5,1), wspace=0.6,
        height_ratios=(1,1,1), hspace=0.5)
    ax1 = fig.add_subplot(gs[0,0])
    m = DB
    depth, _, vs, _ = m.stepwise()
    ax1.plot(vs, depth, '-k', lw=0.5)
    yplot.axis_params((2,4), 'Vs (km/s)', (5,0), 'Depth (km)', ax=ax1)
    yplot.ticklocator_params(1, 0.5, 1, 0.2, ax=ax1)

    ax_c = fig.add_subplot(gs[0,1])
    f, _ = example_para2_for_paper()
    c = cR(f)
    ax_c.plot(f, c)
    ax_c.plot(f0, c0, 'o', ms=5,
            mfc='r',
            mec='k', mew=0.25)

    ax_c.set_xscale('log')
    #ax_c.yaxis.set_label_position('right')
    #ax_c.xaxis.set_label_position('top')
    #ax_c.tick_params(right=True, labelright=True,
    #    labelleft=False, labeltop=True, labelbottom=False)
    ax_c.text(f0, c0,'')

    yplot.axis_params((0.1, 30), 'Frequency (Hz)',
        (2,4), 'Phase velocity (km/s)', ax=ax_c)
    yplot.ticklocator_params(None, None, 1, 0.2, ax=ax_c)


    ax2 = fig.add_subplot(gs[1,:])
    ax2.plot(sw['t'], sw['seis'], '-k', lw=0.5)
    yplot.axis_params((0,3), 'Time (s)', (-2.5,2.5), ax=ax2)
    yplot.ticklocator_params(0.5, 0.1, 1, 0.2, ax=ax2)
    ax2.text(0.02,0.98,
        (f'$\Delta$ = {sw["dist"]:g} km\n'
        ),
        ha='left', va='top', transform=ax2.transAxes)

    ax3 = fig.add_subplot(gs[2,:])
    ax3.plot(sw['t'], seis_filtered, '-r', lw=0.5)
    #print(t0, t0_theo)
    ax3.text(0.02,0.98,
        (f'$f_c =${f0} Hz\n'
        f'$\\alpha=${gamma**2 * 2*np.pi*f0:.3g}\n'
        f'$d\phi=${(t0-t0_theo)* (2*np.pi*f0):.3g}\n'
        ),
        ha='left', va='top', transform=ax3.transAxes)
    #print((t0-t0_theo)* (2*np.pi*f0))
    ax3.axvline(t0, color='r')
    ax3.axvline(t0_theo)
    yplot.axis_params((0.8,1.2), 'Time (s)', (-1.2, 1.2), ax=ax3)
    yplot.ticklocator_params(0.1, 0.02, 0.5, 0.1, ax=ax3)

    yplot.labelax([ax1, ax_c, ax2, ax3], loc='lower left')

    #yplot.auxplot()

    return Figure(fig)

def plot_example_for_paper_v2():
    f0 = 5
    gamma = 0.5
    sw = surface_wave_for_paper(dist=2.5)
    c0 = np.interp(f0, sw['f'], sw['c'])
    t0_theo = phaseshift.ptt_theo(sw, f0, c0)
    seis_filtered = phaseshift.gaussian_filter(sw,
        f=f0, gamma=gamma)
    t0 = phaseshift.closest_maximum_accurate(t0_theo,
        sw['t'], seis_filtered)

    fig = plt.figure(figsize=yplot.a4)
    gs = fig.add_gridspec(3,2,left=0.15, right=0.45, bottom=0.6, top=0.95,
        width_ratios=(0.5,1), wspace=0.6,
        height_ratios=(1,1,1), hspace=0.5)
    ax1 = fig.add_subplot(gs[0,0])
    m = DB
    depth, _, vs, _ = m.stepwise()
    ax1.plot(vs, depth, '-k', lw=0.5)
    yplot.axis_params((2,4), 'Vs (km/s)', (5,0), 'Depth (km)', ax=ax1)
    yplot.ticklocator_params(1, 0.5, 1, 0.2, ax=ax1)

    ax_c = fig.add_subplot(gs[0,1])
    f, _ = example_para2_for_paper()
    c = cR(f)
    ax_c.plot(f, c)
    ax_c.plot(f0, c0, 'o', ms=5,
            mfc='r',
            mec='k', mew=0.25)

    ax_c.set_xscale('log')
    #ax_c.yaxis.set_label_position('right')
    #ax_c.xaxis.set_label_position('top')
    #ax_c.tick_params(right=True, labelright=True,
    #    labelleft=False, labeltop=True, labelbottom=False)
    ax_c.text(f0, c0,'')

    yplot.axis_params((0.1, 30), 'Frequency (Hz)',
        (2,4), 'Phase velocity (km/s)', ax=ax_c)
    yplot.ticklocator_params(None, None, 1, 0.2, ax=ax_c)


    ax2 = fig.add_subplot(gs[1,:])
    ax2.plot(sw['t'], sw['seis'], '-k', lw=0.5)
    yplot.axis_params((0,3), 'Time (s)', (-2.5,2.5), ax=ax2)
    yplot.ticklocator_params(0.5, 0.1, 1, 0.2, ax=ax2)
    ax2.text(0,1,
        (f'Broadband synthetic ($\Delta$ = {sw["dist"]:g} km)'
        ),
        ha='left', va='top', transform=yplot.offset_transform(2/72, -2/72, transform=ax2.transAxes))

    ax3 = fig.add_subplot(gs[2,:])
    ax3.plot(sw['t'], seis_filtered, '-r', lw=0.5)
    #print(t0, t0_theo)
    ax3.text(0.02,0.98,
        (f'$f_c =${f0} Hz\n'
        f'$\gamma =${gamma:.3g} ($\\alpha=${gamma**2 * 2*np.pi*f0:.3g})\n'
        f'$d\phi=${(t0-t0_theo)* (2*np.pi*f0):.3g}\n'
        ),
        ha='left', va='top', transform=ax3.transAxes)
    #print((t0-t0_theo)* (2*np.pi*f0))
    ax3.axvline(t0, color='r')
    ax3.axvline(t0_theo)
    yplot.axis_params((0.8,1.2), 'Time (s)', (-1.2, 1.2), ax=ax3)
    yplot.ticklocator_params(0.1, 0.02, 0.5, 0.1, ax=ax3)

    yplot.labelax([ax1, ax_c, ax2, ax3], loc='lower left')

    #yplot.auxplot()

    return Figure(fig)

def plot_ps_pcolormesh_for_paper(phase_tts, cmap='jet'):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.arange(-0.25, 0.251, 0.125))
    ax_cb.set_yticklabels(['-$\pi$/4','-$\pi$/8','0','$\pi$/8','$\pi$/4'])
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    ax1 = ax.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    return Figure(fig)

def plot_ps_pcolormesh_for_paper_smaller_range_ax(f, dist, phase_tts, cmap='jet',
        vmin=-np.pi/30, vmax=np.pi/30, ax=None, ax_cb=None):
    if ax is None:
        ax = plt.gca()

    dphase = phase_tts * 2 * np.pi * f.reshape(1,-1)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xscale('log')

    if ax_cb is not None:
        cb = plt.colorbar(im, cax=ax_cb)
        cb.set_label('Phase shift (radian)')
        cb.set_ticks(np.pi * np.array([-1/30, -1/60, 0, 1/60, 1/30]))
        ax_cb.set_yticklabels(['-$\pi$/30','-$\pi$/60','0','$\pi$/60','$\pi$/30'])
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    ax1 = ax.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    return ax


def plot_ps_pcolormesh_for_paper_smaller_range(phase_tts, cmap='jet',
        vmin=-np.pi/30, vmax=np.pi/30):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xscale('log')

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.array([-1/30, -1/60, 0, 1/60, 1/30]))
    ax_cb.set_yticklabels(['-$\pi$/30','-$\pi$/60','0','$\pi$/60','$\pi$/30'])
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    ax1 = ax.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    return Figure(fig)

def plot_nearfield_finite_frequency_for_paper(f, dist, ptts_theo, ptts_sf, ptts_ag):
    left, right, bottom, top = 0.15, 0.35, 0.80, 0.91
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=left, right=right, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_smaller_range_ax(f, dist, ptts_ag-ptts_theo, cmap='jet', ax=axs[0], ax_cb=None)
    axs[1].axis('off')
    axs[0].set_title('a. Finite frequency + near field',
            fontweight='bold', fontsize=9)

    xoffset = 0.23
    axs2 = fig.subplots(1,2,gridspec_kw=dict(left=left+xoffset, right=right+xoffset, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_smaller_range_ax(f, dist, ptts_ag-ptts_sf, cmap='jet', ax=axs2[0], ax_cb=None)
    axs2[1].axis('off')
    axs2[0].set_ylabel('')
    axs2[0].tick_params(labelleft=False)
    axs2[0].set_title('b. Finite frequency',
            fontweight='bold', fontsize=9)

    xoffset = 0.46
    axs3 = fig.subplots(1,2, gridspec_kw=dict(left=left+xoffset, right=right+xoffset, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_smaller_range_ax(f, dist, ptts_sf-ptts_theo, cmap='jet', ax=axs3[0], ax_cb=axs3[1])
    axs3[0].set_ylabel('')
    axs3[0].tick_params(labelleft=False)
    axs3[0].set_title('c. Near field',
            fontweight='bold', fontsize=9)

    #yplot.labelax([axs[0], axs2[0], axs3[0]])


    return Figure(fig)


def plot_ps_pcolormesh_for_single_frequency_smaller_range(phase_tts, cmap='jet',
        vmin=-np.pi/30, vmax=np.pi/30):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    #dphase[dphase>0] = np.nan
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xscale('log')

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.array([-1/30, -1/60, 0, 1/60, 1/30]))
    ax_cb.set_yticklabels(['-$\pi$/30','-$\pi$/60','0','$\pi$/60','$\pi$/30'])
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    ax1 = ax.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    return Figure(fig)

def d2kdw2(sw):
    omega = sw['f'] * 2 * np.pi
    k = omega / sw['c']
    domega = omega[1] - omega[0]
    return yutils.diff(k, domega, n=2)

def d2kdw2_smooth(sw, every=5):
    f_samples = sw['f'][::every]
    c_samples = np.interp(f_samples, sw['f'], sw['c'])
    from scipy.interpolate import interp1d
    f = interp1d(f_samples, c_samples, kind='cubic')
    c_interp = f(sw['f'])
    omega = sw['f'] * 2 * np.pi
    k = omega / c_interp
    domega = omega[1] - omega[0]
    return yutils.diff(k, domega, n=2)

def plot_ps_pcolormesh_for_paper_smaller_range_v2(sw, phase_tts, cmap='jet',
        vmin=-np.pi/30, vmax=np.pi/30):
    fig, axs = plt.subplots(2,2, figsize=yplot.a4,
            gridspec_kw=dict(left=0.15, right=0.4, bottom=0.78, top=0.95,
        width_ratios=(1,0.05), wspace=0.1,
        height_ratios=(0.2,1), hspace=0))
    ax_d2c, ax, ax_cb = axs[0,0], axs[1,0], axs[1,1]
    axs[0,1].axis('off')

    ddk = d2kdw2_smooth(sw, every=10)
    ax_d2c.plot(1/sw['f'], ddk, '-k', lw=0.5)
    ax_d2c.fill_between(1/sw['f'], ddk, y2=0,
            where=ddk>0, color='salmon')
    ax_d2c.fill_between(1/sw['f'], ddk, y2=0,
            where=ddk<0, color='lightblue')
    #ax_d2c.plot(1/sw['f'], d2kdw2_smooth(sw, every=10),
    #        '-r', lw=0.5)
    ax_d2c.set_xscale('log')
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    ax_d2c.set_xticks(xticks)
    ax_d2c.xaxis.set_major_formatter(plt.NullFormatter())
    ax_d2c.set_xlim(1/25, 1)
    ax_d2c.set_ylim(-0.01, 0.01)

    ax1 = ax_d2c.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xscale('log')

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.array([-1/30, -1/60, 0, 1/60, 1/30]))
    ax_cb.set_yticklabels(['-$\pi$/30','-$\pi$/60','0','$\pi$/60','$\pi$/30'])
    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels(['1', '2', '3', ''])
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')


    return Figure(fig)

def ptts2dcc(phase_tts):
    dist = []
    dc_over_c = []
    for phase_tt in phase_tts:
        c0 = phase_tt['dist'] / (phase_tt['t0'] + 1  / (8*phase_tt['f']))
        c = phase_tt['dist'] / (phase_tt['t'] + 1  / (8*phase_tt['f']))
        dc_over_c.append((c-c0)/c0)
        dist.append(phase_tt['dist'])
    dc_over_c = np.array(dc_over_c)
    dist = np.array(dist)
    f = phase_tt['f']

    return f, dist, dc_over_c

def plot_dcc_pcolormesh_for_paper(phase_tts, cmap='jet',
        vmin=-0.01, vmax=0.01):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dcc = ptts2dcc(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dcc[:,::-1],
            cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xscale('log')

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('$\delta$c/c (%)', labelpad=-5)
    ax_cb.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax_cb.yaxis.set_minor_locator(plt.MultipleLocator(0.001))
    ax_cb.yaxis.set_major_formatter(plt.FuncFormatter(\
            lambda x, pos: '%g' % (x*100)))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    xticks = [0.04, 0.1, 0.2, 0.4, 1.0]
    xticklabels = ['%g' % xtick for xtick in xticks]
    ax.set_xticks(xticks)
    #ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(xticklabels)
    #ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(1/25, 1)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    ax1 = ax.secondary_xaxis('top',
        functions=(lambda x: 1/x, lambda x: 1/x))
    ax1.set_xlabel('Frequency (Hz)', labelpad=5)
    xticks1 = [1/xtick for xtick in xticks[::-1]]
    ax1.set_xticks(xticks1)
    xticklabels1 = ['%g' % xtick for xtick in xticks1]
    ax1.set_xticklabels(xticklabels1, va='baseline')

    return Figure(fig)

def ptts_sf_for_paper(gamma=None, dist=2.5,
        f=5, iridge=0, ngamma=101):
    if gamma is None:
        gamma = np.logspace(np.log10(0.1),
                np.log10(10), ngamma)

    sw = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw['f'], sw['c'])
    t0_theo = phaseshift.ptt_theo(sw, f,
        c=c0, iridge=iridge)

    phase_tts = []
    t0 = t0_theo
    for each_gamma in gamma[::-1]:
        phase_tt = phaseshift.ptt_sf(sw, f, t0, each_gamma)
        #print(f'debug: t0 = {t0:.3g} s, phase_tt[0] = {phase_tt[0]:.3g} s', flush=True)
        phase_tts.append(phase_tt[0])
        t0 = phase_tt[0]
    phase_tts = np.array(phase_tts[::-1])

    res = {'gamma': gamma, 'ptt': phase_tts,
            'f0': f,
            'c0': c0, 't0': t0_theo}
    return res

def ptts_sf_meier_for_paper(gamma=None, dist=2.5,
        f=5, iridge=0, ngamma=101):
    if gamma is None:
        gamma = np.logspace(np.log10(0.1),
                np.log10(10), ngamma)

    sw = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw['f'], sw['c'])
    #print(sw)
    t0_theo = phaseshift.ptt_theo(sw, f, c0,
        iridge=iridge)
    
    ptts = []
    for i, each_gamma in enumerate(gamma):
        res = phaseshift.ptt_sf_meier2(sw, f,
                t0_theo,
                gamma=each_gamma)
        ptts.append(res['ptt'])
    ptts = np.array(ptts)

    return {'gamma': gamma, 'ptt': ptts,
            'f0': f,
            'c0': c0, 't0': t0_theo}

def plot_ptts_sf_phaseshift_list_for_paper(phase_tts_sf_list,
        labels=None):

    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']

    fig, ax = plt.subplots(1, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        ))
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list)))
    colors = plt.get_cmap('tab10')(np.linspace(0,1,11))[:4]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-0.5,0.50001,0.125/4)*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/4, np.pi/4)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.6, 0.8, 0.4),
            bbox_transform=ax.transAxes)

    return Figure(fig)

def plot_ptts_sf_phaseshift_list_for_paper_v2(phase_tts_sf_list,
        labels=None):

    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']

    fig, ax = plt.subplots(1, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        ))
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list)))
    colors = plt.get_cmap('tab10')(np.linspace(0,1,11))[:4]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-1/16,1/16+1e-10,1/(16*8))*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-1/16,1/16+1e-10,1/32)*np.pi)
    ax.set_yticklabels(['-$\pi/16$', '-$\pi/32$', '$0$', '$\pi/32$', '$\pi/16$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/16, np.pi/16)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='center',
            bbox_to_anchor=(0.1, 0.0, 0.8, 0.5),
            bbox_transform=ax.transAxes)

    return Figure(fig)
def plot_ptts_sf_phaseshift_lists_for_paper(phase_tts_sf_list1,
        phase_tts_sf_list2,
        labels=None):

    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']

    fig, axs = plt.subplots(1,2, figsize=yplot.a4,
            gridspec_kw=dict(left=0.15, right=0.8,
                             bottom=0.80, top=0.95,
                             wspace=0.5)
            )
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list1)))
    colors = plt.get_cmap('tab10')(np.linspace(0,1,11))[:4]
    
    ax = axs[0]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list1):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-0.5,0.50001,0.125/4)*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/4, np.pi/4)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.6, 0.8, 0.4),
            bbox_transform=ax.transAxes)
    ax.set_title('Time domain method')
    
    
    ax = axs[1]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list2):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-0.5,0.50001,0.125/4)*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/4, np.pi/4)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.6, 0.8, 0.4),
            bbox_transform=ax.transAxes)
    ax.set_title('Frequency domain method')

    return Figure(fig)


def plot_ptts_sf_phaseshift_lists_for_paper_v2(phase_tts_sf_list1,
        phase_tts_sf_list2,
        labels=None):
    
    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']

    fig, axs = plt.subplots(1,2, figsize=yplot.a4,
            gridspec_kw=dict(left=0.15, right=0.8,
                             bottom=0.80, top=0.95,
                             wspace=0.5)
            )
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list1)))
    colors = plt.get_cmap('tab10')(np.linspace(0,1,11))[:4]
    
    ax = axs[0]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list1):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-1/16,1/16+0.00001,1/64)*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-1/16,1/16+0.00001,1/32)*np.pi)
    ax.set_yticklabels(['-$\pi/16$', '-$\pi/32$', '$0$', '$\pi/32$', '$\pi/16$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/16, np.pi/16)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.05, 0.8, 0.4),
            bbox_transform=ax.transAxes)
    ax.set_title('Time domain method')
    
    
    ax = axs[1]
    for i, phase_tts_sf in enumerate(phase_tts_sf_list2):
        if labels is None:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=f'{1/phase_tts_sf["f0"]:g} s',
                )
        else:
            ax.plot(phase_tts_sf['gamma'],
                ptt2ps(phase_tts_sf['ptt']),
                lw=0.5, color=colors[i],
                label=labels[i],
                )

    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$ ($\\alpha=2\\pi f\\gamma^2$)')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-1/16,1/16+0.00001,1/64)*np.pi,
        minor=True)
    ax.set_yticks(np.arange(-1/16,1/16+0.00001,1/32)*np.pi)
    ax.set_yticklabels(['-$\pi/16$', '-$\pi/32$', '$0$', '$\pi/32$', '$\pi/16$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(-np.pi/16, np.pi/16)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.05, 0.8, 0.4),
            bbox_transform=ax.transAxes)
    ax.set_title('Frequency domain method')

    yplot.labelax(axs, loc='upper right')

    return Figure(fig)


def plot_simplest_phaseshift_for_paper_v8(dist=3):
    #c_factor = np.linspace(0.9, 1.1, 101)
    #df_example = 0.01
    #c_factor_example = 1.04
    #c_factor_example = 1.1

    #t0_single, t0_sum = simplest_phaseshift_for_paper(\
    #        dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    #t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, t0_single, t0_sum = simplest_phaseshift_for_paper(dist=dist, fc=fc, df=0.01,
    #        c_factor=c_factor_example)
    
    f = np.array([0.99, 1., 1.01])
    c = np.array([1.5, 1.4, 1.3])
    c1 = np.array([1.5, 1.4, 1.4])

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))
    #print(t0_single[0,0])

    ax = axs[0]
    #cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    #dcdf = (cref[2] - cref[0]) / (0.2)
    #freqs_example = np.array([fc-df_example, fc, fc+df_example])
    #c0 = cref[1] - dcdf * df_example
    #c1 = np.array([c0, cref[1],
            #cref[1] + dcdf * df_example])
    #c1 = np.array([1.5, 1.4, 1.3])
    #c2 = np.array([c1[0], c1[1],
    #            c1[2]*c_factor_example])
    ax.plot(f, c, '-k', lw=0.5)
    ax.plot(f, c1, '-r', lw=0.5)
    for ftmp in f:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (1.2, 1.6), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]
    import matplotlib.transforms as mtransforms
    
    def plot_single_frequency_and_sum(t, f, c, dist, color='k'):
        wvfm = synthetic.single_frequencies_surface_wave(f, c, t, dist)
        wvfm_sum = np.sum(wvfm, axis=1)
        nf = len(f)
        for i in range(nf):
            ax.plot(t, wvfm[:,i] / np.abs(wvfm[:,i]).max() + i,
                   '-', color=color, lw=0.5)
            idx = np.argmin(np.abs(t - (-0.5)))
            ax.text(0, wvfm[idx,i] /np.abs(wvfm[:,i]).max() + i, f'{f[i]:.2f} Hz',
                    ha='right', va='center',
                    #transform=ax.get_yaxis_transform(),
                    transform=ax.get_yaxis_transform()+ \
                            mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                    )

        ax.plot(t, wvfm_sum / np.abs(wvfm_sum).max() + 3, '-',
                color=color,
                lw=1)
        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm_sum[idx] /np.abs(wvfm_sum).max() + 3, f'Sum',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                    )
        return wvfm, wvfm_sum
        
    _, t = example_para2_for_paper()
    wvfm, wvfm_sum = plot_single_frequency_and_sum(t, f, c, dist)
    
    t0 = phaseshift.closest_maximum_accurate(2.0,
            t, wvfm[:,1])
    peak0 = np.interp(t0, t, wvfm[:,1]/np.abs(wvfm[:,1]).max()+1)
    t_sum = phaseshift.closest_maximum_accurate(2.0,
            t, wvfm_sum)
    ax.axvline(t0, dashes=(5,5), lw=1)
    ax.plot(t0, peak0, 'D', ms=5, mec='k', mfc='lightgray')
    peak_sum = np.interp(t_sum, t, wvfm_sum /np.abs(wvfm_sum).max() + 3)
    ax.annotate('', xy=(t_sum, peak_sum),
               xytext=(t_sum, peak_sum+1),
               arrowprops=dict(arrowstyle='-|>',
                              shrinkA=0, shrinkB=0,
                              lw=1))
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)
    

    ax = axs[2]
    
    wvfm, wvfm_sum = plot_single_frequency_and_sum(t, f, c1, dist,
                                                  color='r')
    t0 = phaseshift.closest_maximum_accurate(2.0,
            t, wvfm[:,1])
    peak0 = np.interp(t0, t, wvfm[:,1]/np.abs(wvfm[:,1]).max()+1)
    
    t_sum = phaseshift.closest_maximum_accurate(2.0,
            t, wvfm_sum)
    ax.axvline(t0, dashes=(5,5), lw=1)
    ax.plot(t0, peak0, 'D', ms=5, mec='k', mfc='red')

    peak_sum = np.interp(t_sum, t, wvfm_sum /np.abs(wvfm_sum).max() + 3)
    ax.annotate('', xy=(t_sum, peak_sum),
               xytext=(t_sum, peak_sum+1),
               arrowprops=dict(arrowstyle='-|>',
                              shrinkA=0, shrinkB=0,
                              lw=1))
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    yplot.labelax(axs, loc='upper left')

    return Figure(fig)

def plot_debug_simplest_phaseshift():
    dist = 3
    freqs = np.array([0.9, 1, 1.1])
    #c = np.array([2.88, 2.85, 2.82])
    #c_nl = np.array([2.88, 2.85, 2.85])
    #c = np.array([2.5, 2.8, 3.1])
    #c_nl = np.array([2.5, 2.8, 3.1]) 
    c = np.array([2.8, 2.8, 2.8])
    c_nl = np.array([2.8, 2.8, 2.8]) 
    c_nl
    cref = cR(freqs)
    print(cref, c)

    _, t = example_para2_for_paper()

    t_colvec = t.reshape(-1,1)

    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*freqs*dist/c) * \
        np.exp(1j * 2*np.pi*freqs*t_colvec)
    wvfm = wvfm.real

    wvfm_nl = 1 * \
        - hankel1(0, - 2*np.pi*freqs*dist/c_nl) * \
        np.exp(1j * 2*np.pi*freqs*t_colvec)
    wvfm_nl = wvfm_nl.real

    print(wvfm.shape)

    plt.figure(figsize=(4,5))
    plt.subplot(221)
    for i in range(len(freqs)):
        plt.plot(t, wvfm[:,i]/np.abs(wvfm[:,i]).max() + i,
                '-k', lw=0.25)
        plt.text(0, i, f'{freqs[i]} Hz',
                transform=plt.gca().get_yaxis_transform())

    for i in range(len(freqs)):
        plt.plot(t, wvfm_nl[:,i]/np.abs(wvfm_nl[:,i]).max() + i,
                '-r', lw=0.25)
    plt.xlim(0.5,1.5)

    plt.subplot(222)
    wvfm_sum = np.sum(wvfm, axis=1)
    wvfm_nl_sum = np.sum(wvfm_nl, axis=1)
    plt.plot(t, wvfm_sum / np.abs(wvfm_sum).max())
    plt.plot(t, wvfm_nl_sum / np.abs(wvfm_nl_sum).max()+1,'-r')
    plt.xlim(0.5, 1.5)

    def plot_tick(t0, t, x, *args, **kwargs):
        x0 = np.interp(t0, t, x)
        plt.plot(t0, x0, *args, **kwargs)

    def tstar(dist, c):
        return dist/c

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    #print(t0(dist, c[1], freqs[1]))
    t0_theo = t0(dist, c[1], freqs[1])
    t0_measure = phaseshift.closest_maximum_accurate(t0_theo,
            t, wvfm_sum)
    t0_measure_nl = phaseshift.closest_maximum_accurate(t0_theo,
            t, wvfm_nl_sum)
    print(t0_theo, t0_measure, t0_measure_nl)

    plot_tick(t0_theo,
            t, wvfm_sum / np.abs(wvfm_sum).max(),
            'k+', ms=5.5, mfc='lightgray')
    plot_tick(t0_measure,
            t, wvfm_sum / np.abs(wvfm_sum).max(),
            'kD', ms=5.5, mfc='none')

    plot_tick(t0_theo,
            t, wvfm_nl_sum / np.abs(wvfm_nl_sum).max()+1,
            'k+', ms=5.5, mfc='lightgray')
    plot_tick(t0_measure_nl,
            t, wvfm_nl_sum / np.abs(wvfm_nl_sum).max()+1,
            'kD', ms=5.5, mfc='none')

def debug_simplest_phaseshift(dist,
        freqs, c):
    _, t = example_para2_for_paper()
    #t = np.linspace(0.5, 1.5, 3001)

    assert(len(freqs) == 3)
    assert(len(c) == 3)

    t_colvec = t.reshape(-1,1)

    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*freqs*dist/c) * \
        np.exp(1j * 2*np.pi*freqs*t_colvec)
    wvfm = wvfm.real

    #weight = np.array([0.5, 1, 0.5])
    weight = np.array([1, 1, 1])
    wvfm_sum = np.sum(weight*wvfm, axis=1)

    def t0(dist, c, f):
        return dist/c - 1/(8*f)

    t0_theo = t0(dist, c[1], freqs[1])
    t0_measure = phaseshift.closest_maximum_accurate(t0_theo,
            t, wvfm[:,1])
    t0_sum_measure = phaseshift.closest_maximum_accurate(t0_theo,
            t, wvfm_sum)

    return t, wvfm, wvfm_sum, t0_theo, t0_measure, t0_sum_measure

def plot_debug_simplest_phaseshift_df(dist, fc=1):
    dfs = np.arange(0, 0.01000001, 0.001)
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)

    t0_theos = []
    t0_measures = []
    t0_theos1 = []
    t0_measures1 = []
    t0_theos2 = []
    t0_measures2 = []
    for df in dfs:
        freqs = np.array([fc-df, fc, fc+df])
        c = np.array([cref[1] - dcdf * df, cref[1],
            cref[1] + dcdf * df])
        c0 = cref[1] - dcdf * df
        c1 = np.array([c0, cref[1],
            freqs[2] / (2*freqs[1]/cref[1] - freqs[0]/c0)])
        c2 = np.array([cref[1], cref[1], cref[1]])
        #print(c[2], c1[2])

        t, wvfm, wvfm_sum, t0_theo, t0_measure, t0_sum_measure = \
                debug_simplest_phaseshift(dist,
                freqs, c)
        t0_theos.append(t0_measure)
        t0_measures.append(t0_sum_measure)

        t1, wvfm1, wvfm_sum1, t0_theo1, t0_measure1, t0_sum_measure1  = \
                debug_simplest_phaseshift(dist,
                freqs, c1)
        t0_theos1.append(t0_measure1)
        t0_measures1.append(t0_sum_measure1)

        t2, wvfm2, wvfm_sum2, t0_theo2, t0_measure2, t0_sum_measure2  = \
                debug_simplest_phaseshift(dist,
                freqs, c2)
        t0_theos2.append(t0_measure2)
        t0_measures2.append(t0_sum_measure2)

    t0_theos = np.array(t0_theos)
    t0_measures = np.array(t0_measures)
    t0_theos1 = np.array(t0_theos1)
    t0_measures1 = np.array(t0_measures1)
    t0_theos2 = np.array(t0_theos2)
    t0_measures2 = np.array(t0_measures2)

    plt.figure(figsize=(5,2))
    plt.subplot(121)
    plt.subplot(122)
    print(dfs.shape, t0_theos.shape)
    plt.plot(dfs, t0_measures - t0_theos)
    plt.plot(dfs, t0_measures1 - t0_theos1, '-r')
    plt.plot(dfs, t0_measures2 - t0_theos2, '-g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Travel time residual (s)')


def plot_debug_simplest_phaseshift_df_v2(dist, fc=1):
    dfs = np.arange(0, 0.01, 0.001)
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)

    t0_theos = []
    t0_measures = []
    t0_theos1 = []
    t0_measures1 = []
    t0_theos2 = []
    t0_measures2 = []
    for df in dfs:
        freqs = np.array([fc-df, fc, fc+df])
        c = np.array([cref[1] - dcdf * df, cref[1],
            cref[1] + dcdf * df])
        c0 = cref[1] - dcdf * df
        c1 = np.array([c0, cref[1],
            freqs[2] / (2*freqs[1]/cref[1] - freqs[0]/c0)])
        c2 = np.array([c1[0], c1[1], c1[2]*0.9])
        #print(c[2], c1[2])

        t, wvfm, wvfm_sum, t0_theo, t0_measure, t0_sum_measure = \
                debug_simplest_phaseshift(dist,
                freqs, c)
        t0_theos.append(t0_measure)
        t0_measures.append(t0_sum_measure)

        t, wvfm, wvfm_sum, t0_theo1, t0_measure1, t0_sum_measure1  = \
                debug_simplest_phaseshift(dist,
                freqs, c1)
        t0_theos1.append(t0_measure1)
        t0_measures1.append(t0_sum_measure1)

        t, wvfm, wvfm_sum, t0_theo2, t0_measure2, t0_sum_measure2  = \
                debug_simplest_phaseshift(dist,
                freqs, c2)
        t0_theos2.append(t0_measure2)
        t0_measures2.append(t0_sum_measure2)

    t0_theos = np.array(t0_theos)
    t0_measures = np.array(t0_measures)
    t0_theos1 = np.array(t0_theos1)
    t0_measures1 = np.array(t0_measures1)
    t0_theos2 = np.array(t0_theos2)
    t0_measures2 = np.array(t0_measures2)

    c2_factors = np.linspace(0.8, 1.2, 101)
    t0_theos3 = []
    t0_measures3 = []
    for c2_factor in c2_factors:
        #c2 = np.array([c1[0], c1[1], c1[2]*c2_factor])
        c2 = np.array([c[0], c[1], c[2]*c2_factor])
        t, wvfm, wvfm_sum, t0_theo2, t0_measure2, t0_sum_measure2  = \
                debug_simplest_phaseshift(dist,
                freqs, c2)
        t0_theos3.append(t0_measure2)
        t0_measures3.append(t0_sum_measure2)
    t0_theos3 = np.array(t0_theos3)
    t0_measures3 = np.array(t0_measures3)

    plt.figure(figsize=(4,3))
    plt.subplot(221)
    plt.plot(dfs, t0_measures - t0_theos)
    plt.plot(dfs, t0_measures1 - t0_theos1, '-r')
    plt.xlabel('$\Delta f$ (Hz)')
    plt.ylabel('Travel time\ndifference (s)')

    plt.subplot(222)
    plt.plot(dfs, t0_measures - t0_theos)
    plt.plot(dfs, t0_measures1 - t0_theos1, '-r')
    plt.plot(dfs, t0_measures2 - t0_theos2, '-g')

    plt.xlabel('$\Delta f$ (Hz)')
    plt.ylabel('Travel time\ndifference (s)')
    plt.title('c2 reduced by 10%')

    plt.subplot(223)
    plt.plot(c2_factors, t0_measures3 - t0_theos3, '-g')
    plt.xlabel('c2 factor')
    plt.ylabel('Travel time\ndifference (s)')


    plt.tight_layout()

def simplest_phaseshift(dist, fc=1, df=0.01, c_factor=0.9):
    if np.isscalar(df) and np.isscalar(c_factor):
        single_example = True
    else:
        single_example = False

    if np.isscalar(df):
        df = np.array([df])
    if np.isscalar(c_factor):
        c_factor = np.array([c_factor])

    ndf = len(df)
    nc_factor = len(c_factor)

    t0_single = np.zeros((ndf, nc_factor))
    t0_sum = np.zeros((ndf, nc_factor))
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    for i in range(ndf):
        freqs = np.array([fc-df[i], fc, fc+df[i]])
        c0 = cref[1] - dcdf * df[i]
        #c1 = np.array([c0, cref[1],
        #    freqs[2] / (2*freqs[1]/cref[1] - freqs[0]/c0)])
        c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df[i]])
        #c1 = np.array([1.45, 1.4, 1.35])
        t, wvfm1, wvfm_sum1, t0_theo1, t0_measure1, t0_sum_measure1 = \
                debug_simplest_phaseshift(dist,
                freqs, c1)
        t0_single[i,:] = t0_measure1

        for j in range(nc_factor):
            c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor[j]])

            t, wvfm2, wvfm_sum2, t0_theo2, t0_measure2, t0_sum_measure2  = \
                    debug_simplest_phaseshift(dist,
                    freqs, c2)
            t0_sum[i,j] = t0_sum_measure2

    if single_example:
        return t, wvfm1, wvfm2, wvfm_sum1, wvfm_sum2, t0_single, t0_sum
    else:
        return t0_single, t0_sum

def simplest_phaseshift_for_paper(dist, fc=1, df=0.01, c_factor=0.9):
    if np.isscalar(df) and np.isscalar(c_factor):
        single_example = True
    else:
        single_example = False

    if np.isscalar(df):
        df = np.array([df])
    if np.isscalar(c_factor):
        c_factor = np.array([c_factor])

    ndf = len(df)
    nc_factor = len(c_factor)

    t0_single = np.zeros((ndf, nc_factor))
    t0_sum = np.zeros((ndf, nc_factor))
    dcdf = 0.05/0.01
    for i in range(ndf):
        freqs = np.array([fc-df[i], fc, fc+df[i]])
        c1 = np.array([1.4-dcdf*df[i], 1.4, 1.4+dcdf*df[i]])
        c1 = np.array([1.5, 1.4, 1.3])
        print(f'c1={c1}, freqs={freqs}, dist={dist}')
        t, wvfm1, wvfm_sum1, t0_theo1, t0_measure1, t0_sum_measure1 = \
                debug_simplest_phaseshift(dist,
                freqs, c1)
        t0_single[i,:] = t0_measure1

        for j in range(nc_factor):
            c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor[j]])
            c2 = np.array([1.5, 1.4, 1.4])
            print(f'c2={c2}, freqs={freqs}, dist={dist}')
            t, wvfm2, wvfm_sum2, t0_theo2, t0_measure2, t0_sum_measure2  = \
                    debug_simplest_phaseshift(dist,
                    freqs, c2)
            t0_sum[i,j] = t0_sum_measure2

    if single_example:
        return t, wvfm1, wvfm2, wvfm_sum1, wvfm_sum2, t0_single, t0_sum
    else:
        return t0_single, t0_sum

def plot_simplest_phaseshift_for_paper(dist=3, fc=1):
    c_factor = np.linspace(0.8, 1.2, 101)
    df_example = 0.01
    c_factor_example = 1.05

    t0_single, t0_sum = simplest_phaseshift(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85))

    ax = axs[0]
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    c0 = cref[1] - dcdf * df_example
    c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df_example])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (2.8, 3.0), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
                '-k', lw=0.5)
        ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
            lw=1)
    ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    yplot.axis_params((-0.5, 0.5), 'Time (s)', (-1.5, 4),
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot(c_factor, (t0_sum-t0_single)[-1,:], '-k', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot(c_factor_example, t_offset, 'kD', ms=3,
        mfc='r')
    ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
            transform=yplot.offset_transform(1/72, 1/72, ax=ax))


    yplot.axis_params((0.8, 1.2), 'Phase velocity perturbation',
            None, 'Time offset (s)', ax=ax)

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v2(dist=3, fc=1):
    c_factor = np.linspace(0.995, 1.005, 101)
    df_example = 0.01
    c_factor_example = 1.0005

    t0_single, t0_sum = simplest_phaseshift(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85))

    ax = axs[0]
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    c0 = cref[1] - dcdf * df_example
    c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df_example])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (2.850, 2.858), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]
    tlim = (0.915, 0.935)

    def norm_x(t, x, tlim):
        sel = (t >= tlim[0]) & (t < tlim[1])
        x1 = x[sel] - x[sel][0]
        return t[sel], (x1 / np.abs(x1).max() - 0.5)*2

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        t1, wvfm_norm1 = norm_x(t, wvfm1[:,i], tlim)
        t2, wvfm_norm2 = norm_x(t, wvfm2[:,i], tlim)
        ax.plot(t1, wvfm_norm1 + i,
                '-k', lw=0.5)
        ax.plot(t2, wvfm_norm2 + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t1 - tlim[0]))
        ax.text(0, wvfm_norm1[idx] + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    t1, wvfm_norm1 = norm_x(t, wvfm1_sum, tlim)
    t2, wvfm_norm2 = norm_x(t, wvfm2_sum, tlim)
    ax.plot(t1, wvfm_norm1 + 3, '-k',
            lw=1)
    ax.plot(t2, wvfm_norm2 + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t1 - tlim[0]))
    ax.text(0, wvfm_norm1[idx] + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    print(t0_single[0,0])
    yplot.axis_params(tlim, 'Time (s)', None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot(c_factor, (t0_sum-t0_single)[-1,:], '-k', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot(c_factor_example, t_offset, 'kD', ms=3,
        mfc='r')
    ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
            transform=yplot.offset_transform(1/72, 1/72, ax=ax))


    yplot.axis_params(None, 'Phase velocity perturbation',
            None, 'Time offset (s)', ax=ax)

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v3(dist=3, fc=1):
    c_factor = np.linspace(0.995, 1.005, 101)
    df_example = 0.1
    c_factor_example = 1.005

    t0_single, t0_sum = simplest_phaseshift(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift(dist=dist, fc=fc, df=df_example,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.6))

    ax = axs[0]
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    c0 = cref[1] - dcdf * df_example
    c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df_example])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params(None, 'Frequency (s)',
            None, 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]
    tlim = (0.923-0.05, 0.923+0.05)

    def norm_x(t, x, tlim):
        sel = (t >= tlim[0]) & (t < tlim[1])
        x1 = x[sel] - x[sel][0]
        return t[sel], (x1 / np.abs(x1).max() - 0.5)

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        t1, wvfm_norm1 = norm_x(t, wvfm1[:,i], tlim)
        t2, wvfm_norm2 = norm_x(t, wvfm2[:,i], tlim)
        ax.plot(t1, wvfm_norm1 + i,
                '-k', lw=0.5)
        ax.plot(t2, wvfm_norm2 + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t1 - tlim[0]))
        ax.text(0, wvfm_norm1[idx] + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    t1, wvfm_norm1 = norm_x(t, wvfm1_sum, tlim)
    t2, wvfm_norm2 = norm_x(t, wvfm2_sum, tlim)
    ax.plot(t1, wvfm_norm1 + 3, '-k',
            lw=1)
    ax.plot(t2, wvfm_norm2 + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t1 - tlim[0]))
    ax.text(0, wvfm_norm1[idx] + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    print(t0_single[0,0])
    yplot.axis_params(tlim, 'Time (s)', None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot(c_factor, (t0_sum-t0_single)[-1,:], '-k', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot(c_factor_example, t_offset, 'kD', ms=3,
        mfc='r')
    ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
            transform=yplot.offset_transform(1/72, 1/72, ax=ax))


    yplot.axis_params(None, 'Phase velocity perturbation',
            None, 'Time offset (s)', ax=ax)

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v4(dist=3, fc=1):
    c_factor = np.linspace(0.995, 1.005, 101)
    df_example = 1
    c_factor_example = 1.01

    t0_single, t0_sum = simplest_phaseshift(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift(dist=dist, fc=fc, df=df_example,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85))

    ax = axs[0]
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    c0 = cref[1] - dcdf * df_example
    c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df_example])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params(None, 'Frequency (s)',
            None, 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]
    tlim = (1.193-0.2, 1.193+0.2)

    def norm_x(t, x, tlim):
        sel = (t >= tlim[0]) & (t < tlim[1])
        x1 = x[sel] - x[sel][0]
        return t[sel], (x1 / np.abs(x1).max() - 0.5)

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        t1, wvfm_norm1 = norm_x(t, wvfm1[:,i], tlim)
        t2, wvfm_norm2 = norm_x(t, wvfm2[:,i], tlim)
        ax.plot(t1, wvfm_norm1 + i,
                '-k', lw=0.5)
        ax.plot(t2, wvfm_norm2 + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t1 - tlim[0]))
        ax.text(0, wvfm_norm1[idx] + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    t1, wvfm_norm1 = norm_x(t, wvfm1_sum, tlim)
    t2, wvfm_norm2 = norm_x(t, wvfm2_sum, tlim)
    ax.plot(t1, wvfm_norm1 + 3, '-k',
            lw=1)
    ax.plot(t2, wvfm_norm2 + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t1 - tlim[0]))
    ax.text(0, wvfm_norm1[idx] + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    print(t0_single[0,0])
    yplot.axis_params(tlim, 'Time (s)', None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot(c_factor, (t0_sum-t0_single)[-1,:], '-k', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot(c_factor_example, t_offset, 'kD', ms=3,
        mfc='r')
    ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
            transform=yplot.offset_transform(1/72, 1/72, ax=ax))


    yplot.axis_params(None, 'Phase velocity perturbation',
            None, 'Time offset (s)', ax=ax)

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v5(dist=3, fc=1):
    c_factor = np.linspace(0.9, 1.1, 101)
    df_example = 0.01
    c_factor_example = 1.05

    t0_single, t0_sum = simplest_phaseshift(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))
    print(t0_single[0,0])

    ax = axs[0]
    cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    c0 = cref[1] - dcdf * df_example
    c1 = np.array([c0, cref[1],
            cref[1] + dcdf * df_example])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (2.8, 3.0), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
                '-k', lw=0.5)
        ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
            lw=1)
    ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    yplot.axis_params((t0_single[0,0]-0.5, t0_single[0,0]+0.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot(c_factor, (t0_sum-t0_single)[-1,:], '-k', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot(c_factor_example, t_offset, 'kD', ms=3,
        mfc='r')
    ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
            transform=yplot.offset_transform(1/72, 1/72, ax=ax))


    yplot.axis_params((c_factor[0], c_factor[-1]), 'Phase velocity perturbation',
            None, 'Time offset (s)', ax=ax)

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v6(dist=3, fc=1):
    c_factor = np.linspace(0.9, 1.1, 101)
    df_example = 0.01
    c_factor_example = 1.08

    t0_single, t0_sum = simplest_phaseshift_for_paper(\
            dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, _, _ = simplest_phaseshift_for_paper(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))
    print(t0_single[0,0])

    ax = axs[0]
    #cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    #dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    #c0 = cref[1] - dcdf * df_example
    #c1 = np.array([c0, cref[1],
            #cref[1] + dcdf * df_example])
    c1 = np.array([1.5, 1.4, 1.3])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (1.2, 1.6), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
                '-k', lw=0.5)
        ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
            lw=1)
    ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.axvline(t0_single[0,0], dashes=(5,5), lw=0.5)
    ax.text(t0_single[0,0], 1.0, 'Peak time of centre frequency',
            ha='center', va='bottom',
            transform=ax.get_xaxis_transform())
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    ax.plot((c_factor-1)*100, (t0_sum-t0_single)[-1,:], '-r', lw=0.5)
    t_offset = np.interp(c_factor_example,
        c_factor, (t0_sum-t0_single)[-1,:])
    ax.plot((c_factor_example-1)*100, t_offset, 'kD', ms=3,
        mfc='r')
    #ax.text(c_factor_example, t_offset, f'{c_factor_example:g}, {t_offset:.3g}', ha='left', va='bottom',
    #        transform=yplot.offset_transform(1/72, 1/72, ax=ax))
    ax.annotate(f'{(c_factor_example-1)*100:g}%, {t_offset:.3g} s',
            xy=((c_factor_example-1)*100, t_offset),
            xytext=(0.98, 0.9),
            textcoords='axes fraction',
            ha='right', va='top',
            arrowprops=dict(arrowstyle='-|>',
                lw=0.5, shrinkB=5))


    yplot.axis_params(((c_factor[0]-1)*100, (c_factor[-1]-1)*100), 'Phase velocity perturbation (%)',
            None, 'Time offset (s)', ax=ax)

    yplot.labelax(axs, loc='upper left')

    return Figure(fig)

def plot_simplest_phaseshift_for_paper_v7(dist=3, fc=1):
    c_factor = np.linspace(0.9, 1.1, 101)
    df_example = 0.01
    #c_factor_example = 1.04
    c_factor_example = 1.1

    #t0_single, t0_sum = simplest_phaseshift_for_paper(\
    #        dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, t0_single, t0_sum = simplest_phaseshift_for_paper(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))
    print(t0_single[0,0])

    ax = axs[0]
    #cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    #dcdf = (cref[2] - cref[0]) / (0.2)
    freqs_example = np.array([fc-df_example, fc, fc+df_example])
    #c0 = cref[1] - dcdf * df_example
    #c1 = np.array([c0, cref[1],
            #cref[1] + dcdf * df_example])
    c1 = np.array([1.5, 1.4, 1.3])
    c2 = np.array([c1[0], c1[1],
                c1[2]*c_factor_example])
    ax.plot(freqs_example, c1, '-k', lw=0.5)
    ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in freqs_example:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (1.2, 1.6), 'Phase velocity (km/s)',
            ax=ax)

    ax = axs[1]

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
                '-k', lw=0.5)
        #ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
        #        '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
            lw=1)
    #ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
    #        lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.axvline(t0_single[0,0], dashes=(5,5), lw=0.5)
    #ax.text(t0_single[0,0], 1.0, 'Peak time of centre frequency',
    #        ha='center', va='bottom',
    #        transform=ax.get_xaxis_transform())
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        #ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
        #        '-k', lw=0.5)
        ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{freqs_example[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    #ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
    #        lw=1)
    ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.axvline(t0_single[0,0], dashes=(5,5), lw=0.5)
    ax.axvline(t0_sum[0,0], dashes=(5,5), lw=0.5, color='r')
    #ax.text(t0_single[0,0], 1.0, 'Peak time of centre frequency',
    #        ha='center', va='bottom',
    #        transform=ax.get_xaxis_transform())
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    yplot.labelax(axs, loc='upper left')

    return Figure(fig)

def plot_debug_simplest_phaseshift_for_paper_v7(dist=3, fc=1):
    c_factor = np.linspace(0.9, 1.1, 101)
    df_example = 0.01
    #c_factor_example = 1.04
    c_factor_example = 1.1

    #t0_single, t0_sum = simplest_phaseshift_for_paper(\
    #        dist=dist, df=df_example, fc=fc, c_factor=c_factor)

    #t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, t0_single, t0_sum = simplest_phaseshift_for_paper(dist=dist, fc=fc, df=0.01,
    #        c_factor=c_factor_example)

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))
    #print(t0_single[0,0])

    ax = axs[0]
    #cref = cR(np.array([fc-0.1, fc, fc+0.1]))
    #dcdf = (cref[2] - cref[0]) / (0.2)
    #freqs_example = np.array([fc-df_example, fc, fc+df_example])
    f = np.array([0.99, 1.00, 1.01])
    #c0 = cref[1] - dcdf * df_example
    #c1 = np.array([c0, cref[1],
            #cref[1] + dcdf * df_example])
    c = np.array([1.5, 1.4, 1.3])
    #c1 = np.array([1.5, 1.4, 1.3])
    #c2 = np.array([c1[0], c1[1],
    #            c1[2]*c_factor_example])
    ax.plot(f, c, '-k', lw=0.5)
    #ax.plot(freqs_example, c2, '-r', lw=0.5)
    for ftmp in f:
        ax.axvline(ftmp, dashes=(5,5), lw=0.5)
    yplot.axis_params((0.985, 1.015), 'Frequency (s)',
            (1.2, 1.6), 'Phase velocity (km/s)',
            ax=ax)
    #return

    ax = axs[1]

    _, t = example_para2_for_paper()
    t_colvec = t.reshape(-1,1)
    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*f*dist/c) * \
        np.exp(1j * 2*np.pi*f*t_colvec)
    wvfm = wvfm.real

    weight = np.array([1, 1, 1])
    wvfm_sum = np.sum(weight*wvfm, axis=1)

    import matplotlib.transforms as mtransforms
    for i in range(wvfm.shape[1]):
        ax.plot(t, wvfm[:,i] /np.abs(wvfm[:,i]).max() + i,
                '-k', lw=0.5)
        #ax.plot(t, wvfm[:,i] /np.abs(wvfm[:,i]).max(),
        #        '-k', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm[idx,i] /np.abs(wvfm[:,i]).max() + i, f'{f[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.plot(t, wvfm_sum / np.abs(wvfm_sum).max() + 3, '-k',
            lw=1)
    #ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
    #        lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm_sum[idx] /np.abs(wvfm_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    #ax.axvline(t0_single[0,0], dashes=(5,5), lw=0.5)
    #ax.text(t0_single[0,0], 1.0, 'Peak time of centre frequency',
    #        ha='center', va='bottom',
    #        transform=ax.get_xaxis_transform())
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    ax = axs[2]
    t, wvfm1, wvfm2, wvfm1_sum, wvfm2_sum, t0_single, t0_sum = simplest_phaseshift_for_paper(dist=dist, fc=fc, df=0.01,
            c_factor=c_factor_example)

    import matplotlib.transforms as mtransforms
    for i in range(wvfm1.shape[1]):
        ax.plot(t, wvfm1[:,i] /np.abs(wvfm1[:,i]).max() + i,
                '-k', lw=0.5)
        ax.plot(t, wvfm2[:,i] /np.abs(wvfm2[:,i]).max() + i,
                '-r', lw=0.5)

        idx = np.argmin(np.abs(t - (-0.5)))
        ax.text(0, wvfm1[idx,i] /np.abs(wvfm1[:,i]).max() + i, f'{f[i]:.2f} Hz',
                ha='right', va='center',
                #transform=ax.get_yaxis_transform(),
                transform=ax.get_yaxis_transform()+ \
                        mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    #ax.plot(t, wvfm1_sum / np.abs(wvfm1_sum).max() + 3, '-k',
    #        lw=1)
    ax.plot(t, wvfm2_sum / np.abs(wvfm2_sum).max() + 3, '-r',
            lw=1)
    idx = np.argmin(np.abs(t - (-0.5)))
    ax.text(0, wvfm1_sum[idx] /np.abs(wvfm1_sum).max() + 3, f'Sum',
            ha='right', va='center',
            #transform=ax.get_yaxis_transform(),
            transform=ax.get_yaxis_transform()+ \
                    mtransforms.ScaledTranslation(-1/72, 0, fig.dpi_scale_trans)
                )

    ax.axvline(t0_single[0,0], dashes=(5,5), lw=0.5)
    ax.axvline(t0_sum[0,0], dashes=(5,5), lw=0.5, color='r')
    #ax.text(t0_single[0,0], 1.0, 'Peak time of centre frequency',
    #        ha='center', va='bottom',
    #        transform=ax.get_xaxis_transform())
    yplot.axis_params((1.5, 2.5), 'Time (s)',None,
            left=False, top=False, right=False, ax=ax)

    yplot.labelax(axs, loc='upper left')

    return Figure(fig)

import os
import numpy as np
import matplotlib.pyplot as plt

import pvt as ypvt
from layermodel import Model1D
import disp_xu as disp
import plot as yplot

from . import synthetic
from . import phaseshift


def get_ak135_model_file():
    return os.path.join(os.environ['PSLPATH'],
                        'models/mtak135sph.mod')

def surface_wave_ncf_for_paper(dist, farfield=False,
        casual_branch_only=True):
    f, t = example_para_ak135()
    c = cR_ak135(f)

    seis = synthetic.surface_wave_ncf(f, c, t, dist=dist,
            farfield=farfield,
            casual_branch_only=casual_branch_only)

    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

def surface_wave_ballistic_for_paper(dist,
        casual_branch_only=True):
    f, t = example_para_ak135()
    c = cR_ak135(f)

    seis = synthetic.surface_wave_ballistic(f, c, t, dist=dist,
            casual_branch_only=casual_branch_only)

    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

def cR_ak135(f):
    model_file = get_ak135_model_file()
    m = Model1D.read_cps(model_file)
    c = disp.cR(m, 1/f, mode=0)
    return c

def example_para_ak135():
    fmin = 0.002
    fmax = 0.05
    df = 0.00001
    tmin = -2000
    tmax = 4000
    dt = 1

    f = np.arange(fmin, fmax+0.1*df, df)
    t = np.arange(tmin, tmax+0.1*dt, dt)

    return f, t

def example_ak135_mft_fc(sw_ak135):
    f = np.logspace(np.log10(sw_ak135['f'][0]),
        np.log10(sw_ak135['f'][-1]),
        501)
    c = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    return f, c

class Figure:
    def __init__(self, fig):
        self.fig = fig

    def savefig(self, figname, dpi=300):
        self.fig.savefig(figname, dpi=dpi, bbox_inches='tight')

def plot_illustration(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'r', 'g']

    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7) 
    t_0th, t_0th_idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            refine=True,
            return_index=True)

    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(2,2, left=0.2, right=0.7,
                          top=0.95, bottom=0.8,
                          width_ratios=[1,0.5],
                          hspace=0.4)

    axs = [fig.add_subplot(gs[:,0]),
           fig.add_subplot(gs[0,1]),
           fig.add_subplot(gs[1,1])]

    def plot_marker(x0, x, y, *args, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = plt.gca()

        y0 = np.interp(x0, x, y)
        ax.plot(x0, y0, *args, **kwargs)

    def plot_label(x0, x, y, text, *args, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = plt.gca()

        y0 = np.interp(x0, x, y)
        ax.text(x0, y0, text, *args, **kwargs)

    ax = axs[0]
    # plot two examples of filtered waveforms
    sel_freqs = [0.01, 0.02, 0.03, 0.04]
    for sel_freq in sel_freqs:
        i_sel_freq = np.argmin(np.abs(f - sel_freq))
        waveform_shifted  = mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max() * 0.005 + sel_freq
        ax.plot(sw['t'], waveform_shifted,
               lw=0.5)
        plot_marker(t_theo[i_sel_freq], sw['t'], waveform_shifted, marker='D', ms=2, color=colors[0],
                    ax=ax)
        plot_marker(t_0th[i_sel_freq], sw['t'], waveform_shifted, marker='D', ms=2, color=colors[1],
                    ax=ax)
        plot_label(0, sw['t'], waveform_shifted, f'{sel_freq:g} Hz', ha='right',
                  va='center', ax=ax, fontsize=9,
                  transform=yplot.offset_transform(-10/72, 0, ax=ax, transform=ax.transData))

    waveform_shifted  = sw['seis'] / np.abs(sw['seis']).max() * 0.005 + 0.05
    ax.plot(sw['t'], waveform_shifted, lw=1)
    plot_label(0, sw['t'], waveform_shifted, 'Broadband', ha='right',
               va='center', ax=ax, fontsize=9,
               transform=yplot.offset_transform(-5/72, 0, ax=ax, transform=ax.transData))


    yplot.axis_params((0, 100), 'Time (s)',
            (0.01-0.007, 0.05+0.005), None,
            left=False, top=False, right=False,
            ax=ax)

    # zoom in
    ax = axs[1]
    ylim_axs_1 = (0.95, 0.96)
    #ylim_axs_1 = None
    
    sel_freq = 0.04
    i_sel_freq = np.argmin(np.abs(f - sel_freq))
    waveform_shifted = mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max()
    
    def polyfit_waveform(i_sel_freq, tlim):
        idx = t_0th_idx[i_sel_freq]
        pfit = np.polyfit(sw['t'][idx-1:idx+2], waveform_shifted[idx-1:idx+2], 2)
        pfun = np.poly1d(pfit)
        t = np.linspace(tlim[0], tlim[1], 101)
        x = pfun(t)
        return t, x
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(63.3,64.5))
    ax.plot(*waveform_polyfit, color='k',
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((63.3, 64.5), None,
            ylim_axs_1, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(0.4, 0.1, 0.005, 0.001, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))

    # zoom in
    ax = axs[2]
    ylim_axs_2 = (0.997, 1.001)

    sel_freq = 0.01
    i_sel_freq = np.argmin(np.abs(f - sel_freq))
    waveform_shifted  = mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max()
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(45.5,48.5))
    ax.plot(*waveform_polyfit,
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((45.5,48.5), 'Time (s)',
            ylim_axs_2, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(1, 0.5, 0.005, 0.001, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))


    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.73, bottom=0.66, wspace=0.2))
    
    ax = axs2[0]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
    yplot.axis_params(freq_lim, None,
            (-0.5,0.5), 'Time (s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.5, yminor=0.1,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color=colors[1])
#     ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
#            color=colors[2])
    
    def labelright(ax):
        ax.tick_params(labelleft=False,
                      labelright=True)
        ax.yaxis.set_label_position('right')
    labelright(ax)
    ax.set_yticks([-np.pi/72, 0, np.pi/72])
    ax.set_yticks(np.linspace(-np.pi/72, np.pi/72, 9),
            minor=True)
    ax.set_yticklabels(['$-\\pi$/72', '0', '$\\pi$/72'])
    ax.grid(axis='y', which='both')
    yplot.axis_params(freq_lim, None,
            (-np.pi/72, np.pi/72), 'Phase (radian)', ax=ax)
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.63, bottom=0.56, wspace=0.2))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
    
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, c_0th - c, lw=1,
           color=colors[1])
#     ax.plot(f, c_ag - c, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.03, 0.03), 'Velocity (km/s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.03, yminor=0.01,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs3[1]
    
    
    ax.plot(f, c, lw=1, color=colors[0])
    ax.plot(f, c_0th, lw=1,
            color=colors[1])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (3.7,4.2), 'Phase velocity (km/s)', ax=ax)
    labelright(ax)
    yplot.ticklocator_params(ymajor=0.2, yminor=0.1,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    
    def line(color='k', lw=1):
        from matplotlib.lines import Line2D
        return Line2D([], [], color=color, lw=lw)
    
    def point(color='k', ms=2):
        from matplotlib.lines import Line2D
        return Line2D([], [], color=color, lw=0, marker='D', ms=ms)
    
    axs[0].legend(handles=[point(colors[0]), point(colors[1])],
        labels=['Predicted', 'Measured'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs[0].transAxes,
        loc='lower center',
        ncol=2,
        mode='expand')
    
    axs2[0].legend(handles=[line(colors[0]), line(colors[1])],
        labels=['Predicted', 'Measured'],
        bbox_to_anchor=(0, 1, 1.2, 0.5),
        bbox_transform=axs2[0].transAxes,
        loc='lower center',
        ncol=2,
        mode='expand')
    
    yplot.labelax([axs[0], axs[1], axs[2],
                   axs2[0], axs2[1], axs3[0], axs3[1]], loc='lower left', fontdict=dict(size=9))
    yplot.labelax([axs2[0], axs2[1], axs3[0], axs3[1]],
                  ['Phase travel time residual', 'Phase residual', 'Phase velocity residual', 'Phase velocity'],
                  loc='upper left',
                  fontdict=dict(size=9, weight='normal'))
    
    return Figure(fig)

def plot_fig1(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'k', 'g']

    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)

    def phasett_each_ridge(iridge=0):
        t_theo = phaseshift.ptt_theo(sw, f, c, iridge=iridge)
        f0 = 0.03
        t0 = np.interp(f0, f, t_theo)
        t_eachridge = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0,
                refine=True)
        return t_eachridge

    i_ridges = np.arange(-1,4)
    t_ridges = [phasett_each_ridge(iridge=i) for i in i_ridges]

    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(2,1, left=0.15, right=0.45,
                             top=0.95, bottom=0.70,
                          height_ratios=(0.3, 0.8),
                          hspace=0.5,
            )

    gs1 = fig.add_gridspec(1,1, left=0.15, right=0.45,
            top=0.65, bottom=0.60)

    axs = [fig.add_subplot(gs[0,0]),
           fig.add_subplot(gs[1,0]),
           fig.add_subplot(gs1[0,0])]

    def plot_marker(x0, x, y, *args, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = plt.gca()

        y0 = np.interp(x0, x, y)
        ax.plot(x0, y0, *args, **kwargs)

    def plot_label(x0, x, y, text, *args, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = plt.gca()

        y0 = np.interp(x0, x, y)
        ax.text(x0, y0, text, *args, **kwargs)

    ax = axs[0]
    ax.plot(sw['t'], sw['seis'] / np.abs(sw['seis']).max()/2,
            '-k', lw=0.5)
    i_freq = np.argmin(np.abs(f-0.03))
    seis_filtered = mft[i_freq,:]
    yplot.plot_rect(440,-1.5, 480, -0.5, facecolor='lightgray',
            edgecolor='none',
            ax=ax)
    ax.plot(sw['t'], seis_filtered / np.abs(seis_filtered).max()/2-1,
           '-k', lw=0.5)

    ax.text(1, -1, f'0.03 Hz, $\\alpha$={2*np.pi*0.03*gamma**2:.2g}', ha='right', va='bottom',
            fontsize=9,
           transform=ax.get_yaxis_transform())
    ax.text(1, 0, 'Broadband', ha='right', va='bottom',
            fontsize=9,
           transform=ax.get_yaxis_transform())
    ax.annotate('Fig.1c', xytext=(320,-1), xy=(440,-1),
            textcoords=yplot.offset_transform(0,6/72,ax=ax,transform=ax.transData),
            xycoords=yplot.offset_transform(0,6/72,ax=ax,transform=ax.transData),
            ha='left', va='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', shrinkA=0,
                shrinkB=0, lw=0.5),
            )

    def mark_ridges():
        sel_f = 0.03
        ts = [np.interp(sel_f, f, t_ridge) for t_ridge in t_ridges]
        for i in range(len(ts)):
            if i == 1:
                ax.plot([ts[i], ts[i]], [-1.6, -0.4], lw=0.5, color='k')
            else:
                ax.plot([ts[i], ts[i]], [-1.6, -0.4], lw=0.5, color='k', dashes=(5,3))

    mark_ridges()

    yplot.axis_params((300, 800), 'Time (s)',
            (-1.7, 0.7), None,
            left=False, top=False, right=False,
            ax=ax)
    #yplot.ticklocator_params(None, None, 0.01, 0.005, ax=ax)

    ax = axs[1]

    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax, rasterized=True)
    for i_ridge, t_ridge in zip(i_ridges, t_ridges):
        if i_ridge == 0:
            ax.plot(t_ridge, f, color=colors[1], lw=1)
        else:
            ax.plot(t_ridge, f, color=colors[1], lw=1, dashes=(5,3))

    def plot_interval(sel_f, i1, i2):
        i_sel_f = np.argmin(f-sel_f)
        ts = [np.interp(sel_f, f, t_ridge) for t_ridge in t_ridges]
        ax.annotate('', xy=(ts[i2], sel_f), xytext=(ts[i1], sel_f),
                   arrowprops=dict(arrowstyle='<-', linewidth=1,
                                  color='r', shrinkA=0, shrinkB=0))

    plot_interval(sel_f=0.03, i1=1, i2=2)
    plot_interval(sel_f=0.032, i1=1, i2=3)
    plot_interval(sel_f=0.034, i1=1, i2=4)
    plot_interval(sel_f=0.03, i1=1, i2=0)

    ts = [np.interp(0.03, f, t_ridge) for t_ridge in t_ridges]
    ax.text(0.5*(ts[1]+ts[2]), 0.03, '$1/f$\n(2$\pi$)', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax, transform=ax.transData))
    ax.text(0.5*(ts[2]+ts[3]), 0.032, '$2/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax, transform=ax.transData))
    ax.text(0.5*(ts[3]+ts[4]), 0.034, '$3/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax, transform=ax.transData))
    ax.text(0.5*(ts[0]+ts[1]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax, transform=ax.transData))


    yplot.axis_params((400,550), 'Time (s)',
            (0.01, 0.04), 'Frequency (Hz)',
            ax=ax)
    yplot.ticklocator_params(50, 10, 0.01, 0.005, ax=ax)

    ax = axs[2]
    ax.plot(sw['t'], seis_filtered / np.abs(seis_filtered).max()/2,
           '-k', lw=0.5)

    c0 = np.interp(0.03, f, c)
    phase_t0 = sw['dist'] / c0
    def mark_vline(x0, *args, **kwargs):
        ax.plot([x0, x0], [-0.5, 0.5], *args, **kwargs)
    mark_vline(phase_t0, '-r', lw=1)

    phase_tobs = np.interp(0.03, f, t_ridges[1])
    mark_vline(phase_tobs, '-k', lw=1)

    ax.text(phase_t0, -0.5, 't = $\Delta$/c',
            ha='right', va='bottom',
            fontsize=9, color='r',
            transform=yplot.offset_transform(-2/72,1/72,
                ax=ax,transform=ax.transData))
    ax.text(phase_tobs, -0.5, 't$_{obs}$',
            ha='left', va='bottom',
            color='k',
            fontsize=9,
            transform=yplot.offset_transform(2/72,1/72,
                ax=ax,transform=ax.transData))

    ax.text(0.5*(phase_t0+phase_tobs), 0.5,
            '$f/8 + \delta t$ ($\pi$/4 + $\delta\phi$)', ha='center',
            va='bottom',
            fontsize=9)
    ax.annotate('', xytext=(phase_tobs,0.45),
            xy=(phase_t0,0.45),
            fontsize=9,
            arrowprops=dict(arrowstyle='<|-',
                shrinkA=0, shrinkB=0, lw=0.5))
    ax.text(480, 0.45, '$\delta\phi$ - Phase residual',
            fontsize=9,
            ha='right', va='top')

    yplot.axis_params((440, 480), 'Time (s)',
            None, None,
            left=False, top=False, right=False,
            ax=ax)


    yplot.labelax(axs, loc='upper left', fontdict=dict(size=9))


    return Figure(fig)

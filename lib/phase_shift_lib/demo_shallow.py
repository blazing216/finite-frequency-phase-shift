import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from tqdm import tqdm

import plot as yplot
import utils as yutils
from layermodel import Model1D
import disp_xu as disp

from . import synthetic, phaseshift

DB = Model1D.from_thk_vs(\
    thk=[0.15, 0.5, 1.35, 8.0, 0.0],
    vs=[2.44, 2.88, 3.25, 3.50, 3.70])

def surface_wave_ncf_for_paper(dist, para_func=None,
        farfield=False, casual_branch_only=True):
    if para_func is None:
        para_func = example_para_for_paper
    f, t = para_func()
    c = cR(f)

    seis = synthetic.surface_wave_ncf(f, c, t, dist=dist,
            farfield=farfield,
            casual_branch_only=casual_branch_only)

    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

def example_para_for_paper():
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

def ptts_for_paper_to_file(outfile, gamma=1, force=False, **kwargs):
    if not force:
        print('Use force=True to run. Notice that it might take a few minutes', flush=True)
        return

    res = ptts_for_paper(gamma, **kwargs)
    pickle.dump(res,
            open(outfile, 'wb'))
    print(outfile, flush=True)

def ptts_for_paper(gamma=1, **kwargs):
    dist = np.arange(0.5, 4.001, 0.1)
    return ptts(dist, gamma, **kwargs)

def ptts(dist, gamma, **kwargs):
    res = []
    ndist = len(dist)
    for i in tqdm(range(ndist)):
        d = dist[i]
        sw = surface_wave_ncf_for_paper(d)
        phase_tt = ptt(sw, gamma=gamma, **kwargs)
        phase_tt.update({'dist': d, 'gamma': gamma})
        res.append(phase_tt)
    return res

def ptt(sw, gamma, **kwargs):
    f, c = example_mft_fc_for_paper(sw)
    ptt = phaseshift.ptt_ag(sw, f, c,
        gamma=gamma, emin=7, **kwargs)
    ptt0 = phaseshift.ptt_theo(sw, f, c,
        iridge=0)
    return {'f':f, 'c':c, 't':ptt,
            't0':ptt0}

def example_mft_fc_for_paper(sw):
    f = np.logspace(np.log10(sw['f'][0]),
        np.log10(sw['f'][-1]),
        51)
    c = np.interp(f, sw['f'], sw['c'])
    return f, c

def ptts_nearfield_finite_frequency_for_paper():
    dist = np.arange(0.5, 4.001, 0.1)
    return ptts_nearfield_finite_frequency(dist, gamma=1)

def ptts_nearfield_finite_frequency(dist, gamma, **kwargs):
    res = []
    ndist = len(dist)
    ptts_theo = []
    ptts_sf = []
    ptts_ag = []
    for i in tqdm(range(ndist)):
        d = dist[i]
        sw = surface_wave_ncf_for_paper(d)
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

def plot_nearfield_finite_frequency_for_paper(f, dist, ptts_theo, ptts_sf, ptts_ag):
    left, right, bottom, top = 0.15, 0.35, 0.80, 0.91
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=left, right=right, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_ax(f, dist, ptts_ag-ptts_theo, cmap='jet', ax=axs[0], ax_cb=None)
    axs[1].axis('off')
    axs[0].set_title('a. Finite frequency + near field',
            fontweight='bold', fontsize=9)

    xoffset = 0.23
    axs2 = fig.subplots(1,2,gridspec_kw=dict(left=left+xoffset, right=right+xoffset, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_ax(f, dist, ptts_ag-ptts_sf, cmap='jet', ax=axs2[0], ax_cb=None)
    axs2[1].axis('off')
    axs2[0].set_ylabel('')
    axs2[0].tick_params(labelleft=False)
    axs2[0].set_title('b. Finite frequency',
            fontweight='bold', fontsize=9)

    xoffset = 0.46
    axs3 = fig.subplots(1,2, gridspec_kw=dict(left=left+xoffset, right=right+xoffset, bottom=bottom, top=top,
        width_ratios=(1,0.05), wspace=0.1))
    plot_ps_pcolormesh_for_paper_ax(f, dist, ptts_sf-ptts_theo, cmap='jet', ax=axs3[0], ax_cb=axs3[1])
    axs3[0].set_ylabel('')
    axs3[0].tick_params(labelleft=False)
    axs3[0].set_title('c. Near field',
            fontweight='bold', fontsize=9)

    return Figure(fig)

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

def plot_ps_pcolormesh_for_paper(sw, phase_tts, cmap='jet',
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

def plot_ps_pcolormesh_for_paper_ax(f, dist, phase_tts, cmap='jet',
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

def plot_simplest_phaseshift_for_paper(dist=3):
    f = np.array([0.99, 1., 1.01])
    c = np.array([1.5, 1.4, 1.3])
    c1 = np.array([1.5, 1.4, 1.4])

    fig, axs = plt.subplots(1,3,figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.8,
                top=0.95, bottom=0.85, wspace=0.5))

    ax = axs[0]
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
        
    _, t = example_para_for_paper()
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

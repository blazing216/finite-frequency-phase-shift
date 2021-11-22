#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

import plot as yplot
import utils as yutils

from phase_shift_lib import phaseshift, demo_ak135

def roots(x, y):
    assert len(x) == len(y)
    n = len(x)

    bracket_idx = []
    for i in range(n-1):
        if y[i] * y[i+1] < 0:
            bracket_idx.append(i)

    roots = []
    for i0 in bracket_idx:
        r = x[i0] + (0-y[i0]) / (y[i0+1]-y[i0]) * (x[i0+1]-x[i0])
        roots.append(r)

    return np.array(roots)

def single_frequency_wave(sw, tp, phaseshift=0,
                         return_para=False):
    # compute k, k', k''
    omega = 2 * np.pi * sw['f']
    df = sw['f'][1] - sw['f'][0]
    domega = 2 * np.pi * df
    k = omega / sw['c']
    kp = yutils.diff(k, domega)
    #kpp = yutils.diff(kp, domega)

    omega0s = roots(omega, kp - tp/sw['dist'])
    omega0 = omega0s[0]

    # omega0 = 2 * np.pi * 0.002
    komega0 = np.interp(omega0, omega, k)

    if return_para:
        return omega0, komega0, np.cos(omega0*sw['t'] - \
                  (komega0*sw['dist'] + phaseshift))
    else:
        return np.cos(omega0*sw['t'] - \
                  (komega0*sw['dist'] + phaseshift))

def plot_fig4(sw, tp_approx=902,
        plot_time_window=(500, 1200)):
    fig, ax = plt.subplots(1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.1, right=0.4,
                            bottom=0.85, top=0.95))
    
    seis_normed = sw['seis'] / np.abs(sw['seis']).max()
    
    ax.plot(sw['t'], seis_normed, '-k', lw=0.5, label='Broadband')

    tp = phaseshift.closest_maximum_accurate(tp_approx,
            sw['t'], seis_normed)
    amp = np.interp(tp, sw['t'], seis_normed)
    omega0, komega0, single_freq = single_frequency_wave(sw, tp, phaseshift=0,
                                       return_para=True)
    ax.plot(sw['t'], single_freq / np.abs(single_freq).max() * amp,
            '-r', lw=0.5,
           label=f'Single frequency ({omega0/(2*np.pi):.3g} Hz)')

    ax.plot([tp,tp],[amp-0.5, amp+0.4], dashes=(5,2), lw=0.5,
           )
    ax.text(tp, amp+0.4, f't = {tp:.2f} s',
            ha='center', va='bottom',
           fontsize=9)
    
    t_sel = (sw['t'] >= plot_time_window[0]) & (sw['t'] <= plot_time_window[1])
    ymin, ymax = seis_normed[t_sel].min(), seis_normed[t_sel].max()
    
    yplot.axis_params(plot_time_window, 'Time (s)',
        [ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin)],
        left=False, top=False, right=False,
        ax=ax)
    yplot.ticklocator_params(xminor=100, ax=ax)
    
    ax.legend(bbox_to_anchor=(0, 1, 1, 1),
             bbox_transform=ax.transAxes,
              loc='lower left',
             ncol=1,
             borderpad=0,
             borderaxespad=0)
    
    return yplot.Figure(fig)

if __name__ == '__main__':
    #plt.style.use('gmt_paper')

    sw = demo_ak135.surface_wave_ncf_for_paper(dist=3000)
        
    fig = plot_fig4(sw, tp_approx=902,
            plot_time_window=(700, 1000))

    fig.savefig('../figs/fig4.pdf')
    fig.savefig('../figs/fig4.png')

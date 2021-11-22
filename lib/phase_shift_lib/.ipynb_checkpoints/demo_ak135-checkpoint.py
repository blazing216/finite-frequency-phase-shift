import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import ygeolib.pvt as ypvt
from ygeolib.layermodel import Model1D
from ygeolib import disp_xu as disp
import ygeolib.utils as yutils
import ygeolib.plot as yplot

from .synthetic import surface_wave
from . import synthetic
from . import phaseshift


def get_ak135_model_file():
    return ('/mnt/seismodata2/YX/Marathon/Project_Phase_shift'
        '/src/phase_shift_lib/mtak135sph.mod')

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

def surface_wave_for_paper(dist, farfield=False,
        casual_branch_only=True):
    f, t = example_para_ak135()
    c = cR_ak135(f)
    
    seis = synthetic.surface_wave(f, c, t, dist=dist,
            farfield=farfield,
            casual_branch_only=casual_branch_only)
        
    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

def ptt(sw_ak135, gamma, **kwargs):
    f, c = example_ak135_mft_fc(sw_ak135)
    ptt = phaseshift.ptt_ag(sw_ak135, f, c, gamma=gamma,
            emin=7, **kwargs)
    #ptt_0th = phaseshift.ptt_sg(sw_ak135, f, c,
    #        gamma=gamma, emin=7, iridge=0)
    #ptt[f<0.012] = ptt_0th[f<0.012]
    ptt0 = phaseshift.ptt_theo(sw_ak135, f, c, iridge=0)
    return {'f':f, 'c':c, 't': ptt,
            't0': ptt0}

def example_ak135_mft_fc(sw_ak135):
    f = np.logspace(np.log10(sw_ak135['f'][0]),
        np.log10(sw_ak135['f'][-1]),
        501)
    c = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    return f, c

def ptts(dist, gamma, **kwargs):
    res = []
    ndist = len(dist)
    for i in tqdm(range(ndist)):
    #for i, d in enumerate(dist):
        d = dist[i]
        sw_ak135 = surface_wave_for_paper(d)
        phase_tt = ptt(sw_ak135, gamma=gamma, **kwargs)
        phase_tt.update({'dist': d, 'gamma': gamma})
        res.append(phase_tt)
    return res

def ptts_sf(dist, f, gamma, iridge=0):
    sw_ak135 = surface_wave_for_paper(dist)
    t0 = phaseshift.ptt_theo(sw_ak135, f,
        c=np.interp(f, sw_ak135['f'], sw_ak135['c']),
        iridge=iridge)
    phase_tt = phaseshift.ptt_sf(sw_ak135, f, t0, gamma)
    c0 = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    t0 = phaseshift.ptt_theo(sw_ak135, f, c0,
        iridge=iridge)
    res = {'gamma': gamma, 'ptt': phase_tt,
            'f0': f,
            'c0': c0, 't0': t0}
    return res

def ptts_sf_for_paper(gamma=None, dist=1000,
        f=0.02, iridge=0, ngamma=101):
    if gamma is None:
        gamma = np.logspace(np.log10(1),
                np.log10(100), ngamma)

    sw_ak135 = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    t0_theo = phaseshift.ptt_theo(sw_ak135, f,
        c=c0, iridge=iridge)

    phase_tts = []
    t0 = t0_theo
    for each_gamma in gamma[::-1]:
        phase_tt = phaseshift.ptt_sf(sw_ak135, f, t0, each_gamma)
        #print(f'debug: t0 = {t0:.3g} s, phase_tt[0] = {phase_tt[0]:.3g} s', flush=True)
        phase_tts.append(phase_tt[0])
        t0 = phase_tt[0]
    phase_tts = np.array(phase_tts[::-1])

    res = {'gamma': gamma, 'ptt': phase_tts,
            'f0': f,
            'c0': c0, 't0': t0_theo}
    return res

def ptts_sf_meier_for_paper(gamma=None, dist=1000,
        f=0.02, iridge=0, ngamma=101):
    if gamma is None:
        gamma = np.logspace(np.log10(1),
                np.log10(100), ngamma)

    sw_ak135 = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    t0_theo = phaseshift.ptt_theo(sw_ak135, f, c0)
    ptts = []
    for i, each_gamma in enumerate(gamma):
        res = phaseshift.ptt_sf_meier2(sw_ak135, f,
                t0_theo,
                gamma=each_gamma)
        #print(f'ptt = {res["ptt"]}')
        ptts.append(res['ptt'])
    ptts = np.array(ptts)

    return {'gamma': gamma, 'ptt': ptts,
            'f0': f,
            'c0': c0, 't0': t0_theo}

def ptts_sf_approx_for_paper(gamma=None, dist=1000,
        f=0.02, iridge=0, ngamma=101):
    if gamma is None:
        gamma = np.logspace(np.log10(1),
                np.log10(100), ngamma)

    sw_ak135 = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    t0_theo = phaseshift.ptt_theo(sw_ak135, f, c0)
    ptts = []
    for i, each_gamma in enumerate(gamma):
        res = phaseshift.ptt_sf_approx(sw_ak135, f,
                gamma=each_gamma)
        ptts.append(res['ptt'])
    ptts = np.array(ptts)

    return {'gamma': gamma, 'ptt': ptts,
            'f0': f,
            'c0': c0, 't0': t0_theo}

def debug_ptt_sf_meier(sw, f=0.02, gamma=16):
    dt = sw['t'][1] - sw['t'][0]
    b = sw['t'][0]
    freq = np.fft.fftfreq(len(sw['seis']), dt)

    fv = f
    # option 1 (interpolate on fv):
    seis_filtered1 = phaseshift.gaussian_filter(sw,
            f=fv, gamma=gamma)
    seis_spec1 = np.fft.fft(seis_filtered1)
    seis_spec1 = seis_spec1 * np.exp(1j * 2* np.pi * freq * (-b))
    phase = np.angle(seis_spec1)
    idx = np.argsort(freq)
    phase1 = np.interp(fv, freq[idx], phase[idx])

    # option 2 (the closest frequency sample to fv):
    ifv = np.argmin(np.abs(fv - freq))
    seis_filtered2 = phaseshift.gaussian_filter(sw,
            f=freq[ifv], gamma=gamma)
    seis_spec2 = np.fft.fft(seis_filtered2)
    phase2 = np.angle(seis_spec2[ifv] * \
        np.exp(1j*2*np.pi*freq[ifv]*(-b)))

    # unwrap
    cv = np.interp(fv, sw['f'], sw['c'])
    t0_theo = phaseshift.ptt_theo(sw, fv, cv)
    t0 = phaseshift.closest_maximum_accurate(t0_theo,
        sw['t'], seis_filtered1)
    n1 = round((t0_theo * 2*np.pi*fv - (-phase1))\
            / (2*np.pi))
    n2 = round((t0_theo * 2*np.pi*fv - (-phase2))\
            / (2*np.pi))
    phase1_ur = -phase1 + n1*2*np.pi
    phase2_ur = -phase2 + n2*2*np.pi

    return {'f': fv, 'c':cv, 'dist':sw['dist'],
        'gamma': gamma, 'seis1': seis_filtered1,
        'seis2': seis_filtered2, 't': sw['t'],
        'phase_fd1': phase1_ur, 'phase_fd2':phase2_ur,
        'phase_theo': t0_theo * 2*np.pi*fv,
        'phase_td': t0 * 2*np.pi*fv,
        't_fd1': phase1_ur / (2*np.pi*fv),
        't_fd2': phase2_ur / (2*np.pi*freq[ifv]),
        't_theo': t0_theo,
        't_td': t0}

def plot_debug_ptt_sf_meier(res):
    plt.plot(res['t'], res['seis1'], color='k')
    plt.plot(res['t'], res['seis2'], color='k')
    plt.axvline(res['t_theo'], color='k',
               label='Theo')
    plt.axvline(res['t_fd1'], color='r',
               label='Freq1')
    plt.axvline(res['t_fd2'], color='g',
               label='Freq2')
    plt.axvline(res['t_td'], color='b',
               label='Time')

    xlim = (res['t_theo']-0.1/res['f'],
             res['t_theo']+0.1/res['f'])
    plt.xlim(*xlim)
    sel = (res['t'] >= xlim[0]) & (res['t'] <= xlim[1])
    ylimit = (res['seis1'][sel].min(),
              res['seis2'][sel].max())
    ylim = (ylimit[0] - 0.1 * (ylimit[1]-ylimit[0]),
           ylimit[1] + 0.1 * (ylimit[1]-ylimit[0]),)
    plt.ylim(*ylim)
    plt.xlabel('Time (s)')
    plt.legend(bbox_to_anchor=(0,1,1,0.3),
              bbox_transform=plt.gca().transAxes,
              loc='lower center',
              mode='expand',
              ncol=4,
              title=(f'Phase travel time\n'
               f'f = {res["f"]:.4g} Hz'
               f', $\gamma$ = {res["gamma"]:.4g}'),
              fontsize=6)
    #plt.title('f =')
    plt.tight_layout()


def plot_debug_ptts_sf_for_paper(gamma=None, dist=1000,
        f=0.02, iridge=0, ngamma=101,
        every=5):
    if gamma is None:
        gamma = np.logspace(np.log10(1),
                np.log10(100), ngamma)

    sw_ak135 = surface_wave_for_paper(dist)
    c0 = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    t0_theo = phaseshift.ptt_theo(sw_ak135, f,
        c=c0, iridge=iridge)

    plt.figure(figsize=(5,4))
    plt.subplot(221)
    t0 = t0_theo
    phase_tts = []
    for i, each_gamma in enumerate(gamma[::-1]):
        phase_tt, seis_filtered = phaseshift.ptt_sf(sw_ak135, f, t0, each_gamma,
            return_waveform=True)
        phase_tts.append(phase_tt[0])
        #print(f'gamma={each_gamma:.4g}, t0={t0:.4g}s, phase_tt[0]={phase_tt[0]:.4g}s')
        t0 = phase_tt[0]
        #print(f'debug: t0 = {t0:.3g} s, phase_tt[0] = {phase_tt[0]:.3g} s', flush=True)

        if i%every == 0:
            plt.plot(sw_ak135['t'],
                    seis_filtered[0,:] + i//every, lw=0.25,
                    )
            markx = phase_tt[0]
            marky = np.interp(markx,
                sw_ak135['t'],
                seis_filtered[0,:] + i//every)
            plt.plot(markx, marky, '|', color='r')
    phase_tts = np.array(phase_tts[::-1])
    plt.axvline(t0_theo, lw=0.5,
            dashes=(10,5), color='k')
    plt.xlabel('Time (s)')
    plt.ylabel('$\gamma$')
    plt.xlim(0, 400)
    #plt.ylim(0, (len(gamma)-1)//every)
    print(gamma.shape)
    gamma_locator = plt.FixedLocator(\
        np.arange(0, (len(gamma)-1)//every + 1),
        nbins=5)
    gamma_formatter = plt.FuncFormatter(lambda x, pos: \
            f'{gamma[::-1][round(x)*every]:.3g}')
    plt.gca().yaxis.set_major_locator(gamma_locator)
    plt.gca().yaxis.set_major_formatter(gamma_formatter)
    plt.title(f'f = {f:.4g} Hz')

    plt.subplot(222)
    plt.plot(gamma, phase_tts)
    plt.xlabel('$\gamma$')
    plt.ylabel('Time (s)')
    plt.title('Travel time difference')

    plt.subplot(223)
    plt.semilogx(gamma, phase_tts)
    plt.xlabel('$\gamma$')
    plt.ylabel('Time (s)')
    plt.title('Travel time difference (log freq)')

    plt.tight_layout()

def plot_debug_phase_shift_single_ridge(sw, gamma=16, iridge=0,
        ampratio=1):
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=iridge)

    start_f = f[0]
    start_t = np.interp(start_f, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag, idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=start_f, t0=start_t,
            branch_jump=False,
            corrected=True,
            ampratio=ampratio,
            return_index=True)
    ps = (t_theo - t_ag) * 2*np.pi*f

    fig = plt.figure(figsize=(6,5))

    plt.subplot(221)
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray')
    plt.plot(t_theo, f, color='r', lw=0.5)
    plt.plot(t_ag, f, color='green', lw=0.5,
            dashes=(10,5))
    plt.plot(sw['t'][idx], f, color='blue', lw=0.5,
            dashes=(10,5))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(200,1200)


    plt.subplot(222)
    t0 = {'f':f, 'c':c, 'ptt':t_ag}
    t1 = {'f':f, 'c':c, 'ptt':t_theo}
    plt.plot(t0['f'], t0['ptt'], '-k')
    plt.plot(t1['f'], t1['ptt'], '-r')
    plt.title('Travel time')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Travel time (s)')

    plt.subplot(223)
    plt.axhline(0, color='k')
    plt.plot(t1['f'], t1['ptt'] - t0['ptt'], '-r')
    plt.title('Travel time difference')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dt (s)')

    plt.subplot(224)
    plt.axhline(0, color='k')
    plt.plot(t1['f'], ps, '-r')
    plt.title('Phase shift')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase shift (radian)')
    plt.ylim(-np.pi/4, np.pi/4)
    plt.gca().set_yticks(np.arange(-0.25, 0.251, 0.125) * np.pi)
    plt.gca().set_yticklabels(['-$\pi$/4', '-$\pi$/8',
            '0', '$\pi$/8', '$\pi$/4'])

    plt.tight_layout()


def plot_ptts_sf(phase_tts_sf):
    plt.plot(phase_tts_sf['gamma'],
        phase_tts_sf['ptt'],
        lw=0.5)
    plt.axhline(phase_tts_sf['t0'],
        lw=0.5, dashes=(5,5))
    plt.xlabel('$\gamma$')
    plt.ylabel('Time (s)')
    plt.tight_layout()

def debug_plot_ptts_sf_for_multifreqs(phase_tts_sf_list):
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list)))
    for i, phase_tts_sf in enumerate(phase_tts_sf_list):
        plt.plot(phase_tts_sf['gamma'],
            phase_tts_sf['ptt'],
            lw=0.5, color=colors[i],
            label='%g Hz' % phase_tts_sf['f0'])
        plt.axhline(phase_tts_sf['t0'],
            lw=0.5, dashes=(5,5),
            color=colors[i])
    plt.xlabel('$\gamma$')
    plt.ylabel('Time (s)')
    plt.legend(title='center freq.')
    plt.tight_layout()

def ptts_for_paper(gamma=16, **kwargs):
    dist = np.arange(100, 3001, 100)
    return ptts(dist, gamma, **kwargs)

def ptt_single_ridge(sw, gamma, **kwargs):
    f, c = example_ak135_mft_fc(sw)
    ptt = phaseshift.ptt_single_ridge(sw, f, c,
            gamma=gamma, emin=7, **kwargs)
    ptt0 = phaseshift.ptt_theo(sw, f, c,
            iridge=0)
    return {'f':f, 'c':c, 't':ptt, 't0':ptt0}

def ptts_single_ridge_for_paper_to_file(outfile, gamma=16, force=False, **kwargs):
    if not force:
        print('Use force=True to run. Notice that it might take a few minutes', flush=True)
        return

    dist = np.arange(100, 3001, 100)
    res = []
    ndist = len(dist)
    for i in tqdm(range(ndist)):
        d = dist[i]
        sw_ak135 = surface_wave_for_paper(d)
        phase_tt = ptt_single_ridge(sw_ak135, gamma=gamma, **kwargs)
        phase_tt.update({'dist': d, 'gamma': gamma})
        res.append(phase_tt)

    pickle.dump(res,
            open(outfile, 'wb'))
    print(outfile, flush=True)

def ptts_for_paper_to_file(outfile, gamma=16, force=False, **kwargs):
    if not force:
        print('Use force=True to run. Notice that it might take a few minutes', flush=True)
        return

    res = ptts_for_paper(gamma, **kwargs)
    pickle.dump(res,
            open(outfile, 'wb'))
    print(outfile, flush=True)

def valid_period_limits(phase_tt):
    return phaseshift.valid_period_limits(\
        phase_tt['f'][0], phase_tt['f'][-1],
        gamma=phase_tt['gamma'])

def valid_freq_limits(phase_tt):
    return phaseshift.valid_freq_limits(\
        phase_tt['f'][0], phase_tt['f'][-1],
        gamma=phase_tt['gamma'])

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

def plot_ps_contourf(phase_tts, cmap='jet'):
    f, dist, dphase = ptts2ps(phase_tts)
    plt.contourf(1/f, dist, dphase, levels=np.linspace(-np.pi/4, np.pi/4, 51), cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    plt.semilogx()
    plt.colorbar()
    plt.xlim(25, 200)
    plt.xlabel('Period (s)')
    plt.ylabel('Distance (km)')

def plot_ps_pcolormesh(phase_tts, cmap='jet'):
    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    plt.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    plt.semilogx()
    plt.colorbar()
    plt.xlim(25, 100)
    plt.ylim(100, 3000)
    plt.xlabel('Period (s)')
    plt.ylabel('Distance (km)')

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

def plot_dcc_pcolormesh(phase_tts, cmap='jet'):
    f, dist, dc_over_c = ptts2dcc(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    plt.pcolormesh(Tbins, distbins, dc_over_c[:,::-1],
            cmap=cmap, vmin=-0.01, vmax=0.01)
    plt.semilogx()
    plt.colorbar()
    plt.xlim(25, 100)
    plt.xlabel('Period (s)')
    plt.ylabel('Distance (km)')

class Figure:
    def __init__(self, fig):
        self.fig = fig

    def savefig(self, figname, dpi=300):
        self.fig.savefig(figname, dpi=dpi, bbox_inches='tight')

def plot_ps_pcolormesh_for_paper(phase_tts, cmap='jet'):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')
    #plt.semilogx()
    #plt.colorbar()
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.arange(-0.25, 0.251, 0.125))
    ax_cb.set_yticklabels(['-$\pi$/4','-$\pi$/8','0','$\pi$/8','$\pi$/4'])
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')
    return Figure(fig)

def plot_ps_pcolormesh_method_cmp_for_paper(phase_tts, phase_tts1, cmap='jet'):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')
    #plt.semilogx()
    #plt.colorbar()
    #cb = fig.colorbar(im, cax=ax_cb)
    #cb.set_label('Phase shift (radian)')
    #cb.set_ticks(np.pi * np.arange(-0.25, 0.251, 0.125))
    #ax_cb.set_yticklabels(['-$\pi$/4','-$\pi$/8','0','$\pi$/8','$\pi$/4'])
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')
    ax_cb.axis('off')
    ax.set_title('Single ridge')
    
    ax_single_ridge = ax
    
    axs = fig.subplots(1,2, gridspec_kw=dict(left=0.45, right=0.7, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts1)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')
    #plt.semilogx()
    #plt.colorbar()
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.arange(-0.25, 0.251, 0.125))
    ax_cb.set_yticklabels(['-$\pi$/4','-$\pi$/8','0','$\pi$/8','$\pi$/4'])
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    #ax.set_ylabel('Distance (km)')
    ax.tick_params(labelleft=False)
    
    ax.set_title('Multiple ridges')
    
    ax_multiple_ridges = ax
    
    yplot.labelax([ax_single_ridge, ax_multiple_ridges],
            loc='lower left')
    
    return Figure(fig)

def plot_ps_pcolormesh_method_cmp_for_paper_v2(sw, phase_tts, phase_tts1, cmap='jet'):
    
    freq_lim = (0.01, 0.04)
    colors = ['k', 'r', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    gamma = phase_tts[0]['gamma']
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.5)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.5)
    
    t_0th, t_0th_idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True,
            return_index=True)
    
    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(1,2, left=0.15, right=0.7,
                             top=0.95, bottom=0.8,
                            width_ratios=[1,0.3],
                           wspace=0.2,
            )
    
    axs = [fig.add_subplot(gs[0,0]),
           fig.add_subplot(gs[0,1])]

    
    
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
    
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax, rasterized=True)
    #ax.plot(t_theo, f, color=colors[0], lw=1)
    ax.plot(t_0th, f, color=colors[1], lw=1)
    ax.plot(t_ag, f, color=colors[2], lw=1)
    ax.plot(t_ag_uncorrected, f, lw=1, color=colors[2],
            dashes=(5,1))

    yplot.axis_params((400,550), 'Time (s)',
            (0.01, 0.04), 'Frequency (Hz)',
            ax=ax)
    yplot.ticklocator_params(50, 10, 0.01, 0.005, ax=ax)
    
    ax = axs[1]
    ax.plot((t_0th - t_theo)*2*np.pi*f, f, lw=1,
            color=colors[1])
    ax.plot((t_ag - t_theo)*2*np.pi*f, f, lw=1,
            color=colors[2])
    
    yplot.axis_params((-np.pi/6, np.pi/6), 'Phase shift (radian)',
            (0.01, 0.04), 'Frequency (Hz)',
            ax=ax)
    ax.tick_params(labelright=True, labelleft=False)
    ax.yaxis.set_label_position('right')
    
    #ax.set_xticks([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])
    ax.set_xticks([-np.pi/6, 0, np.pi/6])
    ax.set_xticks(np.linspace(-np.pi/6, np.pi/6, 9),
            minor=True)
    #ax.set_xticklabels(['$-\\pi$/6', '$-\\pi$/12', '0', '$\\pi$/12', '$\\pi$/6'])
    ax.set_xticklabels(['$-\\pi$/6', '0', '$\\pi$/6'])
    ax.grid(axis='x', which='both')
    
    yplot.ticklocator_params(None, None, 0.01, 0.005, ax=ax)
    
    axs_mft = axs
    
    
    axs = fig.subplots(1,2, gridspec_kw=dict(left=0.15, right=0.43, bottom=0.6, top=0.75,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')

    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')
    ax_cb.axis('off')
    # ax.set_title('Single ridge')
    
    ax_single_ridge = ax
    
    axs = fig.subplots(1,2, gridspec_kw=dict(left=0.445, right=0.725, bottom=0.6, top=0.75,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dphase = ptts2ps(phase_tts1)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dphase[:,::-1], cmap=cmap, vmin=-np.pi/4, vmax=np.pi/4)
    ax.set_xscale('log')
    #plt.semilogx()
    #plt.colorbar()
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Phase shift (radian)')
    cb.set_ticks(np.pi * np.arange(-0.25, 0.251, 0.125))
    ax_cb.set_yticklabels(['-$\pi$/4','-$\pi$/8','0','$\pi$/8','$\pi$/4'])
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    #ax.set_ylabel('Distance (km)')
    ax.tick_params(labelleft=False)
    
    # ax.set_title('Multiple ridges')
    
    ax_multiple_ridges = ax
    
    yplot.labelax([axs_mft[0], axs_mft[1], ax_single_ridge, ax_multiple_ridges],
            loc='lower left')
    yplot.labelax([ax_single_ridge, ax_multiple_ridges],
                 ['Single ridge', 'Multiple ridges'],
                 loc='upper right',
                 fontdict=dict(weight='normal', size=9))
    
    def line(color='k', lw=1, **kwargs):
        from matplotlib.lines import Line2D
        return Line2D([], [], color=color, lw=lw, **kwargs)
    
    axs_mft[0].legend(handles=[line(colors[1]), line(colors[2]), line(colors[2], dashes=(5,1))],
        labels=['Single', 'Multiple', 'Multiple (before correction)'],
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=axs_mft[0].transAxes,
        loc='lower right',
        ncol=1,
        facecolor='w'
        )
    
    return Figure(fig)

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
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')

    return Figure(fig)

def plot_dcc_pcolormesh_for_paper_dis_cmap(phase_tts, cmap='jet',
        vmin=-0.01, vmax=0.01):
    fig, axs = plt.subplots(1,2, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        width_ratios=(1,0.05), wspace=0.1))
    ax, ax_cb = axs[0], axs[1]

    f, dist, dcc = ptts2dcc(phase_tts)
    Tbins = yutils.logbins(1/f[::-1])
    distbins = yutils.linbins(dist)
    im = ax.pcolormesh(Tbins, distbins, dcc[:,::-1],
            cmap=plt.get_cmap(cmap, 20), vmin=vmin, vmax=vmax)
    ax.set_xscale('log')
    cb = fig.colorbar(im, cax=ax_cb)

    cb.set_label('$\delta$c/c (%)', labelpad=-5)
    ax_cb.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax_cb.yaxis.set_minor_locator(plt.MultipleLocator(0.001))
    ax_cb.yaxis.set_major_formatter(plt.FuncFormatter(\
            lambda x, pos: '%g' % (x*100)))
    ax.set_xticks(np.arange(30,101,10))
    ax.set_xticks(np.arange(25,101,5), minor=True)
    ax.set_xticklabels(['30', '40', '', '60',
        '', '80', '', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(25, 100)
    ax.set_ylim(100, 3000)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Distance (km)')
    return Figure(fig)

def plot_ptts_sf_for_paper(phase_tts_sf):
    fig, ax = plt.subplots(1, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        ))
    ax.plot(phase_tts_sf['gamma'],
        phase_tts_sf['ptt'],
        lw=0.5)
    ax.axhline(phase_tts_sf['t0'],
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('Time (s)')
    ax.tick_params(right=False)
    ylim = ax.get_ylim()


    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']
    ps2ptt = lambda x: x / (2*np.pi*phase_tts_sf['f0']) + phase_tts_sf['t0']
    ax1 = ax.secondary_yaxis('right',
            functions=(ptt2ps, ps2ptt))
    ax1.set_ylabel('Phase shift (radian)')
    ax1.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax1.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    #ax1.set_ylim(-np.pi/4, np.pi/4)
    ax.set_ylim(ps2ptt(-np.pi/4), ps2ptt(np.pi/4))
    print(ps2ptt(-np.pi/4), ps2ptt(np.pi/4))

    return Figure(fig)

def plot_ptts_sf_phaseshift_for_paper(phase_tts_sf):

    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']

    fig, ax = plt.subplots(1, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        ))
    ax.plot(phase_tts_sf['gamma'],
        ptt2ps(phase_tts_sf['ptt']),
        lw=0.5)
    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    ax.set_ylim(-np.pi/4, np.pi/4)

    return Figure(fig)

def plot_ptts_sf_phaseshift_list_for_paper(phase_tts_sf_list):

    ptt2ps = lambda x: (x - phase_tts_sf['t0']) * 2 * np.pi * phase_tts_sf['f0']
    fig, ax = plt.subplots(1, figsize=yplot.a4, gridspec_kw=dict(left=0.15, right=0.4, bottom=0.80, top=0.95,
        ))
    colors = plt.get_cmap('jet')(np.linspace(0,1,len(phase_tts_sf_list)))
    for i, phase_tts_sf in enumerate(phase_tts_sf_list):
        ax.plot(phase_tts_sf['gamma'],
            ptt2ps(phase_tts_sf['ptt']),
            lw=0.5, color=colors[i])
        ax.text(phase_tts_sf['gamma'][0],
                ptt2ps(phase_tts_sf['ptt'])[0],
                f'{phase_tts_sf["f0"]:g} Hz',
                ha='center', va='bottom',
                color=colors[i],
                fontsize=8)
    ax.axhline(0,
        lw=0.5, dashes=(5,5))
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('Phase shift (radian)')
    ax.set_yticks(np.arange(-0.5,0.51,0.125)*np.pi)
    ax.set_yticklabels(['-$\pi/2$', '-$3\pi/8$', '-$\pi/4$', '-$\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    #ax.set_ylim(-np.pi/4, np.pi/4)
    ax.set_xscale('log')

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

        #ax.text(phase_tts_sf['gamma'][0],
        #        ptt2ps(phase_tts_sf['ptt'])[0],
        #        f'{phase_tts_sf["f0"]:g} Hz',
        #        ha='center', va='bottom',
        #        color=colors[i],
        #        fontsize=8)
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
    ax.set_xlim(5, 100)
    ax.set_ylim(-np.pi/4, np.pi/4)
    ax.legend(title='Period',
            mode='expand',
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.1, 0.6, 0.8, 0.4),
            bbox_transform=ax.transAxes)

    return Figure(fig)

def plot_ptts_sf_phaseshift_list_cmp_for_paper(phase_tts_sf_list,
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
    ax.set_xlim(5, 100)
    ax.set_ylim(-np.pi/4, np.pi/4)
    ax.legend(title='Method',
            ncol=1,
            loc='upper center',
            fontsize=8,
            bbox_to_anchor=(0.1, 0.60, 0.8, 0.4),
            bbox_transform=ax.transAxes)

    return Figure(fig)


def plot_illustration(sw, gamma=16):
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.2)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.2)
    
    t_0th = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True)
    
    fig, axs = plt.subplots(1,1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.2, right=0.7,
                             top=0.95, bottom=0.8),
            dpi=100)
    
    ax = axs
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
    ax.plot(t_theo, f, color='r', lw=1)
    ax.plot(t_0th, f, color='blue', lw=1,
            dashes=(5,1))
    ax.plot(t_ag, f, color='green', lw=1,
            dashes=(5,1))
    ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color='green',
            markevery=5)
    yplot.axis_params((150,500), 'Time (s)',
            (f[0], f[-1]), 'Frequency (Hz)',
            ax=ax)
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.75, bottom=0.65, wspace=0.3))
    
    ax = axs2[0]
    ax.plot(f, t_theo, lw=1, color='r')
    ax.plot(f, t_0th, lw=1,
           color='blue')
    ax.plot(f, t_ag, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            (t_theo[0], t_theo[-1]), 'Phase travel time (s)', ax=ax)
    ax_phase = ax.twinx()
    ax_phase.plot(f, t_theo * 2*np.pi*f, lw=1, color='r')
    ax_phase.plot(f, t_0th * 2*np.pi*f, lw=1,
           color='blue')
    ax_phase.plot(f, t_ag * 2*np.pi*f, lw=1,
           color='green')
    ax_phase.set_ylim(t_theo[0]*2*np.pi*f[-1],
            t_theo[-1]*2*np.pi*f[-1])
    
    ax = axs2[1]
    ax.plot(f, c, lw=1, color='r')
    ax.plot(f, sw['dist'] / (t_0th + 1/(8*f)), lw=1,
            color='blue')
    ax.plot(f, sw['dist'] / (t_ag + 1/(8*f)), lw=1,
            color='green')
    #ax.set_xscale('log')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            (3,7), 'Velocity (km/s)', ax=ax)
    
    ax = axs2[1]
    
    return Figure(fig)


def plot_illustration_v2(sw, gamma=16):
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.2)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.2)
    
    t_0th = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True)
    
    fig, axs = plt.subplots(1,1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.2, right=0.7,
                             top=0.95, bottom=0.8),
            dpi=100)
    
    ax = axs
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
    ax.plot(t_theo, f, color='r', lw=1)
    ax.plot(t_0th, f, color='blue', lw=1,
            dashes=(5,1))
    ax.plot(t_ag, f, color='green', lw=1,
            dashes=(5,1))
    ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color='green',
            markevery=5)
    yplot.axis_params((150,500), 'Time (s)',
            (f[0], f[-1]), 'Frequency (Hz)',
            ax=ax)
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.75, bottom=0.65, wspace=0.3))
    
    ax = axs2[0]
    ax.plot(f, t_theo, lw=1, color='r')
    ax.plot(f, t_0th, lw=1,
           color='blue')
    ax.plot(f, t_ag, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            (t_theo[0], t_theo[-1]), 'Phase travel time (s)', ax=ax)
    
    ax = axs2[1]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, t_0th - t_theo, lw=1,
           color='blue')
    ax.plot(f, t_ag - t_theo, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            None, 'Phase travel time residual(s)', ax=ax)
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.6, bottom=0.5, wspace=0.3))
    
    ax = axs3[0]
    ax.plot(f, t_theo*2*np.pi*f, lw=1, color='r')
    ax.plot(f, t_0th*2*np.pi*f, lw=1,
           color='blue')
    ax.plot(f, t_ag*2*np.pi*f, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            None, 'Phase (radian)', ax=ax)
    
    ax = axs3[1]
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color='blue')
    ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            None, 'Phase residual(radian)', ax=ax)
    
    axs4 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.45, bottom=0.35, wspace=0.3))
    
    ax = axs4[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
    c_ag = sw['dist'] / (t_ag + 1/(8*f))
    ax.plot(f, c, lw=1, color='r')
    ax.plot(f, c_0th, lw=1,
            color='blue')
    ax.plot(f, c_ag, lw=1,
            color='green')
    #ax.set_xscale('log')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            (3,7), 'Velocity (km/s)', ax=ax)
    
    ax = axs4[1]
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, c_0th - c, lw=1,
           color='blue')
    ax.plot(f, c_ag - c, lw=1,
           color='green')
    yplot.axis_params((f[0], 0.05), 'Frequency (Hz)',
            None, 'Velocity residual (km/s)', ax=ax)
    
    return Figure(fig)
    

def plot_illustration_v3(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.2)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.2)
    
    t_0th = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True)
    
    fig, axs = plt.subplots(1,1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.2, right=0.7,
                             top=0.95, bottom=0.8),
            dpi=100)
    
    ax = axs
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
    ax.plot(t_theo, f, color='r', lw=1)
    ax.plot(t_0th, f, color='blue', lw=1,
            dashes=(5,1))
    ax.plot(t_ag, f, color='green', lw=1,
            dashes=(5,1))
    ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color='green',
            markevery=5)
    yplot.axis_params((300,500), 'Time (s)',
            freq_lim, 'Frequency (Hz)',
            ax=ax)
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.75, bottom=0.65, wspace=0.3))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, t_0th - t_theo, lw=1,
           color='blue')
    ax.plot(f, t_ag - t_theo, lw=1,
           color='green')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-1,1), 'Phase travel time residual(s)', ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color='blue')
    ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
           color='green')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.5, 0.5), 'Phase residual(radian)', ax=ax)
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.6, bottom=0.5, wspace=0.3))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
    c_ag = sw['dist'] / (t_ag + 1/(8*f))
    ax.plot(f, c, lw=1, color='r')
    ax.plot(f, c_0th, lw=1,
            color='blue')
    ax.plot(f, c_ag, lw=1,
            color='green')
    #ax.set_xscale('log')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (3.5,4.5), 'Velocity (km/s)', ax=ax)
    
    ax = axs3[1]
    ax.axhline(0, lw=1, color='r')
    ax.plot(f, c_0th - c, lw=1,
           color='blue')
    ax.plot(f, c_ag - c, lw=1,
           color='green')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.01, 0.01), 'Velocity residual (km/s)', ax=ax)
    
    return Figure(fig)


def plot_illustration_v4(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['r', 'b', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.5)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.5)
    
    t_0th = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True)
    
    fig, axs = plt.subplots(1,1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.2, right=0.7,
                             top=0.95, bottom=0.8),
            dpi=100)
    
    ax = axs
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
    ax.plot(t_theo, f, color=colors[0], lw=1)
    ax.plot(t_0th, f, color=colors[1], lw=1,
            dashes=(5,1))
    ax.plot(t_ag, f, color=colors[2], lw=1,
            dashes=(5,1))
    ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color=colors[2],
            markevery=5)
    yplot.axis_params((350,550), 'Time (s)',
            freq_lim, 'Frequency (Hz)',
            ax=ax)
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.75, bottom=0.65, wspace=0.2))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
    ax.plot(f, t_ag - t_theo, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-1.5,1.5), 'Phase travel time residual(s)', ax=ax)
    yplot.ticklocator_params(ymajor=1, yminor=0.5,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color=colors[1])
    ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.5, 0.5), 'Phase residual (radian)', ax=ax)
    def labelright(ax):
        ax.tick_params(labelleft=False,
                      labelright=True)
        ax.yaxis.set_label_position('right')
    labelright(ax)
    ax.set_yticks([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])
    ax.set_yticks(np.linspace(-np.pi/6, np.pi/6, 9),
            minor=True)
    ax.set_yticklabels(['$-\\pi$/6', '$-\\pi$/12', '0', '$\\pi$/12', '$\\pi$/6'])
    ax.grid(axis='y', which='both')
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.6, bottom=0.5, wspace=0.2))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
    c_ag = sw['dist'] / (t_ag + 1/(8*f))
    
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, c_0th - c, lw=1,
           color=colors[1])
    ax.plot(f, c_ag - c, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.015, 0.015), 'Phase velocity residual (km/s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.01, yminor=0.005,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs3[1]
    
    
    ax.plot(f, c, lw=1, color=colors[0])
    ax.plot(f, c_0th, lw=1,
            color=colors[1])
    ax.plot(f, c_ag, lw=1,
            color=colors[2])
    #ax.set_xscale('log')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (3.7,4.2), 'Phase velocity (km/s)', ax=ax)
    labelright(ax)
    yplot.ticklocator_params(ymajor=0.2, yminor=0.1,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    yplot.labelax([axs, axs2[0], axs2[1], axs3[0], axs3[1]], loc='lower left', fontdict=dict(size=10))
    
    def line(color='k', lw=1):
        from matplotlib.lines import Line2D
        return Line2D([], [], color=color, lw=lw)
    
    axs.legend(handles=[line(colors[0]), line(colors[1]), line(colors[2])],
        labels=['Far-field approx.', 'Obs. (single ridge)',
               'Obs. (ridges)'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs.transAxes,
        loc='lower center',
        ncol=3,
        mode='expand')
    
    
    return Figure(fig)

def plot_illustration_v5(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['r', 'b', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            refine=True,
            ampratio=1.5)
    t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=False,
            refine=True,
            ampratio=1.5)
    
    t_0th = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True)
    
    fig, axs = plt.subplots(1,1, figsize=yplot.a4,
            gridspec_kw=dict(left=0.2, right=0.7,
                             top=0.95, bottom=0.8),
            dpi=100)
    
    ax = axs
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax, rasterized=True)
    ax.plot(t_theo, f, color=colors[0], lw=1)
    ax.plot(t_0th, f, color=colors[1], lw=1,
            dashes=(5,1))
    ax.plot(t_ag, f, color=colors[2], lw=1,
            dashes=(5,1))
    ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color=colors[2],
            markevery=5)
    yplot.axis_params((350,550), 'Time (s)',
            freq_lim, 'Frequency (Hz)',
            ax=ax)
    
    # plot two examples of filtered waveforms
    sel_freqs = [0.02, 0.035]
    for sel_freq in sel_freqs:
        i_sel_freq = np.argmin(np.abs(f - sel_freq))
        ax.plot(sw['t'], mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max() * 0.003 + sel_freq,
               lw=0.5)
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.75, bottom=0.65, wspace=0.2))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
    ax.plot(f, t_ag - t_theo, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-1.5,1.5), 'Phase travel time residual(s)', ax=ax)
    yplot.ticklocator_params(ymajor=1, yminor=0.5,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color=colors[1])
    ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.5, 0.5), 'Phase residual (radian)', ax=ax)
    def labelright(ax):
        ax.tick_params(labelleft=False,
                      labelright=True)
        ax.yaxis.set_label_position('right')
    labelright(ax)
    ax.set_yticks([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])
    ax.set_yticks(np.linspace(-np.pi/6, np.pi/6, 9),
            minor=True)
    ax.set_yticklabels(['$-\\pi$/6', '$-\\pi$/12', '0', '$\\pi$/12', '$\\pi$/6'])
    ax.grid(axis='y', which='both')
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.6, bottom=0.5, wspace=0.2))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
    c_ag = sw['dist'] / (t_ag + 1/(8*f))
    
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, c_0th - c, lw=1,
           color=colors[1])
    ax.plot(f, c_ag - c, lw=1,
           color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.015, 0.015), 'Phase velocity residual (km/s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.01, yminor=0.005,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs3[1]
    
    
    ax.plot(f, c, lw=1, color=colors[0])
    ax.plot(f, c_0th, lw=1,
            color=colors[1])
    ax.plot(f, c_ag, lw=1,
            color=colors[2])
    #ax.set_xscale('log')
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (3.7,4.2), 'Phase velocity (km/s)', ax=ax)
    labelright(ax)
    yplot.ticklocator_params(ymajor=0.2, yminor=0.1,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    yplot.labelax([axs, axs2[0], axs2[1], axs3[0], axs3[1]], loc='lower left', fontdict=dict(size=10))
    
    def line(color='k', lw=1):
        from matplotlib.lines import Line2D
        return Line2D([], [], color=color, lw=lw)
    
    axs.legend(handles=[line(colors[0]), line(colors[1]), line(colors[2])],
        labels=['Far-field approx.', 'Obs. (single ridge)',
               'Obs. (ridges)'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs.transAxes,
        loc='lower center',
        ncol=3,
        mode='expand')
    
    
    return Figure(fig)


def plot_illustration_v6(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'r', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
#     t_ag = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=True,
#             refine=True,
#             ampratio=1.5)
#     t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=False,
#             refine=True,
#             ampratio=1.5)
    
    t_0th, t_0th_idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True,
            return_index=True)
    
    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(2,2, left=0.2, right=0.7,
                             top=0.95, bottom=0.8,
                            width_ratios=[1,0.5],
                          hspace=0.4,
            )
    
    axs = [fig.add_subplot(gs[:,0]),
           fig.add_subplot(gs[0,1]),
           fig.add_subplot(gs[1,1])]
#     ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
#     ax.plot(t_theo, f, color=colors[0], lw=1)
#     ax.plot(t_0th, f, color=colors[1], lw=1,
#            dashes=(5,1))
#     ax.plot(t_ag, f, color=colors[2], lw=1,
#             dashes=(5,1))
#     ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color=colors[2],
#             markevery=5)
    
    
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
        plot_label(350, sw['t'], waveform_shifted, f'{sel_freq:g} Hz', ha='right',
                  va='center', ax=ax, fontsize=9,
                  transform=yplot.offset_transform(-10/72, 0, ax=ax))
        
    waveform_shifted  = sw['seis'] / np.abs(sw['seis']).max() * 0.005 + 0.05
    ax.plot(sw['t'], waveform_shifted, lw=1)
    plot_label(350, sw['t'], waveform_shifted, 'Broadband', ha='right',
               va='center', ax=ax, fontsize=9,
               transform=yplot.offset_transform(-5/72, 0, ax=ax))
    #print(ax.figure, fig)
    
#     ax.tick_params(left=False, right=False, top=False,
#                   labelleft=False, labelright=False, labeltop=False,
#                   )
#     for spine in ['top', 'left', 'right']:
#         ax.spines[spine].set_visible(False)
#     ax.set_ylabel('')

        
    yplot.axis_params((350,550), 'Time (s)',
            (0.01-0.007, 0.05+0.005), None,
            left=False, top=False, right=False,
            ax=ax)
    
    # zoom in
    ax = axs[2]
    ylim_axs_1 = (0.18, 0.24)
    
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
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(460,468))
#     ax.plot(sw['t'], waveform_shifted,
#            lw=0.5)
    ax.plot(*waveform_polyfit, color='k',
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((460, 468), 'Time (s)',
            ylim_axs_1, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(4, 1, 0.02, 0.01, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))
        
    # zoom in
    ax = axs[1]
    ylim_axs_2 = (0.867, 0.871)
    
    sel_freq = 0.01
    i_sel_freq = np.argmin(np.abs(f - sel_freq))
    waveform_shifted  = mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max()
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(407,411))
    ax.plot(*waveform_polyfit,
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((407, 411), None,
            ylim_axs_2, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(2, 1, 0.002, 0.001, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.73, bottom=0.66, wspace=0.2))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
#     ax.plot(f, t_ag - t_theo, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, None,
            (-1.5,1.5), 'Time (s)', ax=ax)
    yplot.ticklocator_params(ymajor=1, yminor=0.5,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color=colors[1])
#     ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, None,
            (-0.5, 0.5), 'Phase (radian)', ax=ax)
    def labelright(ax):
        ax.tick_params(labelleft=False,
                      labelright=True)
        ax.yaxis.set_label_position('right')
    labelright(ax)
    ax.set_yticks([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])
    ax.set_yticks(np.linspace(-np.pi/6, np.pi/6, 9),
            minor=True)
    ax.set_yticklabels(['$-\\pi$/6', '$-\\pi$/12', '0', '$\\pi$/12', '$\\pi$/6'])
    ax.grid(axis='y', which='both')
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.63, bottom=0.56, wspace=0.2))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
#     c_ag = sw['dist'] / (t_ag + 1/(8*f))
    
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, c_0th - c, lw=1,
           color=colors[1])
#     ax.plot(f, c_ag - c, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.015, 0.015), 'Velocity (km/s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.01, yminor=0.005,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs3[1]
    
    
    ax.plot(f, c, lw=1, color=colors[0])
    ax.plot(f, c_0th, lw=1,
            color=colors[1])
#     ax.plot(f, c_ag, lw=1,
#             color=colors[2])
    #ax.set_xscale('log')
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
        labels=['Theorectial', 'Observed'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs[0].transAxes,
        loc='lower center',
        ncol=2,
        mode='expand')
    
    axs2[0].legend(handles=[line(colors[0]), line(colors[1])],
        labels=['Theorectial', 'Observed'],
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

def plot_illustration_v7(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'r', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
#     t_ag = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=True,
#             refine=True,
#             ampratio=1.5)
#     t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=False,
#             refine=True,
#             ampratio=1.5)
    
    t_0th, t_0th_idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True,
            return_index=True)
    
    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(2,2, left=0.2, right=0.7,
                             top=0.95, bottom=0.8,
                            width_ratios=[1,0.5],
                          hspace=0.4,
            )
    
    axs = [fig.add_subplot(gs[:,0]),
           fig.add_subplot(gs[0,1]),
           fig.add_subplot(gs[1,1])]
#     ypvt.plot_mft(mft, sw['t'], f, cmap='gray', ax=ax)
#     ax.plot(t_theo, f, color=colors[0], lw=1)
#     ax.plot(t_0th, f, color=colors[1], lw=1,
#            dashes=(5,1))
#     ax.plot(t_ag, f, color=colors[2], lw=1,
#             dashes=(5,1))
#     ax.plot(t_ag_uncorrected, f, marker='o', ms=2, color=colors[2],
#             markevery=5)
    
    
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
        plot_label(350, sw['t'], waveform_shifted, f'{sel_freq:g} Hz', ha='right',
                  va='center', ax=ax, fontsize=9,
                  transform=yplot.offset_transform(-10/72, 0, ax=ax))
        
    waveform_shifted  = sw['seis'] / np.abs(sw['seis']).max() * 0.005 + 0.05
    ax.plot(sw['t'], waveform_shifted, lw=1)
    plot_label(350, sw['t'], waveform_shifted, 'Broadband', ha='right',
               va='center', ax=ax, fontsize=9,
               transform=yplot.offset_transform(-5/72, 0, ax=ax))
    #print(ax.figure, fig)
    
#     ax.tick_params(left=False, right=False, top=False,
#                   labelleft=False, labelright=False, labeltop=False,
#                   )
#     for spine in ['top', 'left', 'right']:
#         ax.spines[spine].set_visible(False)
#     ax.set_ylabel('')

        
    yplot.axis_params((350,550), 'Time (s)',
            (0.01-0.007, 0.05+0.005), None,
            left=False, top=False, right=False,
            ax=ax)
    
    # zoom in
    ax = axs[1]
    ylim_axs_1 = (0.18, 0.24)
    
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
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(460,468))
#     ax.plot(sw['t'], waveform_shifted,
#            lw=0.5)
    ax.plot(*waveform_polyfit, color='k',
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((460, 468), None,
            ylim_axs_1, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(4, 1, 0.02, 0.01, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))
        
    # zoom in
    ax = axs[2]
    ylim_axs_2 = (0.867, 0.871)
    
    sel_freq = 0.01
    i_sel_freq = np.argmin(np.abs(f - sel_freq))
    waveform_shifted  = mft[i_sel_freq, :] / np.abs(mft[i_sel_freq, :]).max()
    
    waveform_polyfit = polyfit_waveform(i_sel_freq, tlim=(407,411))
    ax.plot(*waveform_polyfit,
           lw=0.5)
    plot_marker(t_theo[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[0],
                ax=ax)
    plot_marker(t_0th[i_sel_freq], waveform_polyfit[0], waveform_polyfit[1], marker='D', ms=2, color=colors[1],
                ax=ax)
    yplot.axis_params((407, 411), 'Time (s)',
            ylim_axs_2, None,
            ax=ax)
    ax.tick_params(labelleft=False, labelright=True)
    yplot.ticklocator_params(2, 1, 0.002, 0.001, ax=ax)
    yplot.labelax(ax, f'{sel_freq:g} Hz', fontdict=dict(weight='normal'))
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.73, bottom=0.66, wspace=0.2))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
#     ax.plot(f, t_ag - t_theo, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, None,
            (-1.5,1.5), 'Time (s)', ax=ax)
    yplot.ticklocator_params(ymajor=1, yminor=0.5,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs2[1]
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, (t_0th - t_theo)*2*np.pi*f, lw=1,
           color=colors[1])
#     ax.plot(f, (t_ag - t_theo)*2*np.pi*f, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, None,
            (-0.5, 0.5), 'Phase (radian)', ax=ax)
    def labelright(ax):
        ax.tick_params(labelleft=False,
                      labelright=True)
        ax.yaxis.set_label_position('right')
    labelright(ax)
    ax.set_yticks([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6])
    ax.set_yticks(np.linspace(-np.pi/6, np.pi/6, 9),
            minor=True)
    ax.set_yticklabels(['$-\\pi$/6', '$-\\pi$/12', '0', '$\\pi$/12', '$\\pi$/6'])
    ax.grid(axis='y', which='both')
    
    
    axs3 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.63, bottom=0.56, wspace=0.2))
    
    ax = axs3[0]
    
    c_0th = sw['dist'] / (t_0th + 1/(8*f))
#     c_ag = sw['dist'] / (t_ag + 1/(8*f))
    
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, c_0th - c, lw=1,
           color=colors[1])
#     ax.plot(f, c_ag - c, lw=1,
#            color=colors[2])
    yplot.axis_params(freq_lim, 'Frequency (Hz)',
            (-0.015, 0.015), 'Velocity (km/s)', ax=ax)
    yplot.ticklocator_params(ymajor=0.01, yminor=0.005,
            ymajorgrid=True, yminorgrid=True,
            ax=ax)
    
    ax = axs3[1]
    
    
    ax.plot(f, c, lw=1, color=colors[0])
    ax.plot(f, c_0th, lw=1,
            color=colors[1])
#     ax.plot(f, c_ag, lw=1,
#             color=colors[2])
    #ax.set_xscale('log')
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
        labels=['Theorectial', 'Observed'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs[0].transAxes,
        loc='lower center',
        ncol=2,
        mode='expand')
    
    axs2[0].legend(handles=[line(colors[0]), line(colors[1])],
        labels=['Theorectial', 'Observed'],
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

def plot_illustration_v8(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'r', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7) 
    t_0th, t_0th_idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            refine=True,
            return_index=True)
    
    fig = plt.figure(figsize=yplot.a4, dpi=100)
    gs = fig.add_gridspec(2,2, left=0.2, right=0.7,
                             top=0.95, bottom=0.8,
                            width_ratios=[1,0.5],
                          hspace=0.4,
            )
    
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
                  transform=yplot.offset_transform(-10/72, 0, ax=ax))
        
    waveform_shifted  = sw['seis'] / np.abs(sw['seis']).max() * 0.005 + 0.05
    ax.plot(sw['t'], waveform_shifted, lw=1)
    plot_label(0, sw['t'], waveform_shifted, 'Broadband', ha='right',
               va='center', ax=ax, fontsize=9,
               transform=yplot.offset_transform(-5/72, 0, ax=ax))

        
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
    
    #yplot.auxplot()
    
    axs2 = fig.subplots(1,2, 
            gridspec_kw=dict(left=0.2, right=0.7,
            top=0.73, bottom=0.66, wspace=0.2))
    
    ax = axs2[0]
    #ax.plot(f, t_theo, lw=1, color='r')
    ax.axhline(0, lw=1, color=colors[0])
    ax.plot(f, t_0th - t_theo, lw=1,
           color=colors[1])
#     ax.plot(f, t_ag - t_theo, lw=1,
#            color=colors[2])
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
#     c_ag = sw['dist'] / (t_ag + 1/(8*f))
    
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
#     ax.plot(f, c_ag, lw=1,
#             color=colors[2])
    #ax.set_xscale('log')
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
        labels=['Theorectial', 'Observed'],
        bbox_to_anchor=(0, 1, 1, 0.5),
        bbox_transform=axs[0].transAxes,
        loc='lower center',
        ncol=2,
        mode='expand')
    
    axs2[0].legend(handles=[line(colors[0]), line(colors[1])],
        labels=['Theorectial', 'Observed'],
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

def plot_measurement_type_v1(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'k', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
    f, c = example_ak135_mft_fc(sw)
    t_theo = phaseshift.ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    # gamma = phase_tts[0]['gamma']
    mft = phaseshift.gaussian_filter(sw, f, gamma=gamma,
            emin=7)
#     t_ag = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=True,
#             refine=True,
#             ampratio=1.5)
#     t_ag_uncorrected = ypvt.track_ridge_mft(f, sw['t'],
#             mft, f0=f0, t0=t0,
#             branch_jump=True,
#             corrected=False,
#             refine=True,
#             ampratio=1.5)
    def phasett_each_ridge(iridge=0):
        t_theo = phaseshift.ptt_theo(sw, f, c, iridge=iridge)
        f0 = 0.03
        t0 = np.interp(f0, f, t_theo)
        t_eachridge = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0,
                branch_jump=False,
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
    
    axs = [fig.add_subplot(gs[0,0]),
           fig.add_subplot(gs[1,0])]

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
    ax.plot(sw['t'], seis_filtered / np.abs(seis_filtered).max()/2-1,
           '-k', lw=0.5)
    
    ax.text(1, -1, f'0.03 Hz, $\\alpha$={2*np.pi*0.03*gamma**2:.2g}', ha='right', va='bottom',
           transform=ax.get_yaxis_transform())
    ax.text(1, 0, 'Broadband', ha='right', va='bottom',
           transform=ax.get_yaxis_transform())
#     ax.axvline(400, dashes=(5,1))
#     ax.axvline(550, dashes=(5,1))

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
    ax.text(0.5*(ts[1]+ts[2]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[2]+ts[3]), 0.032, '$2/f$', ha='center', va='top',
           color='lightgray', transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[3]+ts[4]), 0.034, '$3/f$', ha='center', va='top',
           color='lightgray', transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[0]+ts[1]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', transform=yplot.offset_transform(0/72, -1/72, ax=ax))


    yplot.axis_params((400,550), 'Time (s)',
            (0.01, 0.04), 'Frequency (Hz)',
            ax=ax)
    yplot.ticklocator_params(50, 10, 0.01, 0.005, ax=ax)
    
    yplot.labelax(axs, loc='upper left', fontdict=dict(size=9))
    
    
    return Figure(fig)

def plot_measurement_type_v2(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'k', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
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
                branch_jump=False,
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
    yplot.plot_rect(430,-1.5, 470, -0.5, facecolor='lightgray',
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
    ax.annotate('Fig.1c', xytext=(320,-1), xy=(430,-1),
            textcoords=yplot.offset_transform(0,6/72,ax=ax),
            xycoords=yplot.offset_transform(0,6/72,ax=ax),
            ha='left', va='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', shrinkA=0,
                shrinkB=0, lw=0.5),
            )
#     ax.axvline(400, dashes=(5,1))
#     ax.axvline(550, dashes=(5,1))

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
    ax.text(0.5*(ts[1]+ts[2]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[2]+ts[3]), 0.032, '$2/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[3]+ts[4]), 0.034, '$3/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[0]+ts[1]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))


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
    mark_vline(phase_t0, '-k', lw=1)

    phase_tobs = np.interp(0.03, f, t_ridges[1])
    mark_vline(phase_tobs, '-r', lw=1)

    ax.annotate('t = $\Delta$/c', xytext=(460, 0.0),
            xy=(phase_t0,0.0),
            ha='left', va='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', lw=0.5))

    ax.text(0.5*(phase_t0+phase_tobs), 0.5,
            '$\pi$/4 + $\phi$', ha='center',
            va='bottom',
            fontsize=9)
    ax.annotate('', xytext=(phase_tobs,0.4),
            xy=(phase_t0,0.4),
            fontsize=9,
            arrowprops=dict(arrowstyle='<|-|>',
                shrinkA=0, shrinkB=0, lw=0.5))
    ax.text(phase_tobs, -0.1, 't$_{obs}$',
            ha='right', va='center',
            color='r',
            fontsize=9,
            transform=yplot.offset_transform(-2/72,0/72,
                ax=ax))

    yplot.axis_params((430, 470), 'Time (s)',
            None, None,
            left=False, top=False, right=False,
            ax=ax)


    yplot.labelax(axs, loc='upper left', fontdict=dict(size=9))


    return Figure(fig)

def plot_measurement_type_v3(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'k', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
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
                branch_jump=False,
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
    yplot.plot_rect(430,-1.5, 470, -0.5, facecolor='lightgray',
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
    ax.annotate('Fig.1c', xytext=(320,-1), xy=(430,-1),
            textcoords=yplot.offset_transform(0,6/72,ax=ax),
            xycoords=yplot.offset_transform(0,6/72,ax=ax),
            ha='left', va='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', shrinkA=0,
                shrinkB=0, lw=0.5),
            )
#     ax.axvline(400, dashes=(5,1))
#     ax.axvline(550, dashes=(5,1))

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
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[2]+ts[3]), 0.032, '$2/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[3]+ts[4]), 0.034, '$3/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[0]+ts[1]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))


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

    #ax.annotate('t = $\Delta$/c', xytext=(460, 0.0),
    #        xy=(phase_t0,0.0),
    #        ha='left', va='center',
    #        fontsize=9,
    #        arrowprops=dict(arrowstyle='-|>', lw=0.5))
    ax.text(phase_t0, -0.5, 't = $\Delta$/c',
            ha='left', va='bottom',
            fontsize=9, color='r',
            transform=yplot.offset_transform(2/72,1/72,
                ax=ax))
    ax.text(phase_tobs, -0.5, 't$_{obs}$',
            ha='right', va='bottom',
            color='k',
            fontsize=9,
            transform=yplot.offset_transform(-2/72,1/72,
                ax=ax))

    ax.text(0.5*(phase_t0+phase_tobs), 0.5,
            '$f/8 + \delta t$ ($\pi$/4 + $\delta\phi$)', ha='center',
            va='bottom',
            fontsize=9)
    ax.annotate('', xytext=(phase_tobs,0.45),
            xy=(phase_t0,0.45),
            fontsize=9,
            arrowprops=dict(arrowstyle='<|-|>',
                shrinkA=0, shrinkB=0, lw=0.5))
    ax.text(475, 0.45, '$\delta\phi$ - Phase residual',
            fontsize=9,
            ha='right', va='top')

    yplot.axis_params((435, 475), 'Time (s)',
            None, None,
            left=False, top=False, right=False,
            ax=ax)


    yplot.labelax(axs, loc='upper left', fontdict=dict(size=9))


    return Figure(fig)

def plot_measurement_type_v4(sw, gamma=16):
    freq_lim = (0.01, 0.04)
    colors = ['k', 'k', 'g']
    #colors = [yplot.Orange, yplot.Blue, yplot.Green]
    
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
                branch_jump=False,
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
    yplot.plot_rect(430,-1.5, 470, -0.5, facecolor='lightgray',
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
    ax.annotate('Fig.1c', xytext=(320,-1), xy=(430,-1),
            textcoords=yplot.offset_transform(0,6/72,ax=ax),
            xycoords=yplot.offset_transform(0,6/72,ax=ax),
            ha='left', va='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', shrinkA=0,
                shrinkB=0, lw=0.5),
            )
#     ax.axvline(400, dashes=(5,1))
#     ax.axvline(550, dashes=(5,1))

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
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[2]+ts[3]), 0.032, '$2/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[3]+ts[4]), 0.034, '$3/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))
    ax.text(0.5*(ts[0]+ts[1]), 0.03, '$1/f$', ha='center', va='top',
           color='lightgray', 
           fontsize=9,
           transform=yplot.offset_transform(0/72, -1/72, ax=ax))


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

    #ax.annotate('t = $\Delta$/c', xytext=(460, 0.0),
    #        xy=(phase_t0,0.0),
    #        ha='left', va='center',
    #        fontsize=9,
    #        arrowprops=dict(arrowstyle='-|>', lw=0.5))
    ax.text(phase_t0, -0.5, 't = $\Delta$/c',
            ha='left', va='bottom',
            fontsize=9, color='r',
            transform=yplot.offset_transform(2/72,1/72,
                ax=ax))
    ax.text(phase_tobs, -0.5, 't$_{obs}$',
            ha='right', va='bottom',
            color='k',
            fontsize=9,
            transform=yplot.offset_transform(-2/72,1/72,
                ax=ax))

    ax.text(0.5*(phase_t0+phase_tobs), 0.5,
            '$-f/8 + \delta t$ ($-\pi$/4 + $\delta\phi$)', ha='center',
            va='bottom',
            fontsize=9)
    ax.annotate('', xytext=(phase_tobs,0.45),
            xy=(phase_t0,0.45),
            fontsize=9,
            arrowprops=dict(arrowstyle='<|-',
                shrinkA=0, shrinkB=0, lw=0.5))
    ax.text(475, 0.45, '$\delta\phi$ - Phase residual',
            fontsize=9,
            ha='right', va='top')

    yplot.axis_params((435, 475), 'Time (s)',
            None, None,
            left=False, top=False, right=False,
            ax=ax)


    yplot.labelax(axs, loc='upper left', fontdict=dict(size=9))


    return Figure(fig)

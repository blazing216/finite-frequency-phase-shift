import numpy as np
import matplotlib.pyplot as plt

import ygeolib.pvt as ypvt
import ygeolib.utils as yutils

from . import synthetic

def gaussian_filter(sw, f, gamma, emin=7):
    dt = sw['t'][1] - sw['t'][0]
    if np.isscalar(f):
        if np.isscalar(gamma):
            return ypvt.nbfilter(sw['seis'], dt,
                    fc=f, gamma=gamma, emin=emin)
        else:
            res = []
            for g in gamma:
                res.append(ypvt.nbfilter(sw['seis'], dt,
                        fc=f, gamma=g, emin=emin))
            return np.array(res)
    else:
        return ypvt.mft(sw['seis'], dt,
                f, gamma=gamma, emin=emin)


def valid_freq_limits(fmin, fmax, gamma, emin=2.5):
    f = np.linspace(fmin, fmax, 1001)
    wc = 2 * np.pi * f
    alpha = gamma**2 * wc
    fmin = f * (1 - emin/np.sqrt(alpha))
    fmax = f * (1 + emin/np.sqrt(alpha))
    min_idx = np.where(fmin > f[0])[0][0]
    max_idx = np.where(fmax < f[-1])[0][-1]

    return f[min_idx], f[max_idx]

def valid_period_limits(fmin, fmax, gamma, emin=2.5):
    f0, f1 = valid_freq_limits(fmin, fmax,
            gamma, emin=emin)
    return 1.0/f1, 1.0/f0

def ptt_t0(sw, f, c, iridge=0):
    t = sw['dist'] / c + iridge * (1/f)
    return t

def ptt_theo(sw, f, c, iridge=0):
    t = sw['dist'] / c - 1 / (8*f) + iridge * (1/f)
    return t

def ptt_single_frequency(sw, f, c, iridge=0,
        farfield=False, casual_branch_only=True):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)
    
    wvfm = synthetic.surface_wave(f, c, sw['t'], sw['dist'],
            farfield=farfield, casual_branch_only=casual_branch_only,
            return_single_frequencies=True)
    t = []
    for i in range(len(f)):
        t.append(closest_maximum_accurate(t_theo[i],
                sw['t'], wvfm[:,i]))
    return np.array(t)

def ptt_ag(sw, f, c, gamma, emin=7, iridge=0,
        ampratio=1, t0=None, auto_t0=False):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    f0 = f[0]
    dt = sw['t'][1] - sw['t'][0]

    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)

    if auto_t0:
        t_ag = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=None,
                branch_jump=True,
                ampratio=ampratio,
                corrected=True)
    else:
        if t0 is None:
            t0 = np.interp(f0, f, t_theo)

        t_ag = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0,
                branch_jump=True,
                ampratio=ampratio,
                corrected=True)
    return t_ag

def ptt_single_ridge(sw, f, c, gamma, emin=7, iridge=0,
        t0=None, auto_t0=False):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    f0 = f[0]
    dt = sw['t'][1] - sw['t'][0]

    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)

    if auto_t0:
        t_single_ridge = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=None,
                branch_jump=False,
                corrected=True)
    else:
        if t0 is None:
            t0 = np.interp(f0, f, t_theo)

        t_single_ridge = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0,
                branch_jump=False,
                corrected=True)
    return t_single_ridge

def closest_maximum_accurate(t0, t, x):
    t0_idx = np.argmin(np.abs(t-t0))
    peak_idx = ypvt.closest_maximum(x,
        t0_idx)
    peak = ypvt.precise_localmax(t[peak_idx-1:peak_idx+2],
        x[peak_idx-1:peak_idx+2])
    return peak

def ptt_sf(sw, f, t0, gamma, emin=7,
        return_waveform=False):
    '''phase travel time at single frequency
    which cannot use multiple frequency filters
    '''
    seis_filtered = gaussian_filter(sw,
        f=f, gamma=gamma)
    if np.isscalar(gamma):
        seis_filtered.shape = (1,-1)
    nseis = seis_filtered.shape[0]

    #t0 = ptt_theo(sw_ak135, f=f, c=c,
    #        iridge=iridge)
    t = sw['t']
    dt = t[1] - t[0]
    t0_idx = round((t0 - t[0])/dt)

    phase_tt = []
    for i in range(nseis):
        peak_idx = ypvt.closest_maximum(seis_filtered[i,:],
            t0_idx)
        peak = ypvt.precise_localmax(t[peak_idx-1:peak_idx+2],
            seis_filtered[i,peak_idx-1:peak_idx+2])
        phase_tt.append(peak)

    phase_tt = np.array(phase_tt)

    if return_waveform:
        return phase_tt, seis_filtered
    else:
        return phase_tt

def ptt_sf_meier1(sw, f, t0, gamma, emin=7):
    # option 1 (interpolate on fv):
    dt = sw['t'][1] - sw['t'][0]
    b = sw['t'][0]
    freq = np.fft.fftfreq(len(sw['seis']), dt)

    fv = f
    seis_filtered = gaussian_filter(sw,
            f=fv, gamma=gamma)
    seis_spec = np.fft.fft(seis_filtered)
    seis_spec = seis_spec * np.exp(1j * 2* np.pi * freq * (-b))
    phase = np.angle(seis_spec)
    idx = np.argsort(freq)
    phase1 = np.interp(fv, freq[idx], phase[idx])

    # unwrap
    cv = np.interp(fv, sw['f'], sw['c'])
    t0_theo = ptt_theo(sw, fv, cv)
    n = round((t0_theo * 2*np.pi*fv - (-phase1))\
            / (2*np.pi))
    phase1_ur = -phase1 + n1*2*np.pi

    return {'f': fv, 'c':cv,
        'gamma': gamma, 'seis': seis_filtered,
        'phase': phase1_ur,
        'phase_theo': t0_theo * 2*np.pi*fv,
        'ptt': phase1_ur / (2*np.pi*fv),
        't_theo': t0_theo,
        }

def ptt_sf_meier2(sw, f, t0, gamma, emin=7):
    # option 2 (the closest frequency sample to fv):
    dt = sw['t'][1] - sw['t'][0]
    b = sw['t'][0]
    freq = np.fft.fftfreq(len(sw['seis']), dt)

    ifv = np.argmin(np.abs(f - freq))
    fv = freq[ifv]
    seis_filtered = gaussian_filter(sw,
            f=freq[ifv], gamma=gamma)
    seis_spec = np.fft.fft(seis_filtered)
    phase = np.angle(seis_spec[ifv] * \
        np.exp(1j*2*np.pi*freq[ifv]*(-b)))

    # unwrap
    cv = np.interp(fv, sw['f'], sw['c'])
    t0_theo = ptt_theo(sw, fv, cv)
    n = round((t0_theo * 2*np.pi*fv - (-phase))\
            / (2*np.pi))
    phase_ur = -phase + n*2*np.pi

    return {'f': fv, 'c':cv,
        'gamma': gamma, 'seis': seis_filtered,
        'phase': phase_ur,
        'phase_theo': t0_theo * 2*np.pi*fv,
        'ptt': phase_ur / (2*np.pi*fv),
        't_theo': t0_theo,
        }

def ptt_sf_approx(sw, f, gamma):
    # TODO: smoothing the dispersion curve to reduce
    # glitches in the second derivatives of k
    wc = 2 * np.pi * f
    alpha = wc * gamma**2

    k = 2*np.pi*sw['f'] / sw['c']

    omega = 2*np.pi*sw['f']
    domega = omega[1] - omega[0]

    d2kdw2 = yutils.diff(k, dt=domega, n=2)
    d2kdw2_wc = np.interp(wc, omega, d2kdw2)

    # only keep the 4*alpha/wc**2 term due
    # to the assumption of large alpha
    rp1 = 4 * alpha / wc**2
    ip = -d2kdw2_wc * sw['dist']
    dphase = np.angle(rp1+1j*ip)

    # unwrap
    c0 = np.interp(f, sw['f'], sw['c'])
    t0_theo = ptt_theo(sw, f, c0)
    phase = t0_theo * 2*np.pi*f - dphase

    return {'f': f, 'c':c0,
        'gamma': gamma,
        'phase': phase,
        'phase_theo': t0_theo * 2*np.pi*f,
        'ptt': phase / (2*np.pi*f),
        't_theo': t0_theo,
        }

def plot_debug_ptt_sf(sw, f, gamma, t0=None, emin=7):
    '''phase travel time at single frequency
    which cannot use multiple frequency filters
    '''
    seis_filtered = gaussian_filter(sw,
        f=f, gamma=gamma)
    if np.isscalar(gamma):
        seis_filtered.shape = (1,-1)
    nseis = seis_filtered.shape[0]

    if t0 is None:
        c = np.interp(f, sw['f'],
                sw['c'])
        t0 = ptt_theo(sw, f=f, c=c,
                iridge=0)
    t = sw['t']
    dt = t[1] - t[0]
    t0_idx = round((t0 - t[0])/dt)

    phase_tt = []
    for i in range(nseis):
        peak_idx = ypvt.closest_maximum(seis_filtered[i,:],
            t0_idx)
        peak = ypvt.precise_localmax(t[peak_idx-1:peak_idx+2],
            seis_filtered[i,peak_idx-1:peak_idx+2])
        phase_tt.append(peak)

    phase_tt = np.array(phase_tt)

    fig = plt.figure(figsize=(3,2.5))
    plt.subplot(111)
    for i in range(seis_filtered.shape[0]):
        plt.plot(t, seis_filtered[i,:] + i)
        peak_value = np.interp(phase_tt[i],
                t,
                seis_filtered[i,:]+i)
        plt.plot(phase_tt[i], peak_value, '|',
                mec='r')
    plt.xlim(100, 500)
    plt.gca().set_yticks(np.arange(seis_filtered.shape[0]))
    plt.gca().set_yticklabels(['%.3g' % g for g in gamma])
    plt.axvline(t0, color='r', lw=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('$\gamma$')
    plt.title('f = %g Hz' % f)

    plt.tight_layout()

    return phase_tt

def plot_debug_ptt_sf_meier(sw, f, gamma, t0=None, emin=7):
    '''phase travel time at single frequency
    which cannot use multiple frequency filters
    '''
    seis_filtered = gaussian_filter(sw,
        f=f, gamma=gamma)
    if np.isscalar(gamma):
        seis_filtered.shape = (1,-1)
    nseis = seis_filtered.shape[0]

    if t0 is None:
        c = np.interp(f, sw['f'],
                sw['c'])
        t0 = ptt_theo(sw, f=f, c=c,
                iridge=0)
    t = sw['t']
    dt = t[1] - t[0]
    t0_idx = round((t0 - t[0])/dt)

    phase_tt = []
    for i in range(nseis):
        peak_idx = ypvt.closest_maximum(seis_filtered[i,:],
            t0_idx)
        peak = ypvt.precise_localmax(t[peak_idx-1:peak_idx+2],
            seis_filtered[i,peak_idx-1:peak_idx+2])
        phase_tt.append(peak)

    phase_tt = np.array(phase_tt)

    fig = plt.figure(figsize=(3,2.5))
    plt.subplot(111)
    for i in range(seis_filtered.shape[0]):
        plt.plot(t, seis_filtered[i,:] + i)
        peak_value = np.interp(phase_tt[i],
                t,
                seis_filtered[i,:]+i)
        plt.plot(phase_tt[i], peak_value, '|',
                mec='r')
    plt.xlim(100, 500)
    plt.gca().set_yticks(np.arange(seis_filtered.shape[0]))
    plt.gca().set_yticklabels(['%.3g' % g for g in gamma])
    plt.axvline(t0, color='r', lw=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('$\gamma$')
    plt.title('f = %g Hz' % f)

    plt.tight_layout()

    return phase_tt

def plot_debug_ptt_sf_approx(sw, f, t0=None, gamma=16, t=None):
    # approx valid for large gamma (narrow
    # band filters)
    wc = 2 * np.pi * f
    alpha = wc * gamma**2

    k = 2*np.pi*sw['f'] / sw['c']
    omega = 2*np.pi*sw['f']
    domega = omega[1] - omega[0]

    dkdw = yutils.diff(k, dt=domega)
    d2kdw2 = yutils.diff(k, dt=domega, n=2)

    dkdw_wc = np.interp(wc, omega, dkdw)
    d2kdw2_wc = np.interp(wc, omega, d2kdw2)

    if t is None:
        t = sw['t']

    if t0 is None:
        c0 = np.interp(f, sw['f'], sw['c'])
        t0 = ptt_theo(sw, f, c0, iridge=0)

    rp = 4 * alpha / wc**2 - \
        (t - dkdw_wc*sw['dist'])**2
    ip = -d2kdw2_wc * sw['dist']

    phase = np.angle(rp+1j*ip)

    plt.figure(figsize=(5,6))
    plt.subplot(321)

    rp1 = np.ones_like(t)*4*alpha/wc**2
    rp2 = -(t-dkdw_wc*sw['dist'])**2
    plt.plot(t, rp1, '-k',
            label='$4\\alpha/\\omega_c^2$')
    plt.plot(t, rp2, '-r',
            label="$-f'^2(\\omega_c)$")
    plt.plot(t, rp1+rp2, '-g',
            label='all')

    xlim = (100,400)
    sel = (t >= xlim[0]) & (t <= xlim[1])
    ylimit = (min(rp1[sel].min(), rp2[sel].min()),
              max(rp1[sel].max(), rp2[sel].max()))
    ylim = (ylimit[0] - 0.1 * (ylimit[1]-ylimit[0]),
           ylimit[1] + 0.1 * (ylimit[1]-ylimit[0]))
    ylim = (rp1[0] * 0.5, rp1[0] * 1.5)
    plt.axvline(t0, color='k', dashes=(10,10))
    plt.axhline(0, dashes=(10,10), color='orange')
    plt.xlabel('Time (s)')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title('Real part')

    plt.legend()

    plt.subplot(322)
    ip = np.ones_like(t) * (-d2kdw2_wc) * sw['dist']
    plt.plot(t, ip)
    ylim = (ip[0] *0.5, ip[0] * 1.5)
#     ylimit = (ip[sel].min(), ip[sel].max())
#     ylim = (ylimit[0] - 0.1 * (ylimit[1]-ylimit[0]),
#            ylimit[1] + 0.1 * (ylimit[1]-ylimit[0]))

    plt.xlabel('Time (s)')
    plt.title('Imag part')
    plt.xlim(*xlim)
    plt.ylim(*ylim)


    plt.subplot(323)
    dphase = np.angle(rp1[sel]+rp2[sel]+1j*ip[sel])
    plt.plot(t[sel], dphase)
    #plt.axis('equal')
    plt.axhline(0, dashes=(10,10), color='orange')
    plt.xlim(*xlim)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (radian)')
    plt.title('Phase')

    plt.subplot(324)
    plt.plot(t[sel], -dphase)
    plt.xlabel('Time (s)')
    plt.ylabel('Phase shift (radian)')
    plt.axhline(0, dashes=(10,10), color='orange')
    plt.xlim(*xlim)
    plt.title('Phase shift')

    plt.subplot(325)
    plt.plot(t[sel], -dphase / (2*np.pi*f))
    plt.xlabel('Time (s)')
    plt.ylabel('Time difference (s)')
    plt.axhline(0, dashes=(10,10), color='orange')
    plt.xlim(*xlim)
    plt.title('Travel time difference')


    plt.subplot(326)
    plt.plot(sw['f'], d2kdw2)
    plt.axhline(0, dashes=(10,10), color='orange')
    plt.axvline(f, dashes=(10,10), color='orange')
    plt.semilogx()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('$d^2kd\\omega^2$')

    plt.tight_layout()


def plot_debug_ptt_ag(sw, gamma, emin=7, iridge=0,
        ampratio=1):
    f, c = exmaple_ak135_mft_fc(sw)
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)
    t_ag, idx = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=True,
            corrected=True,
            ampratio=ampratio,
            return_index=True)
    fig = plt.figure(figsize=(4,3))
    ypvt.plot_mft(mft, sw['t'], f, cmap='gray')
    plt.plot(t_theo, f, color='r', lw=0.5)
    plt.plot(t_ag, f, color='green', lw=0.5,
            dashes=(10,5))
    plt.plot(sw['t'][idx], f, color='blue', lw=0.5,
            dashes=(10,5))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    return fig

def plot_debug_ridge_amp(sw, gamma, emin=7, iridge=0):
    if np.isscalar(iridge):
        iridge = np.array([iridge])
    else:
        iridge = np.array(iridge)

    f, c = exmaple_ak135_mft_fc(sw)
    t_theo = ptt_theo(sw, f, c, iridge=0)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)
    print(mft.shape)

    res = []
    for ir in iridge:
        t_sg, idx = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0+ir/f0,
                branch_jump=False,
                corrected=True,
                return_index=True)
        res.append({'t':t_sg, 'idx':idx})

    #return mft, sw, res

    fig = plt.figure(figsize=(4,2))

    plt.subplot(121)

    ypvt.plot_mft(mft, sw['t'], f, cmap='gray')
    plt.plot(t_theo, f, color='r', lw=0.5)
    for each_res in res:
        plt.plot(each_res['t'], f, color='blue', lw=0.5, dashes=(10,5))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(0, 1000)

    plt.subplot(122)

    colors = ['r', 'b', 'g', 'c']
    for i, each_res in enumerate(res):
        plt.plot(f,
            mft[np.arange(len(f)),
                each_res['idx']], color=colors[i%4])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')

    plt.tight_layout()

    return fig

def ptt_sg(sw, f, c, gamma, emin=7, iridge=0):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    f0 = f[0]
    t0 = np.interp(f0, f, t_theo)

    dt = sw['t'][1] - sw['t'][0]
    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)
    t_ag = ypvt.track_ridge_mft(f, sw['t'],
            mft, f0=f0, t0=t0,
            branch_jump=False,
            corrected=True)
    return t_ag

def ptt2ps(t, t_theo, f):
    return (t - t_theo) * 2 * np.pi * f

def ptt2c(t, f, dist, iridge):
    return dist / (t + 1/(8*f) - iridge / f)

def example_ak135_ptt_ag(sw_ak135, gamma, **kwargs):
    f, c = exmaple_ak135_mft_fc(sw_ak135)
    ptt = ptt_ag(sw_ak135, f, c, gamma=gamma,
            emin=7, **kwargs)
    return {'f':f, 'c':c, 'ptt':ptt}

def example_ak135_ptt_theo(sw_ak135, iridge=0):
    f, c = exmaple_ak135_mft_fc(sw_ak135)
    ptt = ptt_theo(sw_ak135, f, c, iridge=iridge)
    return {'f':f, 'c':c, 'ptt':ptt}

def exmaple_ak135_mft_fc(sw_ak135):
    f = np.logspace(np.log10(sw_ak135['f'][0]),
        np.log10(sw_ak135['f'][-1]),
        501)
    c = np.interp(f, sw_ak135['f'], sw_ak135['c'])
    return f, c

# debug phase shift
def debug_phase_shift(sw_ak135, gamma, **kwargs):
    t0 = example_ak135_ptt_theo(sw_ak135, iridge=0)
    t1 = example_ak135_ptt_ag(sw_ak135, gamma=gamma, **kwargs)
    ps = (t1['ptt'] - t0['ptt']) * 2 * np.pi * t0['f']
    return t0, t1, ps

def plot_debug_phase_shift(sw_ak135, gamma):
    t0, t1, ps = debug_phase_shift(sw_ak135, gamma)

    fig = plt.figure(figsize=(6,5))

    plt.subplot(221)
    plt.plot(t0['f'], t0['ptt'], '-k')
    plt.plot(t1['f'], t1['ptt'], '-r')
    plt.title('Travel time')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Travel time (s)')

    plt.subplot(222)
    plt.axhline(0, color='k')
    plt.plot(t1['f'], t1['ptt'] - t0['ptt'], '-r')
    plt.title('Travel time difference')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dt (s)')

    plt.subplot(223)
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

    return fig
def plot_debug_phase_shift(sw_ak135, gamma):
    t0, t1, ps = debug_phase_shift(sw_ak135, gamma)

    fig = plt.figure(figsize=(6,5))

    plt.subplot(221)
    plt.plot(t0['f'], t0['ptt'], '-k')
    plt.plot(t1['f'], t1['ptt'], '-r')
    plt.title('Travel time')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Travel time (s)')

    plt.subplot(222)
    plt.axhline(0, color='k')
    plt.plot(t1['f'], t1['ptt'] - t0['ptt'], '-r')
    plt.title('Travel time difference')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dt (s)')

    plt.subplot(223)
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

    return fig

# debug dc/c
def debug_dc_over_c(sw_ak135, gamma):
    t0 = example_ak135_ptt_theo(sw_ak135, iridge=0)
    t1 = example_ak135_ptt_ag(sw_ak135, gamma=gamma, **kwargs)

    c0 = sw_ak135['dist'] / (t0['ptt'] + 1 / (8*t0['f']))
    c1 = sw_ak135['dist'] / (t1['ptt'] + 1 / (8*t1['f']))

    return t0, t1, c0, c1

def plot_debug_dc_over_c(sw_ak135, gamma):
    t0, t1, c0, c1 = debug_dc_over_c(sw_ak135, gamma)

    fig = plt.figure(figsize=(6,5))

    plt.subplot(221)
    plt.plot(t0['f'], t0['ptt'], '-k')
    plt.plot(t1['f'], t1['ptt'], '-r')
    plt.title('Travel time')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Travel time (s)')

    plt.subplot(222)
    plt.plot(t0['f'], c0, '-k')
    plt.plot(t1['f'], c1, '-r')
    plt.title('Phase velocity')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase velocity (km/s)')

    plt.subplot(223)
    plt.axhline(0)
    plt.plot(t0['f'],
             c1 - c0,
             '-r')
    plt.title('dc')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dc (km/s)')

    plt.subplot(224)
    plt.axhline(0)
    plt.plot(t0['f'],
             (c1 - c0)/c0,
             '-r')
    plt.axhline(-0.01)
    plt.axhline(0.01)

    ylim = plt.gca().get_ylim()
    plt.title('dc/c')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dc/c')

    ax = plt.gca().twinx()
    #ylim = plt.gca().get_ylim()
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(\
        plt.FuncFormatter(lambda x, pos: '%g' % (x*100)))
    ax.set_ylabel('dc/c (%)')

    plt.tight_layout()

    return fig

def plot_debug_dc_over_c_logT(sw_ak135, gamma):
    t0, t1, c0, c1 = debug_dc_over_c(sw_ak135, gamma)

    fig = plt.figure(figsize=(6,5))

    plt.subplot(221)
    plt.plot(1/t0['f'], t0['ptt'], '-k')
    plt.plot(1/t1['f'], t1['ptt'], '-r')
    plt.semilogx()
    plt.title('Travel time')
    plt.xlabel('Period (s)')
    plt.ylabel('Travel time (s)')

    plt.subplot(222)
    plt.plot(1/t0['f'], c0, '-k')
    plt.plot(1/t1['f'], c1, '-r')
    plt.semilogx()
    plt.title('Phase velocity')
    plt.xlabel('Period (s)')
    plt.ylabel('Phase velocity (km/s)')

    plt.subplot(223)
    plt.axhline(0)
    plt.plot(1/t0['f'],
             c1 - c0,
             '-r')
    plt.semilogx()
    plt.title('dc')
    plt.xlabel('Period (s)')
    plt.ylabel('dc (km/s)')

    plt.subplot(224)
    plt.axhline(0)
    plt.plot(1/t0['f'],
             (c1 - c0)/c0,
             '-r')
    plt.semilogx()
    plt.axhline(-0.01)
    plt.axhline(0.01)

    ylim = plt.gca().get_ylim()
    plt.title('dc/c')
    plt.xlabel('Period (s)')
    plt.ylabel('dc/c')

    ax = plt.gca().twinx()
    #ylim = plt.gca().get_ylim()
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(\
        plt.FuncFormatter(lambda x, pos: '%g' % (x*100)))
    ax.set_ylabel('dc/c (%)')

    plt.tight_layout()

    return fig

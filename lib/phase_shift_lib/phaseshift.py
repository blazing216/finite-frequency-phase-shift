import numpy as np

import pvt as ypvt

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

def ptt_theo(sw, f, c, iridge=0):
    t = sw['dist'] / c - 1 / (8*f) + iridge * (1/f)
    return t

def ptt_single_frequency(sw, f, c, iridge=0,
        farfield=False, casual_branch_only=True):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    wvfm = synthetic.surface_wave_ncf(f, c, sw['t'], sw['dist'],
            farfield=farfield, casual_branch_only=casual_branch_only,
            return_single_frequencies=True)
    t = []
    for i in range(len(f)):
        t.append(closest_maximum_accurate(t_theo[i],
                sw['t'], wvfm[:,i]))
    return np.array(t)

def ptt_ag(sw, f, c, gamma, emin=7, iridge=0,
        t0=None, auto_t0=False):
    t_theo = ptt_theo(sw, f, c, iridge=iridge)

    f0 = f[0]
    dt = sw['t'][1] - sw['t'][0]

    mft = gaussian_filter(sw, f, gamma=gamma,
            emin=emin)

    if auto_t0:
        t_ag = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=None,
                refine=True)
    else:
        if t0 is None:
            t0 = np.interp(f0, f, t_theo)

        t_ag = ypvt.track_ridge_mft(f, sw['t'],
                mft, f0=f0, t0=t0,
                refine=True)
    return t_ag

def closest_maximum_accurate(t0, t, x):
    t0_idx = np.argmin(np.abs(t-t0))
    peak_idx = ypvt.closest_maximum(x,
        t0_idx)
    peak = ypvt.precise_localmax(t[peak_idx-1:peak_idx+2],
        x[peak_idx-1:peak_idx+2])
    return peak

import numpy as np
from scipy.special import hankel2, jv

def surface_wave_ncf(f, c, t, dist, farfield=False,
                casual_branch_only=True,
                return_single_frequencies=False):
    assert np.all(f>=0), "Formula used is incorrect for negative frequencies"

    df = f[1] - f[0]
    omega = 2 * np.pi * f

    t_colvec = t.reshape(-1,1)

    if casual_branch_only:
        if farfield:
            wvfm = 1/(np.pi) * np.sqrt(c/f*dist) * \
                np.cos(2*np.pi*f*(t_colvec - dist/c) + np.pi/4)
        else:
            wvfm = 1 * \
                hankel2(0, 2*np.pi*f*dist/c) * \
                np.exp(1j * 2*np.pi*f*t_colvec)
    else:
        if farfield:
            wvfm = 1/(np.pi) * np.sqrt(c/f*dist) * \
                np.cos(2*np.pi*f* (- dist/c) + np.pi/4) * \
                np.cos(2*np.pi*f*t_colvec)
        else:
            wvfm = 2 * \
                jv(0, 2*np.pi*f*dist/c) * \
                np.cos(2*np.pi*f*t_colvec)

    if return_single_frequencies:
        return wvfm.real
    else:
        nf = len(f)
        weight = np.ones(nf)
        weight[0] = 0.5
        weight[-1] = 0.5
        wvfm = np.sum(weight * wvfm, axis=1) * df
        wvfm = wvfm.real
        return wvfm

def surface_wave_ballistic(f, c, t, dist,
                casual_branch_only=True,
                return_single_frequencies=False):
    assert np.all(f>=0), "Formula used is incorrect for negative frequencies"

    df = f[1] - f[0]
    omega = 2 * np.pi * f

    t_colvec = t.reshape(-1,1)

    if casual_branch_only:
        wvfm = -1j * 2*np.pi*f * \
            hankel2(0, 2*np.pi*f*dist/c) * \
            np.exp(1j * 2*np.pi*f*t_colvec)
    else:
        wvfm = -1j * 2*np.pi*f * 2 * \
            jv(0, 2*np.pi*f*dist/c) * \
            np.cos(2*np.pi*f*t_colvec)

    if return_single_frequencies:
        return wvfm.real
    else:
        nf = len(f)
        weight = np.ones(nf)
        weight[0] = 0.5
        weight[-1] = 0.5
        wvfm = np.sum(weight * wvfm, axis=1) * df
        wvfm = wvfm.real
        return wvfm

def single_frequency_surface_wave(f, c, t, dist):
    assert f>=0, "Formula used is incorrect for negative frequencies"
    wvfm = 1 * \
        hankel2(0, 2*np.pi*f*dist/c) * \
        np.exp(1j * 2*np.pi*f*t)
    wvfm = wvfm.real
    return wvfm

def single_frequencies_surface_wave(f, c, t, dist):
    assert np.all(f>=0), "Formula used is incorrect for negative frequencies"
    assert len(f) == len(c)
    nf = len(f)
    return np.column_stack([single_frequency_surface_wave(f[i], c[i], t, dist) for i in range(nf)])

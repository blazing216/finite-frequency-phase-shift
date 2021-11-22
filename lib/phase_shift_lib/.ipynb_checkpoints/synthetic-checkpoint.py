import numpy as np
from scipy.special import hankel1, jv

from ygeolib.layermodel import Model1D
from ygeolib import disp_xu as disp

def get_ak135_model_file():
    return ('/mnt/seismodata2/YX/Marathon/Project_Phase_shift'
        '/src/phase_shift_lib/mtak135sph.mod')

def surface_wave(f, c, t, dist, farfield=False,
                casual_branch_only=True,
                return_single_frequencies=False):
    df = f[1] - f[0]
    omega = 2 * np.pi * f

    t_colvec = t.reshape(-1,1)

    if casual_branch_only:
        if farfield:
            wvfm = 1/(np.pi) * np.sqrt(c/f*dist) * \
                np.cos(2*np.pi*f*(t_colvec - dist/c) + np.pi/4)
        else:
            # TODO: use hankel2(0, 2*np.pi*f*dist/c) to replace
            # - hankel1(0, - 2*np.pi*f*dist/c)
            # make sure to test
            wvfm = 1 * \
                - hankel1(0, - 2*np.pi*f*dist/c) * \
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

def single_frequency_surface_wave(f, c, t, dist):
    wvfm = 1 * \
        - hankel1(0, - 2*np.pi*f*dist/c) * \
        np.exp(1j * 2*np.pi*f*t)
    wvfm = wvfm.real
    return wvfm

def single_frequencies_surface_wave(f, c, t, dist):
    assert len(f) == len(c)
    nf = len(f)
    return np.column_stack([single_frequency_surface_wave(f[i], c[i], t, dist) for i in range(nf)])
    
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

def surface_wave_for_paper(dist):
    f, t = example_para_ak135()
    c = cR_ak135(f)
    seis = surface_wave(f, c, t, dist=dist)
    return {'t': t, 'seis': seis, 'f':f,
            'c':c, 'dist':dist}

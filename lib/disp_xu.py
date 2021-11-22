#-*-coding:utf-8-*-
""" Dispersion and Kernels using my wrapper of surf96

Changelog:
    7 Apr 2021, Yihe Xu, uR supports fast, which computes group 
        velocity using centre difference
    7 Apr 2021, Yihe Xu, add uR_kernel (Rayleigh + group)
    6 Apr 2021, Yihe Xu, add cR_kernel computes kernels
    6 Apr 2021, Yihe Xu, cR, uR, cL, uL supports spherical
        Earth correctly.
    6 Apr 2021, Yihe Xu, add senker_elastic to compute depth
        kernels
    6 Apr 2021, Yihe Xu, rename disp_full_model to disp_elastic
    3 Apr 2021, Yihe Xu, modified from dispinv.py
"""
import numpy as np
import surf96py as surf96_wrapper
from velrho import vs2vp, vp2rho

def cR(m, t, mode=0):
    '''Rayleigh wave phase velocity

    m: layermodel.Model1D
    '''
    if m.flat_Earth:
        c = disp_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=True,
            wavetype='rayleigh', velocity='phase')
    else:
        c,_,_,_ = senker_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=False,
            wavetype='rayleigh', velocity='phase')
    return c

def cR_kernel(m, t, mode=0):
    '''Depth sensitivity kernels for Rayleigh + phase
    '''
    c, dcda, dcdb, dcdh = senker_elastic(m.thk, m.vp, m.vs, m.rho,
        t, mode=mode, flat_earth=False,
        wavetype='rayleigh', velocity='phase')
    return c, dcda, dcdb, dcdh


def uR(m, t, mode=0, flat_earth=True, fast=False):
    '''Rayleigh wave group velocity

    m: layermodel.Model1D
    '''
    if m.flat_Earth and fast:
        c = disp_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=True,
            wavetype='rayleigh', velocity='group')
    else:
        c,_,_,_ = senker_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=m.flat_Earth,
            wavetype='rayleigh', velocity='group')
    return c

def uR_kernel(m, t, mode=0):
    '''Depth sensitivity kernels for Rayleigh + group
    '''
    c, dcda, dcdb, dcdh = senker_elastic(m.thk, m.vp, m.vs, m.rho,
        t, mode=mode, flat_earth=False,
        wavetype='rayleigh', velocity='group')
    return c, dcda, dcdb, dcdh

def cL(m, t, mode=0, flat_earth=True):
    '''Love wave phase velocity

    m: layermodel.Model1D
    '''
    if m.flat_Earth:
        c = disp_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=True,
            wavetype='love', velocity='phase')
    else:
        c,_,_,_ = senker_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=False,
            wavetype='love', velocity='phase')
    return c

def uL(m, t, mode=0, flat_earth=True, fast=False):
    '''Love wave group velocity

    m: layermodel.Model1D
    '''
    if m.flat_Earth and fast:
        c = disp_elastic(m.thk, m.vp, m.vs, m.rho,
            t, mode=mode, flat_earth=True,
            wavetype='love', velocity='group')
    else:
        raise NotImplementedError
        #c,_,_,_ = senker_elastic(m.thk, m.vp, m.vs, m.rho,
        #    t, mode=mode, flat_earth=m.flat_Earth,
        #    wavetype='love', velocity='group')
    return c

def disp_elastic(thk, vp, vs, rho, t, flat_earth=True, mode=0, wavetype='rayleigh',
        velocity='phase'):
    '''Dispersion curves at periods given by `t` (in seconds)
    The model is given by thk (km) and vs (km/s).
    '''
    iflsph = 0 if flat_earth else 1

    model = np.column_stack([thk, vp, vs, rho])

    # get unique periods
    tu, indices = np.unique(t, return_inverse=True)

    cgu = surf96_wrapper.disper(model, tu,
        iflsph, wavetype=wavetype,
        velocitytype=velocity, wave_mode=mode)

    # back to all periods
    cg = cgu[indices]

    return cg

def senker_elastic(thk, vp, vs, rho, t, flat_earth=True, mode=0, wavetype='rayleigh',
        velocity='phase'):
    '''Depth sensitivity kernels at periods given by `t` (in seconds)
    The model is given by thk (km) and vs (km/s).
    '''
    iflsph = 0 if flat_earth else 1

    model = np.column_stack([thk, vp, vs, rho])

    # get unique periods
    tu, indices = np.unique(t, return_inverse=True)

    if velocity.upper() == 'BOTH':
        cgu, ugu, dcdau, dcdbu, dcdhu, dudau, dudbu, dudhu = \
            surf96_wrapper.senker(model, tu,
            iflsph, wavetype=wavetype,
            velocitytype='both', wave_mode=mode)

        # back to all periods
        cg = cgu[indices]
        dcda = dcdau[indices,:]
        dcdb = dcdbu[indices,:]
        dcdh = dcdhu[indices,:]
        ug = ugu[indices]
        duda = dudau[indices,:]
        dudb = dudbu[indices,:]
        dudh = dudhu[indices,:]

        return cg, ug, dcda, dcdb, dcdh, duda, dudb, dudh

    else:
        cgu, dcdau, dcdbu, dcdhu = surf96_wrapper.senker(model, tu,
            iflsph, wavetype=wavetype,
            velocitytype=velocity, wave_mode=mode)

        # back to all periods
        cg = cgu[indices]
        dcda = dcdau[indices,:]
        dcdb = dcdbu[indices,:]
        dcdh = dcdhu[indices,:]

        return cg, dcda, dcdb, dcdh

def disp(thk, vs, t, flat_earth=True, mode=0, Rayleigh=True, Phase=True):
    '''Dispersion curves at periods given by `t` (in seconds)
    The model is given by thk (km) and vs (km/s).
    '''
    vp = vs2vp(vs)
    rho = vp2rho(vp)

    cg = disp_full_model(thk, vp, vs, rho, t,
        flat_earth=flat_earth, mode=mode,
        Rayleigh=Rayleigh, Phase=Phase)

    return cg


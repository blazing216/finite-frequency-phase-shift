#-*-coding:utf-8-*-
"""Layered model

The model assumes 1-D layered model. Each layer is denoted by
(1) layered thickness and (2) velocity/density

Layer thickness is in km, velocity (P and S) are in km/s,
density are in g/cm^3.

The last layer represents the bottom half space. Its thickness
is meaningless, so it should be set to 0.

Changelog:
    3 Apr 2021, Yihe Xu, modified from dispinv.py
    4 Apr 2021, Yihe Xu, fix a bug in `stepwise` of the result being
        dependent on the shape of the input
    4 Apr 2021, Yihe Xu, add property function `bottom` which can be
        used as the maximum depth when plotting
    4 Apr 2021, Yihe Xu, fix a bug in `stepwise` of an additional
        tranpose.
"""
from __future__ import print_function, division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from velrho import vs2vp, vp2rho

def _extract_one_kwarg(kwarg, kwargs, default=None):
        if kwarg in kwargs:
            kwarg = kwargs.pop(kwarg)
        else:
            kwarg = default
        return kwarg, kwargs

class Model1D:
    def __init__(self, thk, vp, vs, rho, flat_Earth=True):
        self.thk = np.array(thk, dtype=float)
        self.vp  = np.array(vp,  dtype=float)
        self.vs  = np.array(vs,  dtype=float)
        self.rho = np.array(rho, dtype=float)
        self.flat_Earth = flat_Earth

    @classmethod
    def read_cps(cls, model_file):
        '''Read a model from a file in CPS330 format

        The fist 12 lines are the meta information of the model.
        Model parameters are given since line 13.
        The first 4 columns are layer thickness, vp, vs and density.
        For elastic models, the first 4 columns are enough.

        Model1D.read_cps(model_file)
        model_file: string
            The name of the model file.
        '''
        with open(model_file, 'r') as mf:
            lines = mf.readlines()
        title = lines[1].strip()
        iiso_options = {'ISOTROPIC':0,
            'TRANSVERSE ISOTROPIC':1,
            'ANISOTROPIC':2}
        iiso = iiso_options[lines[2].strip().upper()]
        iunit = 0
        iflsph_options = {'FLAT EARTH': True, 'SPHERICAL EARTH': False}
        flat_Earth = iflsph_options[lines[4].strip().upper()]
        idimen_options = {'1-D':1, '2-D':2, '3-D':3}
        idimen = idimen_options[lines[5].strip().upper()]
        icnvel_options = {'CONSTANT VELOCITY':0, 'VARIABLE VELOCITY':1}
        icnvel = icnvel_options[lines[6].strip().upper()]

        data = np.loadtxt(model_file, skiprows=12, usecols=(0,1,2,3))
        thk, vp, vs, rho = data[:,0], data[:,1], data[:,2], data[:,3]
        return cls(thk, vp, vs, rho, flat_Earth=flat_Earth)

    def to_cps_str(self, decimal=4):
        '''Print out the model into a CPS330 format.
        Assumptions:
        (1) Isotropic
        (2) Flat earth (instead of spherical earth
        (3) Elastic

        m.to_cps(decimal=4)
        decimal: int. Default: 4
            The number of digits after the decimal point when printing
            a float point number
        '''
        flat_Earth_options = {True:'FLAT EARTH', False: 'SPHERICAL EARTH'}
        header = f'''MODEL.01
MODEL CREATED BY DISPINV
ISOTROPIC
KGS
{flat_Earth_options[self.flat_Earth]}
1-D
CONSTANT VELOCITY
LINE08
LINE09
LINE10
LINE11
H VP VS RHO QP QS ETAP ETAS FREQP FREQS
'''
        layer_string = []
        for h, p, s, r in zip(self.thk, self.vp, self.vs, self.rho):
            layer_string.append((f'{h:.{decimal}f} {p:.{decimal}f} {s:.{decimal}f} {r:.{decimal}f} '
                                f'{0:.{decimal}f} {0:.{decimal}f} {0:.{decimal}f} {0:.{decimal}f} '
                                f'{1:.{decimal}f} {1:.{decimal}f}\n'
                                ))
        return header + ''.join(layer_string)

    def to_cps(self, model_file, decimal=4):
        '''Print out the model into a CPS330 format.
        Assumptions:
        (1) Isotropic
        (2) Elastic

        m.to_cps(decimal=4)
        model_file: string
            Name of the output file

        decimal: int. Default: 4
            The number of digits after the decimal point when printing
            a float point number
        '''
        with open(model_file, 'w') as f:
            f.write(self.to_cps_str(decimal=decimal))

    @classmethod
    def from_thk_vs(cls, thk, vs):
        '''Generate vp and density from vs
        using `velrho`. Useful for dispersion inversion.
        '''
        thk = np.array(thk)
        vs = np.array(vs)
        vp = vs2vp(vs)
        rho = vp2rho(vp)
        return cls(thk, vp, vs, rho)

    @property
    def bottom(self):
        dep = self.depth(mode='interface', maxdepth=None)
        # dep[-2] is the top of the bottom half space
        return dep[-2]

    def depth(self, mode='interface', maxdepth=None):
        '''get depths of the model

        mode == 'interface': Depth of all interfaces. (n+1) interfaces for
        a n-layer model. The first interface is 0 (surface). The last
        interface (bottom of the bottom half space) is set to `maxdepth`,
        if provided and deeper than the second last inerface depth[n-1].
        Otherwise, it is set to 2 * depth[n-1]

            interface[0] = 0
            interface[1] = thk[0]
            interface[2] = thk[0] + thk[1]
            interface[n-1] = sum(thk[i], i = 0 to n-2)
            note: thk[n-1] == 0
            interface[n] = maxdepth     if maxdepth is not None 
                                        and larger than interface[n-1]
                     2 * interface[n-1] else

        mode == 'middle': Depth of middle points of all layers. n middle points
        for a n-layer model. Middle points of the bottom half space is determined
        by 0.5 * (interface[n-1] + interface[n]

            middle[0] = 0.5 * thk[0] = 0.5 * (interface[0] + interface[1])
            middle[1] = thk[0] + 0.5 * thk[1] = 0.5 * (interface[1] + interface[2])
            middle[i] = 0.5 * (interface[i] + interface[i+1])
            middle[n-2] = 0.5 * (interface[n-2] + interface[n-1])
            middle[n-1] = 0.5 * (interface[n-1] + interface[n])
        '''
        dep = np.hstack([0.0, np.cumsum(self.thk)])
        if maxdepth is not None and maxdepth > dep[-2]:
            dep[-1] = maxdepth
        else:
            dep[-1] = dep[-2] * 2.0

        if mode == 'interface':
            return dep
        elif mode == 'middle':
            return 0.5 * (dep[1:] + dep[:-1])

    def stepwise(self, maxdepth=None):
        dep = self.depth(mode='interface', maxdepth=maxdepth)
        stepwise_dep = np.column_stack([dep[:-1], dep[1:]]).flatten()
        stepwise_vp  = np.column_stack([self.vp, self.vp]).flatten()
        stepwise_vs  = np.column_stack([self.vs, self.vs]).flatten()
        stepwise_rho  = np.column_stack([self.rho, self.rho]).flatten()

        return stepwise_dep, stepwise_vp, stepwise_vs, stepwise_rho

    def plot(self, **kwargs):
        ax, kwargs = _extract_one_kwarg('ax', kwargs, plt.gca())
        maxdepth, kwargs = _extract_one_kwarg('maxdepth', kwargs,
                                             None)

        dep, vp, vs, rho = self.stepwise(maxdepth)

        ax.plot(vp, dep, color='k', **kwargs)
        ax.plot(vs, dep, color='b', **kwargs)
        ax.plot(rho, dep, color='r', **kwargs)



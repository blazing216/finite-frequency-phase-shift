#-*-coding:utf-8-*-
import numpy as np

def linbins(x):
    '''get bins for a linearly spaced series x'''
    dx = x[1] - x[0]
    xbins = np.hstack([x, x[-1]+dx]) - 0.5 * dx
    return xbins

def logbins(x):
    '''get bins for a logarithmically spaced series x'''
    dx = x[1] / x[0]
    sqrt_dx = np.sqrt(dx)
    xbins = np.hstack([x, x[-1] * dx]) / sqrt_dx
    return xbins

# ####################################################################
# # Derivative that returns the same length
# ####################################################################
def diff(x, dt=1, t=None, n=1, axis=-1):
    '''Finite difference that returns the same length. Time domain method
    Can handle multi-dimensional matrices and non-uniform spacing
    
    diff(x)               -- first order difference, uniform spacing of 1.0
    diff(x, dt=0.01)      -- first order difference, uniform spacing of 0.01
    diff(x, dt=0.01, n=2) -- second order difference, uniform spacing of 0.01
    diff(x, t=t1, n=2)    -- second order difference, non-uniform spacing given by t1
    diff(x, ..., axis=-1) -- difference along the last dimension (default)
    diff(x, ..., axis=0)  -- difference along the first dimension
    '''
    for i in range(n):
        # if t is given, overrides dt
        if t is not None:
            x1 = diff_non_uniform(x, t, axis=axis)
            x = x1.copy()
        else:
            x1 = diff_uniform(x, dt, axis=axis)
            x = x1.copy()
    return x1

def diff_uniform_f(data_in, dt=1.0, n=1):
    '''Derivative of any order for uniform spaced series using frequency domain method'''
    X = np.fft.fft(data_in)
    f = np.fft.fftfreq(len(data_in), dt)
    X_diff = X * (1j * 2 * np.pi * f) ** n
    data_diff = np.fft.ifft(X_diff)
    return data_diff.real

def diff_uniform(data_in, dt=1.0, axis=-1):
    '''First-order derivative for uniform spaced series and returns the same length as the input series'''
    # move the axis to be differentiated to the last axis
    if axis != -1 or axis != len(data_in.shape)-1:
        data = np.moveaxis(data_in, axis, -1)
    else:
        data = data_in.copy()
    
    # compute finite difference using the central difference, except the two end nodes,
    # where forward and backward difference are used, respectively
    data_ext = np.concatenate([ (2*data[...,0] - data[...,1]).reshape(data.shape[:-1]+(1,)),
                                data,
                                (2*data[...,-1] - data[...,-2]).reshape(data.shape[:-1]+(1,)) ],
                              axis=-1)
    data_diff = (data_ext[..., 2:] - data_ext[..., :-2]) / (2*dt)
    
    if axis != -1 or axis != len(data_in.shape)-1:
        return np.moveaxis(data_diff, -1, axis)
    else:
        return data_diff

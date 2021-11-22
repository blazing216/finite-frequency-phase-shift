import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt

def mft(data, dt, f, gamma, emin=7):
    """Multiple filter technique
    """

    cc_mft = np.zeros([len(f), len(data)])
    for i in range(len(f)):
        cc_mft[i,:] = nbfilter(data, dt, f[i], gamma=gamma, emin=emin)
    return cc_mft

def nbfilter(data, dt, fc, gamma, emin=7):
    """narrow bandpass filter by convolving a Gaussian function
    Ref: Soomro et al., 2016, eq. 11"""
    wc = 2 * np.pi * fc
    alpha = gamma**2 * wc
    yfil, tfil = gauss_window(dt, wc, gamma, emin)
    nfil = len(tfil)
    datafiltered, ib = conv_offset(data, yfil, 0, -(nfil//2))
    datafiltered = datafiltered[np.arange(len(data)) - ib]
    datafiltered = detrend(datafiltered)
    absmax = np.abs(datafiltered).max()
    if absmax > 0:
        datafiltered /= absmax
    return datafiltered

def gauss_window(dt, wc, gamma, emin):
    alpha = gamma ** 2 * wc
    half_width = np.sqrt(emin * np.log(10) * 4 * alpha) / wc
    N = int(np.round(half_width / dt / 2)) * 4 - 1
    halfN = (N - 1) / 2
    t = dt * np.arange(-halfN, halfN+1)
    win = wc / (2 * np.sqrt(np.pi * alpha)) \
        * np.exp(-wc**2 * t**2/ (4 * alpha)) \
        * np.cos(wc * t)
    return win, t

def conv_offset(x1, x2, i1=0, i2=0):
    """assume x1 and x2 has the same time interval, but start at different time.
    x1 starts at i1 * dt and x2 starts at i2 * dt.
    The convolution of x1 and x2 are y, which starts at (i1+i2)*dt"""
    y = np.convolve(x1, x2, 'full')
    return y, i1+i2

def closest_maximum(x, ix):
    """Find local maximum closest to the position ix.

    Local maximum is defined by x[i-1] < x[i] < x[i+1].
    Hence, the first and last element of x are considered
    as local maximum when x[0] > x[1] or x[-1] > x[-2],
    when there can be a possible local maximum outside the
    range
    When x is a monotomic function, the "local maximum"
    is defined as the maximum of x[0] and x[-1].

    Args:
        x (numpy.ndarray): a seismogram.
        ix (int): an index.

    Returns:
        int: index of local maximum.
            1..nx-2, if x is not monotomic
            0 or nx-1, if x is monotomic
            ix, if x is a constant function
    """

    nx = len(x)

    ix = min(max(0, ix), nx-1)
#     assert ix >= 0 and ix <= nx-1, 'ix = %d is not in [0, %d (=len(x)-1)]' % (ix, nx-1)

    # search for local maximum in [ix, nx-1]
    ix1_max_found = False
    for ix1 in range(ix,nx):
        if ix1 == 0 and x[ix1] > x[ix1+1]:
            ix1_max_found = True
            break
        if ix1 == nx-1 and x[ix1] > x[ix1-1]:
            ix1_max_found = True
            break
        if x[ix1-1] < x[ix1] and x[ix1] > x[ix1+1]:
            ix1_max_found = True
            break

    # search for local maximum in [0, ix]
    ix2_max_found = False
    for ix2 in range(ix,-1,-1):
        if ix2 == 0 and x[ix2] > x[ix2+1]:
            ix2_max_found = True
            break
        if ix2 == nx-1 and x[ix2] > x[ix2-1]:
            ix2_max_found = True
            break
        if x[ix2-1] < x[ix2] and x[ix2] > x[ix2+1]:
            ix2_max_found = True
            break

    if ix1_max_found and ix2_max_found:
        ix_out = ix1 if abs(ix1-ix) <= abs(ix2-ix) else ix2
    elif ix1_max_found and not ix2_max_found:
        ix_out = ix1
    elif not ix1_max_found and ix2_max_found:
        ix_out = ix2
    else:
        ix_out = ix

    return ix_out

def track_ridge_mft(f, t, mft, f0, t0=None, refine=True, return_index=False):
    """get the phase velocity track maximum (ridge) along x-direction with correction based on amplitude
    f: sampling frequency in MFT
    t: lag time series
    mft: narrow-bandpass filtered cc. mft.shape == len(f), len(t)
    refine: if True, correct the time of local maximum by polynomial fitting
    return_index: if True, return an additional output that is actual time index of the tracking
    """

    nf, npts = len(f), len(t)
    dt = t[1]-t[0]

    if_start = np.argmin(np.abs(f-f0))
    if_start = min(max(0, if_start), nf-1)

    if t0 is None:
        it_start = np.argmax(mft[if_start,:])
    else:
        it_start = int(np.round((t0-t[0]) / dt))
        it_start = min(max(0, it_start), npts-1)

    trace_it = np.zeros(nf, dtype=int)
    trace_t = np.ones(nf) * np.nan
    trace_t_refined = np.ones(nf) * np.nan

    # track towards lower frequency
    it = it_start

    for ifreq in range(if_start, -1, -1):
        it = closest_maximum(mft[ifreq,:], it)

        fv = f[ifreq]

        trace_it[ifreq] = it
        if refine:
            if it == 0 or it == npts-1:
                trace_t_refined[ifreq] = t[it]
            else:
                trace_t_refined[ifreq] = precise_localmax(t[it-1:it+2],
                                                            mft[ifreq,it-1:it+2])
        else:
            trace_t[ifreq] = t[it]

    # track towards higher frequency
    it = it_start

    for ifreq in range(if_start, nf):
        fv = f[ifreq]
        it = closest_maximum(mft[ifreq,:], it)

        trace_it[ifreq] = it
        if refine:
            if it == 0 or it == npts-1:
                trace_t_refined[ifreq] = t[it]
            else:
                trace_t_refined[ifreq] = precise_localmax(t[it-1:it+2],
                                                            mft[ifreq,it-1:it+2])
        else:
            trace_t[ifreq] = t[it]

    if return_index:
        if refine:
            return trace_t_refined, trace_it
        else:
            return trace_t, trace_it
    else:
        if refine:
            return trace_t_refined
        else:
            return trace_t

def plot_mft(cc, t, f, ax=None, **kwargs):
    """Plot Multiple Filter Technique results. Used as a basic plot function.

    The results are plotted using pcolormesh. x, y-axis are
    t, f respectively. 

    Args: 
        cc (numpy.ndarray): Filtered seismograms.
            2D array with shape == (len(f), len(t))
        t (numpy.ndarray): Time.
        f (numpy.ndarray): (Center) frequencies of the filters.
        kwargs: keyword arguments passed to pcolormesh.

    Returns:
        return type of pcolormesh.

    Changelog:

    """
    if ax is None:
        ax = plt.gca()
    
    return ax.pcolormesh(bins(t), bins(f), cc, **kwargs)
    
def bins(x):
    if is_logspace(x):
        return logbins(x)
    else:
        return linbins(x)

def is_logspace(x, eps=1e-2):
    return is_linspace(np.log10(x))

def linbins(x):
    return np.hstack([2*x[0]-x[1], 0.5 * (x[1:]+x[:-1]), 2*x[-1]-x[-2]])

def logbins(x):
    return np.hstack([x[0]**2 / x[1], np.sqrt(x[1:]*x[:-1]),
                          x[-1]**2 / x[-2]])

def is_linspace(x, eps=1e-2):
    return np.abs(((x[1:] - x[:-1]) - (x[1] - x[0])) / (x[1] - x[0])).max() < eps

def precise_localmax(t, x):
    p = np.polyfit(t, x, 2)
    return -p[1]/(2*p[0])

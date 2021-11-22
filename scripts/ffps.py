import matplotlib.pyplot as plt
import numpy as np

from phase_shift_lib import demo_shallow
from phase_shift_lib import synthetic, phaseshift
import utils as yutils


def compute_finite_frequency_phase_shift(f, c, dist,
                                         t=None,
                                         f_mft=None, gamma=1):
    """Compute finite frequency phase shift, which requries
    (1) phase disperison curve (sampled evenly): f (Hz), c (km/s)
    (2) distance: dist. km

    Other Parameters:
        t:     Time samples to compute the seismogram
        f_mft: frequency samples used in multiple filter technique
        gamma: parameters used to control the width of gaussian filters.
          default 1.
    """
    if t is None:
        t = np.linspace(0, dist/0.5, 10001)
    if f_mft is None:
        f_mft = np.logspace(np.log10(f[0]), np.log10(f[-1]), 51)

    seis = synthetic.surface_wave_ncf(f, c, t, dist=dist,
                                      farfield=False,
                                      casual_branch_only=True)
    sw = {'t': t, 'seis': seis, 'f': f, 'c': c, 'dist': dist}

    c_mft = np.interp(f_mft, sw['f'], sw['c'])
    phase_tt = phaseshift.ptt_ag(sw, f_mft, c_mft, gamma=gamma,
                                 emin=7)
    phase_tt0 = phaseshift.ptt_theo(sw, f_mft, c_mft, iridge=0)

    dphase = (phase_tt - phase_tt0) * 2 * np.pi * f_mft

    return dphase


def test_shallow():
    f, t = demo_shallow.example_para_for_paper()
    f_mft = np.logspace(np.log10(f[0]), np.log10(f[-1]), 51)
    c = demo_shallow.cR(f)
    dists = np.arange(0.5, 4.001, 0.1)

    dphase_array = []
    for dist in dists:
        print(dist)
        dphase = compute_finite_frequency_phase_shift(f, c, dist,
                                                      t=t,
                                                      f_mft=f_mft,
                                                      gamma=1)
        dphase_array.append(dphase)
    dphase_array = np.array(dphase_array)

    # plot
    T = 1/f_mft[::-1]
    plt.pcolormesh(yutils.logbins(T),
                   yutils.linbins(dists),
                   dphase_array[:, ::-1],
                   cmap='jet',
                   vmin=-np.pi/30,
                   vmax=np.pi/30)
    plt.xlim(0.04, 1)
    plt.ylim(0.5, 4)
    plt.gca().set_xscale('log')
    plt.xlabel('Period (s)')
    plt.ylabel('Distance (km)')
    plt.savefig('ffps_shallow.pdf', bbox_inches='tight')


if __name__ == '__main__':
    test_shallow()

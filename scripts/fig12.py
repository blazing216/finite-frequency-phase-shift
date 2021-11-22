#!/usr/bin/env python

import pickle

from phase_shift_lib import demo_shallow

# plt.style.use('gmt_paper')

f, dist, ptts_theo, ptts_sf, ptts_ag = \
    demo_shallow.ptts_nearfield_finite_frequency_for_paper()

fig = demo_shallow.plot_nearfield_finite_frequency_for_paper(f, dist,
        ptts_theo, ptts_sf, ptts_ag)

fig.savefig('../figs/fig12.pdf')
fig.savefig('../figs/fig12.png', dpi=300)

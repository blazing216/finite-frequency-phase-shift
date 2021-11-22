#!/usr/bin/env python

from phase_shift_lib import demo_shallow
#plt.style.use('gmt_paper')

fig = demo_shallow.plot_simplest_phaseshift_for_paper(dist=3)

fig.savefig('../figs/fig2.pdf')
fig.savefig('../figs/fig2.png', dpi=300)

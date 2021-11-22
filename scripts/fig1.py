#!/usr/bin/python

from phase_shift_lib import demo_ak135

#plt.style.use('gmt_paper')

sw_ak135_1750 = demo_ak135.surface_wave_ballistic_for_paper(dist=1750)

fig = demo_ak135.plot_fig1(sw_ak135_1750)

fig.savefig('../figs/fig1.pdf')
fig.savefig('../figs/fig1.png', dpi=300)

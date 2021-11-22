#!/usr/bin/env python


from phase_shift_lib import demo_ak135

#plt.style.use('gmt_paper')

sw_ak135 = demo_ak135.surface_wave_ncf_for_paper(dist=250)

fig = demo_ak135.plot_illustration(sw_ak135)

fig.savefig('../figs/fig5.png', dpi=300)
fig.savefig('../figs/fig5.pdf')

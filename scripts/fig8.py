#!/usr/bin/env python

import pickle

from phase_shift_lib import demo_shallow

#plt.style.use('gmt_paper')

phase_tts_ag = pickle.load(open('../results/DB_gamma1.pkl', 'rb'))

f, dist, dcc = demo_shallow.ptts2dcc(phase_tts_ag)

fig = demo_shallow.plot_dcc_pcolormesh_for_paper(phase_tts_ag, cmap='jet')

fig.savefig('../figs/fig8.pdf')
fig.savefig('../figs/fig8.png')

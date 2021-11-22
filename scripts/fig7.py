#!/usr/bin/env python

import pickle

from phase_shift_lib import demo_shallow, phaseshift

#plt.style.use('gmt_paper')

sw = demo_shallow.surface_wave_ncf_for_paper(dist=4)

print(phaseshift.valid_freq_limits(sw['f'][0], sw['f'][-1],
                            gamma=1, emin=2.5))

demo_shallow.ptts_for_paper_to_file('../results/DB_gamma1.pkl',
    gamma=1,
    force=True)

phase_tts_ag = pickle.load(open('../results/DB_gamma1.pkl', 'rb'))

fig = demo_shallow.plot_ps_pcolormesh_for_paper(sw, phase_tts_ag, cmap='jet')

fig.savefig('../figs/fig7.pdf')
fig.savefig('../figs/fig7.png')


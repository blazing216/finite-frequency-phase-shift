# Installation

Python enviroment (This example uses anaconda. Ubuntu 20)
```
conda create -n ffps python=3.7
conda install numpy matplotlib scipy
conda install tqdm
```

`surf96py`
```
cd src/surf96py
make
```

Enviroment variables. In the folder `finite-frequency-phase-shift`, run
```
export PYTHONPATH=`pwd`/lib:$PYTHONPATH
```
In the folder `lib/phase-shift-lib`, run
```
export PSLPATH=`pwd`
```


# Run examples
In the folder `scripts`, run
```
python fig1.py
python fig2.py
python fig4.py
python fig5.py
python fig7.py
python fig8.py
python fig12.py
```
The generated figures are in `figs`. The figures may have a different appearance
from ones in the paper. To get the exact appearance, copy the file `mplstyle/gmt_paper.mplstyle`
to `~/.config/matplotlib/stylelib/` (the exact path might change on different OS), uncomment the
line
```
#plt.style.use('gmt_paper')
```
and run the scripts again.

Note: fig7.py must be run before fig8.py.


# Compute your own finite frequency phase shift
Modify based on the example `test_shallow` in the file `scripts/ffps.py`. You need to have
- A phase velocity dispersion curve (f, c)
- A function that measures the phase travel time, to replace `phaseshift.ptt_ag`

Parameters
- Use larger gamma for larger array

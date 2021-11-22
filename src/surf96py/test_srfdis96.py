import srfdis96 as sd
import numpy as np
import matplotlib.pyplot as plt

d = np.array([1,9,10,20,0])/10
a = [5,6.1,6.4,6.7,8.15]
b = [2.89,3.52,3.7,3.87,4.7]
rho = [2.5,2.73,2.82,2.9,3.36]
periods = np.linspace(0.05,1, 100)
t, cg = sd.compute_fundamental_Rayleigh_phase(periods,
    d,a,b,rho
    )
plt.plot(t, cg)
plt.show()
'''Empirical relationship between seismic velocities and
density

seismic velocities are in km/s and density in g/cm^3

Source:
(1) Brocher 2005, Relations between elastic wavespeeds and
density in the Earth's crust, BSSA 96(6), 2081-2092
'''

def vs2vp(vs):
    '''Convert shear wave velocity to P wave velocity
    using eq.(9) in Brocher (2005).

    Valid for vs in [0, 4.5] km/s.
    '''
    vp = 0.9409 + 2.0947 * vs \
        - 0.8206 * vs**2 + 0.2683 * vs**3 \
        - 0.0251 * vs**4
    return vp

def vp2rho(vp):
    ''' Convert P wavel velocity to density
    using eq.(1) in Brocher (2005).

    The equation is called the Nafe-Drake curve.
    Valide for vp in [1.5, 8.5] km/s
    '''
    rho = 1.6612 * vp - 0.4721 * vp**2 \
        + 0.0671 * vp**3 - 0.0043 * vp**4 \
        + 0.000106 * vp**5
    return rho

def vp2vs(vp):
    '''eq.(6) in Brocher (2005)
    '''
    vs = 0.7858 - 1.2344 * vp \
        + 0.7949 * vp**2 - 0.1238 * vp**3 \
        + 0.0064 * vp**4
    return vs


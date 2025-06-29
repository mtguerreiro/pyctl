import numpy as np

def zeta_wn(ts, os):

    zeta = -np.log(os / 100) / np.sqrt( np.pi**2 + np.log(os / 100)**2 )
    wn = 4 / ts / zeta

    return (zeta, wn)

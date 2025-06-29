import numpy as np
import scipy.signal

from . import design_utils
from . import energy as _energy

def casc_fblin(ts, pct_os, L, C, alpha=10):


    zeta_i, wn_i = design_utils.zeta_wn(ts / alpha, pct_os)
    k_ei = - L * wn_i**2
    ki  =   2 * L * zeta_i * wn_i

    zeta_v, wn_v = design_utils.zeta_wn(ts, pct_os)
    k_ev = - C * wn_v**2
    kv  =   2 * C * zeta_v * wn_v
    
    return (ki, k_ei, kv, k_ev)


def energy(ts, pct_os, alpha=5):
    
    return _energy.gains(ts, pct_os, alpha=alpha)

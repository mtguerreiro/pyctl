import numpy as np
import scipy.signal

from . import design_utils
from . import energy as _energy

def casc_fblin(ts, pct_os, Li, Co, alpha=10):

    k_ei = Li * 4 / (ts / alpha)

    zeta_v, wn_v = design_utils.zeta_wn(ts, pct_os)
    k_ev = - Co * wn_v**2
    kv  =   2 * Co * zeta_v * wn_v
    
    return (k_ei, kv, k_ev)


def energy(ts, pct_os, alpha=5):
    
    return _energy.gains(ts, pct_os, alpha=alpha)

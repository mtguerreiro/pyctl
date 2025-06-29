import numpy as np
import scipy.signal

from . import energy as _energy


def energy(ts, pct_os, alpha=5):
    
    return _energy.gains(ts, pct_os, alpha=alpha)

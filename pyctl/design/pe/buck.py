import numpy as np
import scipy.signal

from . import design_utils

def sfb(V_in, R, L, Co, ts, pct_os, alpha=5):

    A = np.array([
        [0,         -1 / L],
        [1 / Co,    -1 / R / Co]
        ])

    B = np.array([
        [V_in / L],
        [0]
        ])

    C = np.array([0, 1])

    # Augmented model (integrator)
    Aa = np.zeros((3,3))
    Aa[:2, :2] = A
    Aa[2, :2] = -C

    Ba = np.zeros((3,1))
    Ba[:2, 0] = B[:, 0]

    zeta, wn = design_utils.zeta_wn(ts, pct_os)

    p1 = -zeta * wn + 1j * wn * np.sqrt(1 - zeta**2)
    p2 = np.conj(p1)
    p3 = alpha * p1.real

    poles = [p1, p2, p3]

    K = scipy.signal.place_poles(Aa, Ba, poles).gain_matrix.reshape(-1)
    ki = K[0]
    kv = K[1]
    k_ev = K[2]
    
    return (ki, kv, k_ev)

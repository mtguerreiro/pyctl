import numpy as np
import scipy.signal
import pyctl

from . import design_utils

def gains(ts, pct_os, alpha=5):
   
    zeta, wn = design_utils.zeta_wn(ts, pct_os)

    A = np.array([[ 0.0, 1.0, 0.0],
                  [ 0.0, 0.0, 0.0],
                  [-1.0, 0.0, 0.0]])

    B = np.array([[0.0], [1.0], [0.0]])
    
    p1 = -zeta * wn + 1j * wn * np.sqrt(1 - zeta**2)
    p2 = np.conj(p1)
    p3 = alpha * p1.real

    poles = [p1, p2, p3]

    K = scipy.signal.place_poles(A, B, poles).gain_matrix.reshape(-1)

    ky = K[0]
    k_y_dot = K[1]
    k_ey = K[2]
    
    return (ky, k_y_dot, k_ey)


def mpc_gains(rw, l_pred, dt, alpha=1.0):
    
    Am = np.array([[0.0, 1.0],
                   [0.0, 0.0]])

    Bm = np.array([[0.0],
                   [alpha]])

    Cm = np.array([1.0, 0.0])

    Ad, Bd, Cd, _, _ = scipy.signal.cont2discrete((Am, Bm, Cm, 0), dt, method='zoh')

    sys = pyctl.mpc.System(Ad, Bd, Cd, l_pred=l_pred, rw=rw)

    return (sys.Kx, sys.Ky)

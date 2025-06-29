import numpy as np
import scipy.signal

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

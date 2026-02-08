import numpy as np
import scipy.signal

from . import design_utils


def lin_state_feedback(ts, pct_os, L1, L2, Cc, Co, v_in, vo, po, N, alpha=5):

    zeta, wn = design_utils.zeta_wn(ts, pct_os)

    p1 = -zeta * wn + 1j * np.sqrt(1 - zeta**2)
    
    poles = [p1, p1.conj(), alpha * p1, alpha * p1.conj()]

    x4 = vo
    x1 = po / v_in
    x2 = po / vo
    x3 = vo / N + v_in
    u = vo / (N * v_in + vo)
        
    A = np.array([
        [0,                         0,                  -1/L1 * (1 - u),    0],
        [0,                         0,                  N/L2 * u,           -1/L2],
        [(N**2+1)/(N**2*Cc)*(1-u),  -(N**2+1)/(N*Cc)*u, 0,                  0],
        [0,                         1/Co,               0,                  po/(Co*vo**2)]
        ])

    B = np.array([
        [1/L1 * x3],
        [N/L2 * x3],
        [-(N**2+1)/(Cc*N**2) * (x1 + N*x2)],
        [0]
        ])

    K = scipy.signal.place_poles(A, B, poles).gain_matrix[0]

    return (K, x1, x2, x3, x4, u)

import pyctl as ctl
import numpy as np

# --- Model ---
Am = np.array([[1, 1], [0, 1]])
Bm = np.array([[0.5], [1]])
Cm = np.array([1, 0])

# --- Augmented model ---
A, B, C = ctl.mpc.aug(Am, Bm, Cm)

import pyctl as ctl
import numpy as np

# --- Model ---
a = 0.8
b = 0.1
c = 1
r_w = 10
x_ki = np.array([[0.1], [0.2]])
r_ki = 1

n_p = 10
n_c = 4

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# --- Augmented model ---
A, B, C = ctl.mpc.aug(Am, Bm, Cm)

DU = ctl.mpc.opt(A, B, C, x_ki, r_ki, r_w, n_p, n_c)

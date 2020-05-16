import pyctl as ctl
import numpy as np
import matplotlib.pyplot as plt

# --- Model ---
a = 0.8
b = 0.1
c = 1
r_w = 0
x_ki = np.array([[0.1], [0.2]])
r_ki = 1

n_p = 10
n_c = 4

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# --- Augmented model ---
A, B, C = ctl.mpc.aug(Am, Bm, Cm)

# --- Optimization and sim ---
DU = ctl.mpc.opt(A, B, C, x_ki, r_ki, r_w, n_p, n_c)

(x, y) = ctl.mpc.sim(A, B, C, DU, x_ki, n_p + 1)

# --- Plots ---
plt.ion()
plt.plot(x, '--x')
#plt.plot(y)

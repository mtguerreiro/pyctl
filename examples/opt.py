import pyctl as ctl
import numpy as np
import matplotlib.pyplot as plt

# --- Model ---
a = 0.8
b = 0.1
c = 1
r_w = 10
x_ki = np.array([[0.1], [0.2]])
r_ki = 1

n_p = 10
n_c = 9

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# --- Augmented model ---
A, B, C = ctl.mpc.aug(Am, Bm, Cm)

# --- Optimization and sim ---
du = ctl.mpc.opt(A, B, C, x_ki, r_ki, r_w, n_p, n_c)

(dx, y) = ctl.mpc.predict_horizon(A, B, C, du, x_ki, n_p + 1)

# --- Plots ---
plt.ion()
plt.plot(dx, '--x')

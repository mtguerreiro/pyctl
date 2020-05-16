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
n_c = 4

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# --- System ---
sys = ctl.mpc.system(Am, Bm, Cm)

# --- Optimization and sim ---
u = sys.opt(x_ki, r_ki, r_w, n_p, n_c)
x, y = sys.sim(u, x_ki, n_p + 1)

# --- Plots ---
plt.ion()
plt.plot(x, '--x')


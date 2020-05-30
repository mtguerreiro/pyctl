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
#sys = ctl.mpc.system(Am, Bm, Cm)
sys = ctl.mpc.system(Am, Bm, Cm, n_p=n_p, n_c=n_c, r_w=r_w)

# --- Sim with receding horizon ---
#(u, x, y, dx) = sys.sim(x_ki, 0, r_ki, r_w, 10, n_p, n_c)

# --- Plots ---
#plt.ion()
#plt.plot(u, '--o')
#plt.plot(x, '--x')


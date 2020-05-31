import pyctl as ctl
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
a = 0.8
b = 0.1
c = 1
r_w = 0
x_i = np.array([[0.2, 0.1]])
r = 1

n_p = 10
n_c = 4

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# --- System ---
#sys = ctl.mpc.system(Am, Bm, Cm)
sys = ctl.mpc.System(Am, Bm, Cm, n_p=n_p, n_c=n_c, r_w=r_w)

# --- Sim with receding horizon ---
#(u, x, y, dx) = sys.sim(x_i, 0, r, 10)
data = sys.sim(x_i, 0, r, 10)

# --- Plots ---
#plt.ion()
#plt.plot(u, '--o')
#plt.plot(x, '--x')


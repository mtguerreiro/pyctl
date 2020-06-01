import pyctl as ctl
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
a = 0.8
b = 0.1
c = 1
x_i = np.array([[0.2, 0.1]])
u_i = 0

Am = np.array([a])
Bm = np.array([b])
Cm = np.array([c])

# Optimization parameters
r_w = 0.5
n_p = 10
n_c = 4

# Reference
r = 1

# Sim points
n = 100

# --- System ---
sys = ctl.mpc.System(Am, Bm, Cm, n_p=n_p, n_c=n_c, r_w=r_w)

# --- Sim with receding horizon ---
data = sys.dmpc(x_i, u_i, r, n)

# --- Plots ---
#plt.ion()
plt.plot(data['u'], '--o')
plt.plot(data['y'], '--x')

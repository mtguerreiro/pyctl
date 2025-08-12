import pyctl as ctl
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
Am = np.array([[0.0, 1.0],
               [0.0, 0.0]])

Bm = np.array([[0.0],
               [1.0]])

Cm = np.array([1.0, 0.0])

x0 = [-2.0, 0.0]
u0 = [0.0]

u_lim = [[-10], [10]] # or None
x_lim = [[None, -4], [None, 4]] # or None

# Discretization
dt = 0.1

# Optimization parameters
rw = 0.025
l_pred = 10
l_ctl = 4

# Reference
r = 5

# Sim points
n = 50

# --- System ---
Ad, Bd, Cd, _, _ = scipy.signal.cont2discrete((Am, Bm, Cm, 0), dt, method='zoh')

sys = ctl.mpc.System(Ad, Bd, Cd, l_pred=l_pred, l_ctl=l_ctl, rw=rw, u_lim=u_lim, x_lim=x_lim)

# --- Sim with receding horizon ---
data = sys.sim(x0, u0, r, n)

# --- Plots ---
ax = plt.subplot(3,1,1)
plt.plot(data['u'], label='u')
plt.legend()
plt.xlim([0, n - 1])
plt.gca().tick_params(labelbottom=False)
plt.grid()

plt.subplot(3,1,2, sharex=ax)
plt.plot(data['xm'][:, 1], label='$x_2$')
plt.legend()
plt.gca().tick_params(labelbottom=False)
plt.grid()

plt.subplot(3,1,3, sharex=ax)
plt.plot(data['y'], label='$y$')
plt.legend()
plt.grid()

plt.tight_layout()

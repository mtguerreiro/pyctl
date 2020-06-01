import pyctl as ctl
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# --- Model ---
R = 1
C = 1
L = 1
V = 1

x_i = np.array([0, 0])

A = np.array([[0, 1 / C], [-1 / L, -R / L]])
B = np.array([[0], [1 / L]])
C = np.array([[1, 0]])

# --- Sim ---
sys = ctl.discrete.System(A, B, C, dt=0.1)
(x, y) = sys.p_sim(x_i, V, 2, 500)

#A, B, C = ctl.mpc.aug(Am, Bm, Cm)

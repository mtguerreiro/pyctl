import numpy as np
import pyctl as ctl


def aug(Am, Bm, Cm):
    r"""Determines the augmented model. For now, only one control signal is
    supported.

    Parameters
    ----------
    Am : np.array
        An (n, n) numpy matrix.

    Bm : np.array
        An (n, 1) numpy matrix.

    Cm : np.array
        A (1, n) numpy matrix.

    Returns
    -------
    (A, B, C) : tuple
        A tuple containing the augmented matrices.
    
    """
    n = Am.shape[0]
    zeros_n = np.zeros((n, 1))

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Am
    A[-1, :n] = Cm @ Am
    A[-1, -1] = 1

    B = np.zeros((n + 1, 1))
    B[:n] = Bm
    B[-1] = Cm @ Bm

    C = np.zeros((1, n + 1))
    C[0, -1] = 1

    return (A, B, C)


def opt(A, B, C, x_ki, r_ki, r_w, n_p, n_c):
    r"""Provides the control vector miniming the expression

    .. :math:

        J = (R_s - Y)^T(R_s - Y) + \Delta U^T R \Delta U.

    """
    R_s = r_ki * np.ones((n_p, 1))
    R = r_w * np.eye(n_c)
    
    F = np.zeros((n_p, (C @ A).shape[1]))
    F[0, :] = C @ A
    for i in range(1, n_p):
        F[i, :] = F[i - 1, :] @ A
    
    Phi = np.zeros((n_p, n_c))
    Phi[0, 0] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[i, 0] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[i, j + 1] = Phi[i - 1, j]

    Phi_t = Phi.T

    DU = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t @ (R_s - F @ x_ki)
    
    return DU


def sim(A, B, C, u, x_ki, n_p):
    
    n_c = u.shape[0]

    x = np.zeros((n_p, x_ki.shape[0]))
    y = np.zeros((n_p, C.shape[0]))

    x[0, :] = x_ki.reshape(-1)
    y[0, :] = C @ x[0, :]
    for i in range(1, n_c):
        x[i, :] = A @ x[i - 1, :] + B @ u[i - 1]
        y[i, :] = C @ x[i, :]
        
    for i in range(n_c, n_p):
        x[i, :] = A @ x[i - 1, :]
        y[i, :] = C @ x[i, :]

    return (x, y)


class system:

    def __init__(self, Am, Bm, Cm):
        self.A, self.B, self.C = ctl.mpc.aug(Am, Bm, Cm)

    
    def opt(self, x_ki, r_ki, r_w, n_p, n_c):

        DU = ctl.mpc.opt(self.A, self.B, self.C, x_ki, r_ki, r_w, n_p, n_c) 

        return DU

    
    def sim(self, u, x_ki, n_p):

        x, y = ctl.mpc.sim(self.A, self.B, self.C, u, x_ki, n_p)

        return (x, y)
    

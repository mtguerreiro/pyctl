import numpy as np
import pyctl
import qpsolvers as qps
import sys

def opt_unc_gains_freq(A, B, C, n_pred, n_ctl, rw, Qq, Ql):
    r"""Computes the optimum gains `Ky` and `K_mpc` for the unconstrained
    closed-loop system.

    Parameters
    ----------
    A : :class:`np.array`
        :math:`A` matrix. An (n, n) numpy matrix.

    B : :class:`np.array`
        :math:`B` matrix. An (n + q, m) numpy matrix.

    C : :class:`np.array`
        :math:`C_m` matrix. A (q, n) numpy matrix.

    n_pred : :class:`int`
        Length of prediction horizon.

    n_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `n_ctl` is less than or equal to `n_pred`. For
        `n_ctl` less than `n_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `n_ctl = n_pred` is
        assumed.

    n_ctn : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints are enforced.
        Note that `n_ctn` is less than or equal to `n_ctl`. If set to `None`,
        `n_ctn = n_ctl` is assumed.
    
    rw : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weighting factor for the control inputs.If set to `None`, `rw` is set
        to zero.

    Returns
    -------
    (Ky, K_mpc) : :class:`tuple`
        A tuple, containing two elements. The first element is the vector
        `Ky` and the second element is the vector `K_mpc`.

    """    
    # Number of states
    n = A.shape[0]

    # Number of inputs
    if B.ndim == 1:
        m = 1
    else:
        m = B.shape[1]

    # Number of outputs
    if C.ndim == 1:
        q = 1
    else:
        q = C.shape[0]

    Rs_bar = pyctl.mpc.reference_matrix(q, n_pred)

    R = pyctl.mpc.control_weighting_matrix(rw, n_ctl)

    F, Phi = pyctl.mpc.opt_matrices(A, B, C, n_pred, n_ctl)
    Phi_t = Phi.T
    
    K = np.linalg.inv(Phi_t @ Phi + R + Qq) @ Phi_t
    K_mpc = K @ F
    Ky = K @ Rs_bar

    Kq = 0    
    #if Ql is not None:
    #    Kq = -np.linalg.inv(Phi_t @ Phi + R + Qq) @ Ql
    #    Kq = Kq[:m, :]
    #else:
    #    Kq = None
    
    return (Ky[:m], K_mpc[:m, :], Kq)


class FreqWeighting:

    def __init__(self, q=None, lp=None, n_ctl=0, nu=0):
        
        self.q = q
        self.lp = lp
        self.n_ctl = n_ctl
        self.nu = nu

        if q is None:
            self.lp = 0
            self.Qq = 0
            self.Ql = 0
            self.du_1 = 0
        else:
            self.du_1 = np.zeros((lp * nu, 1))
            self.gen_static_matrices()

        self.Kq = None


    def set_Kq(self, Kq):

        pass


    def du_unc(self, ui):

        if self.q is None:
            return 0

        n = self.Qq.shape[0]
        
        U_1 = np.ones((n * self.nu, 1)) * ui
        #print(U_1)
        lin_cost = np.tri(n).T @ self.Qw.T @ U_1
        #print(lin_cost)
        lin_cost = lin_cost[0,0]
        #print(lin_cost)
        #print('lin cost', lin_cost.shape)
        #print(lin_cost)
            
        return lin_cost
    

    def quad_cost_static(self):

        return self.Qq


    def lin_cost_dyn(self, ui):

        if self.q is None:
            return 0

        n = self.Qq.shape[0]
        
        U_1 = np.ones((n * self.nu, 1)) * ui
        #print(U_1)
        lin_cost = np.tri(n).T @ self.Qw.T @ U_1
        #print('lin cost', lin_cost.shape)
        #print(lin_cost)
            
        return lin_cost


    def lin_cost_matrix(self):

        n = self.Qq.shape[0]
        
        U_1 = np.ones((n * self.nu, 1))
        lin_cost = np.tri(n).T @ self.Qq.T @ U_1
        
        return lin_cost

    
    def gen_static_matrices(self):

        if self.q is None:
            return

        lp = self.nu * self.lp
        lf = self.nu * self.n_ctl
        
        N = lp + lf

        W = np.zeros((N, N), dtype=complex)

        for ni in range(N):
            W[ni, :] = np.exp(-1j * 2 * np.pi * ni * np.arange(N) / N)

        Q = np.diag(self.q)
        Qw = (W.conj() @ Q @ W).real / N**2

        self.Qw = (W.conj() @ Q @ W).real / N**2
        self.Qq = np.tri(N).T @ Qw @ np.tri(N)


##def opt_unc_gains_freq(A, B, C, n_pred, n_ctl, rw, Qq, Ql):
##    r"""Computes the optimum gains `Ky` and `K_mpc` for the unconstrained
##    closed-loop system.
##
##    Parameters
##    ----------
##    A : :class:`np.array`
##        :math:`A` matrix. An (n, n) numpy matrix.
##
##    B : :class:`np.array`
##        :math:`B` matrix. An (n + q, m) numpy matrix.
##
##    C : :class:`np.array`
##        :math:`C_m` matrix. A (q, n) numpy matrix.
##
##    n_pred : :class:`int`
##        Length of prediction horizon.
##
##    n_ctl : :class:`NoneType`, :class:`int`
##        Length of the prediction horizon where the control input increments
##        can be set. Note that `n_ctl` is less than or equal to `n_pred`. For
##        `n_ctl` less than `n_pred`, the increments are set zero to for the
##        remaining prediction steps. If set to `None`, `n_ctl = n_pred` is
##        assumed.
##
##    n_ctn : :class:`NoneType`, :class:`int`
##        Lenght of the prediction horizon where the constraints are enforced.
##        Note that `n_ctn` is less than or equal to `n_ctl`. If set to `None`,
##        `n_ctn = n_ctl` is assumed.
##    
##    rw : :class:`NoneType`, :class:`int`, :class:`np.array`
##        Weighting factor for the control inputs.If set to `None`, `rw` is set
##        to zero.
##
##    Returns
##    -------
##    (Ky, K_mpc) : :class:`tuple`
##        A tuple, containing two elements. The first element is the vector
##        `Ky` and the second element is the vector `K_mpc`.
##
##    """    
##    # Number of states
##    n = A.shape[0]
##
##    # Number of inputs
##    if B.ndim == 1:
##        m = 1
##    else:
##        m = B.shape[1]
##
##    # Number of outputs
##    if C.ndim == 1:
##        q = 1
##    else:
##        q = C.shape[0]
##
##    Rs_bar = pyctl.mpc.reference_matrix(q, n_pred)
##
##    R = pyctl.mpc.control_weighting_matrix(rw, n_ctl)
##
##    F, Phi = pyctl.mpc.opt_matrices(A, B, C, n_pred, n_ctl)
##    Phi_t = Phi.T
##    
##    K = np.linalg.inv(Phi_t @ Phi + R + Qq) @ Phi_t
##    K_mpc = K @ F
##    Ky = K @ Rs_bar
##
##    if Ql is not None:
##        Kq = -np.linalg.inv(Phi_t @ Phi + R + Qq) @ Ql
##        Kq = Kq[:m, :]
##    else:
##        Kq = None
##    
##    return (Ky[:m], K_mpc[:m, :], Kq)
##
##
##class FreqWeighting:
##
##    def __init__(self, q=None, lp=None, n_ctl=0, nu=0):
##        
##        self.q = q
##        self.lp = lp
##        self.n_ctl = n_ctl
##        self.nu = nu
##
##        if q is None:
##            self.lp = 0
##            self.Qq = 0
##            self.Ql = 0
##            self.du_1 = 0
##        else:
##            self.du_1 = np.zeros((lp * nu, 1))
##            self.gen_static_matrices()
##
##        self.Kq = None
##
##
##    def set_Kq(self, Kq):
##
##        self.Kq = Kq
##
##
##    def du_unc(self, du_1):
##        
##        self.update_du_past(du_1)
##        
##        if self.Kq is not None:
##            du = self.Kq @ self.du_1
##        else:
##            du = 0
##
##        return du
##
##    
##    def update_du_past(self, du_1):
##
##        if self.lp == 0:
##            return
##        
##        lp = self.lp
##        nu = self.nu
##        
##        self.du_1[:lp*nu - 1, 0] = self.du_1[1:lp*nu, 0]
##        self.du_1[lp*nu - 1, 0] = du_1
##
##
##    def du_past_vector(self):
##
##        if self.lp == 0:
##            past_vector = 0
##        else:
##            past_vector = self.du_1
##            
##        return past_vector
##
##
##    def quad_cost_static(self):
##
##        return self.Qq
##
##
##    def lin_cost_dyn(self, du_1):
##
##        if self.q is None:
##            return 0
##
##        if self.lp == 0:
##            return 0
##
##        self.update_du_past(du_1)
##
##        lin_cost = self.Ql @ self.du_1
##            
##        return lin_cost
##
##
##    def lin_cost_matrix(self):
##        
##        return self.Ql
##
##    
##    def gen_static_matrices(self):
##
##        if self.q is None:
##            return
##
##        lp = self.nu * self.lp
##        lf = self.nu * self.n_ctl
##        
##        N = lp + lf
##
##        W = np.zeros((N, N), dtype=complex)
##
##        for ni in range(N):
##            W[ni, :] = np.exp(-1j * 2 * np.pi * ni * np.arange(N) / N)
##
##        Q = np.diag(self.q)
##        Qw = (W.conj() @ Q @ W).real / N**2
##
##        self.Ql = Qw[lp:, :lp]
##        self.Qq = Qw[lp:, lp:]

import numpy as np
import qpsolvers

import ctypes


class QP:

    def __init__(self, Ej, M, Hj, F, Phi):

        self.Ej = Ej
        self.Ej_inv = np.linalg.inv(Ej)
        self.M = M
        
        self.Hj = Hj
        self.F = F
        self.Phi = Phi
        
        self.solvers = qpsolvers.available_solvers
        self.solvers.extend(['hild', 'cdmpc'])


    def set_dll(self, dll):

        self.cdmpc = CDMPC(dll)
            

    def solve(self, xm, dx, xa, ui, Xp, r, Fj, y, solver='hild'):
        r"""Solves the QP problem given by:

        .. :math:

            J = \Delta U^T E_J \Delta U^T +  \Delta U^T F_j,

        subject to:

        .. :math:

            M \Delta U \leq y.
            
        """        
        if solver not in self.solvers:
            raise ValueError(f'`solver` should be one of {self.solvers}.')

        Ej = self.Ej; Ej_inv = self.Ej_inv
        M = self.M

        if solver == 'hild':
            F = self.F; Phi = self.Phi
            
            Hj = self.Hj
            Kj = y + M @ Ej_inv @ Fj

            lm, n_iters = hild(Hj, Kj, n_iter=250)
            lm = lm.reshape(-1, 1)
            du_opt = -Ej_inv @ (Fj + M.T @ lm)
            du_opt = du_opt.reshape(-1)

        elif solver == 'cdmpc':
            du_opt, n_iters = self.cdmpc.solve(xm, dx, xa, ui, Xp, r)
            
        else:
            du_opt = qpsolvers.solve_qp(Ej, Fj.reshape(-1), M, y.reshape(-1), solver=solver)
            n_iters = 0

        return (du_opt, n_iters)
            

class CDMPC:

    def __init__(self, dll):

        c_float_p = ctypes.POINTER(ctypes.c_float)
        
        self.dll = ctypes.CDLL(dll)
        
        self.dll.cdmpc_py_step.argtypes = (
                c_float_p, c_float_p, c_float_p, c_float_p, c_float_p
                )

    def solve(self, xm, dx, xa, ui, Xp, r):

        c_float_p = ctypes.POINTER(ctypes.c_float)
        
        xm_1 = xm - dx

        xm = xm.astype(np.float32)
        xm_1 = xm_1.astype(np.float32)
        dx = dx.astype(np.float32)
        ui = ui.astype(np.float32)
        Xp = Xp.astype(np.float32)
        r = r.astype(np.float32)        

        du = np.zeros(ui.shape, dtype=np.float32)

        n_iters = self.dll.cdmpc_py_step(
            xm.ctypes.data_as(c_float_p), xm_1.ctypes.data_as(c_float_p),
            r.ctypes.data_as(c_float_p), ui.ctypes.data_as(c_float_p),
            Xp.ctypes.data_as(c_float_p),
            du.ctypes.data_as(c_float_p))
        
        return du, n_iters


def hild(H, K, n_iter=100, lm=None):

    h_d = -1 / H.diagonal()

    if lm is None:
        lm = np.zeros(H.shape[0])
    lm_p = np.zeros(H.shape[0])
    w = np.zeros(H.shape[0])
    
    k = 0

    while k < n_iter:
        lm_p[:] = lm[:]
        for i in range(0, w.shape[0]):
            lm[i] = 0
            w[i] = h_d[i] * (K[i, 0] + H[i,:] @ lm[:])
            if w[i] > 0: lm[i] = w[i]
        k = k + 1
        if np.allclose(lm_p, lm) == True:
            break

    return (lm, k)

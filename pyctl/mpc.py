import numpy as np
import pyctl
import qpsolvers as qps
import sys

import matplotlib.pyplot as plt

def aug(Am, Bm, Cm):
    r"""Determines the augmented model.

    Parameters
    ----------
    Am : np.array
        An (n, n) numpy matrix.

    Bm : np.array
        An (n, m) numpy matrix.

    Cm : np.array
        A (q, n) numpy matrix.

    Returns
    -------
    (A, B, C) : tuple
        A tuple containing the augmented matrices.
    
    """
    # Number of states
    n = Am.shape[0]

    # Number of inputs
    if Bm.ndim == 1:
        m = 1
    else:
        m = Bm.shape[1]

    # Number of outputs
    if Cm.ndim == 1:
        q = 1
    else:
        q = Cm.shape[0]

    A = np.zeros((n + q, n + q))
    A[:n, :n] = Am
    A[n:, :n] = Cm @ Am
    A[n:, n:] = np.eye(q)

    B = np.zeros((n + q, m))
    B[:n, :] = Bm
    B[n:, :] = Cm @ Bm

    C = np.zeros((q, n + q))
    C[:, n:] = np.eye(q)

    return (A, B, C)


def opt_matrices(A, B, C, n_p, n_c):
    r"""Computes the :math:`F` and :math:`Phi` matrices.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, m) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (q, n) numpy matrix.

    n_p : :class:`int`
        Length of prediction horizon.

    n_c : :class:`int`
        Length of the control window.

    Returns
    -------
    (F, Phi) : :class:`tuple`
        A tuple, where the first item corresponds to the `F` matrix and the
        second item corresponds to the `Phi` matrix.
    
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
    
    F = np.zeros((n_p * q, A.shape[1]))
    F[:q, :] = C @ A
    for i in range(1, n_p):
        F[q * i : q * (i + 1), :] = F[q * (i - 1) : q * i, :] @ A

    Phi = np.zeros((n_p * q, n_c * m))
    Phi[:q, :m] = C @ B
    for i in range(1, n_p):
        A_p = np.linalg.matrix_power(A, i)
        Phi[ (q * i) : ( q * (i + 1) ), :m] = C @ A_p @ B
        for j in range(n_c - 1):
            Phi[ (q * i) : ( q * (i + 1) ), m * (j + 1) : m * (j + 2)] = Phi[ ( q * (i - 1) ) : (q * i), m * j : m * (j + 1)]

    return (F, Phi)


def control_weighting_matrix(r_w, n_c):
    r"""Computes the :math:`\bar{R}` matrix.

    Parameters
    ----------
    r_w : :class:`int`, :class:`list`, :class:`np.array`
        The weighting coefficients, as a 1-d numpy array, an integer or a
        list.

    n_c : :class:`int`
        Length of the control window.

    Returns
    -------
    R_bar : :class:`np.array`
        An (n_c * m, n_c * m) matrix, where `m` is the number of coefficients
        (system input signals).
    
    """
    if type(r_w) is int or type(r_w) is float:
        r_w = np.array([r_w])
    elif type(r_w) is list:
        r_w = np.array(r_w)

    m = r_w.shape[0]
    
    R_bar = np.zeros((n_c * m, n_c * m))
    for i in range(n_c):
        R_bar[m * i : m * (i + 1), m * i : m * (i + 1)] = np.diag(r_w)

    return R_bar


def reference_matrix(q, n_p):
    r"""Computes the :math:`\bar{R_s}` matrix.

    Parameters
    ----------
    q : :class:`int`
        Number of references (system outputs).
        
    n_p : :class:`int`
        Length of prediction horizon.

    Returns
    -------
    R_s_bar : :class:`np.array`
        An (q, n_p * q) matrix, where `q` is the number of references
        (system outputs).
    
    """
    R_s_bar = np.tile(np.eye(q), (n_p, 1))

    return R_s_bar


def opt_unc_gains(A, B, C, n_pred, n_ctl, rw):
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

    Rs_bar = reference_matrix(q, n_pred)

    R = control_weighting_matrix(rw, n_ctl)

    F, Phi = opt_matrices(A, B, C, n_pred, n_ctl)
    Phi_t = Phi.T
    
    K = np.linalg.inv(Phi_t @ Phi + R) @ Phi_t
    K_mpc = K @ F
    Ky = K @ Rs_bar

    return (Ky[:m], K_mpc[:m, :])


class System:
    """A class to create a discrete-time system for model predictive control
    simulations.

    Parameters
    ----------
    Am : :class:`np.array`
        Model matrix :math:`A_m`. An (n, n) numpy matrix.

    Bm : :class:`np.array`
        Model matrix :math:`B_m`. An (n, m) numpy matrix.

    Cm : :class:`np.array`
        Model matrix :math:`C_m`. An (q, n) numpy matrix.

    n_pred : :class:`int`
        Length of prediction horizon.

    n_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `n_ctl` is less than or equal to `n_pred`. For
        `n_ctl` less than `n_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `n_ctl = n_pred` is
        assumed.

    n_cnt : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints are enforced.
        Note that `n_cnt` is less than or equal to `n_ctl`. If set to `None`,
        `n_cnt = n_ctl` is assumed.
    
    rw : :class:`NoneType`, :class:`int`, :class:`np.array`
        Weighting factor for the control inputs. If set to `None`, `rw` is set
        to zero.

    q : :class:`NoneType`,  :class:`int`, :class:`np.array`
        Weighting factor for frequency components of the control inputs.

    x_lim : :class:`NoneType`, :class:`np.array`, :class:`list`
        Lower and upper bounds of the states. If set to `None`, no constraints
        are imposed on the states. 
        
    u_lim : :class:`NoneType`, :class:`np.array`, :class:`list`
        Lower and upper bounds of the control signals. If set to`None`, no
        constraints and set on the input signals.

    """
    def __init__(self, Am, Bm, Cm, n_pred,
                 n_ctl=None, n_cnt=None,
                 rw=None, q=None, lp=None,
                 x_lim=None, u_lim=None):

        # System model and augmented model
        self.Am = Am; self.Bm = Bm; self.Cm = Cm
        
        self.A, self.B, self.C = aug(Am, Bm, Cm)

        # Prediction horizon
        self.n_pred = n_pred

        # Control horizon
        if n_ctl is None:
            self.n_ctl = n_pred
        else:
            self.n_ctl = n_ctl

        # Constraints horizon
        if n_cnt is None:
            self.n_cnt = self.n_ctl
        else:
            self.n_cnt = n_cnt

        # Weighting factor
        if Bm.ndim == 1:
            nu = 1
        else:
            nu = Bm.shape[1]
            
        if rw is None:
            rw = np.zeros(nu)
        
        elif type(rw) is int or type(rw) is float:
            rw = rw * np.ones(nu)

        self.rw = rw
       
        self.fw = pyctl.fw.FreqWeighting(q, lp, n_ctl, nu)
        
        # Bounds
        if type(x_lim) is list:
            x_lim = np.array(x_lim)
        if type(u_lim) is list:
            u_lim = np.array(u_lim)
        
        self.x_lim = x_lim
        self.u_lim = u_lim

        # Gains for unconstrained problem
        n_xm = Am.shape[0]
        if q is None:
            Ky, K_mpc = opt_unc_gains(self.A, self.B, self.C, \
                                      self.n_pred, self.n_ctl, self.rw)
            Kq = None
        else:
            Ky, K_mpc, Kq = pyctl.fw.opt_unc_gains_freq(
                self.A, self.B, self.C,
                self.n_pred, self.n_ctl, self.rw,
                self.fw.quad_cost_static(), self.fw.lin_cost_matrix()
                )

        self.Ky = Ky
        self.Kx = K_mpc[:, :n_xm]
        self.Kq = Kq

        self.fw.set_Kq( Kq )

        if x_lim is None:
            self.x_lim_idx = None

        if u_lim is None:
            self.u_lim_idx = None

        if (x_lim is not None) or (u_lim is not None):
            # Initializes static qp matrices
            self.gen_static_qp_matrices()

            # Creates Hildreth's static matrix        
            self.Hj = self.M @ self.Ej_inv @ self.M.T

        # Creates index vector of outputs
        y_idx = []
        if Cm.ndim != 1:
            Cm = np.sum(Cm, axis=0)
        for i, yi in enumerate(Cm):
            if np.abs(yi) > 0.5: y_idx.append(i)

        self.y_idx = np.array(y_idx)
        
        self.c_gen = c_gen()

        self._du_0 = None

        
    def gen_static_qp_matrices(self):
        """Sets constant matrices, to be used later by the optimization.

        """
        A = self.A; B = self.B; C = self.C
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        n_pred = self.n_pred; n_ctl = self.n_ctl; n_cnt = self.n_cnt
        rw = self.rw

        x_lim = self.x_lim
        u_lim = self.u_lim
                
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
        
        R_bar = control_weighting_matrix(rw, n_ctl)
        Rs_bar = reference_matrix(q, n_pred)

        self.R_bar = R_bar
        self.Rs_bar = Rs_bar

        # Creates left-hand side inequality constraint, starting first with
        # control inequality constraints
        M = None
        if u_lim is not None:
            M_aux = np.tril( np.tile( np.eye(m), (n_cnt, n_ctl) ) )
            Mu = np.concatenate((-M_aux, M_aux))
            M = Mu

            # TODO: possibility to constraint only certain inputs
            # For now, we assume that all controls are constrained
            u_lim_idx = np.arange(len(u_lim[0]))
            self.u_lim_idx = u_lim_idx

        # Now, the state inequality constraints
        if x_lim is not None:
            n_state_ineq = 0
            x_lim_idx = []
            x_lim_new = [[], []]
            Cx = None
            
            for i, xi in enumerate(x_lim[0]):
                if xi is not None:
                    x_lim_new[0].append(x_lim[0][i])
                    x_lim_new[1].append(x_lim[1][i])
                    x_lim_idx.append(i)
                    
                    n_state_ineq = n_state_ineq + 1
                    
                    cx = np.zeros((1, Am.shape[0]))
                    cx[0, i] = 1
                    
                    if Cx is None:
                        Cx = cx
                    else:
                        Cx = np.concatenate((Cx, cx))
                        
            Fx, Phi_x = opt_matrices(Am, Bm, Cx, n_cnt, n_ctl)
            self.Fx = Fx; self.Phi_x = Phi_x
            self.Cx = Cx
            
            self.x_lim = np.array(x_lim_new)
            self.x_lim_idx = np.array(x_lim_idx)

            M_aux = np.tril( np.tile( np.eye(self.x_lim.shape[1]), (n_cnt, n_cnt) ) )
            self.Mx_aux = M_aux
            Mx = np.concatenate((-M_aux @ Phi_x, M_aux @ Phi_x))

            if M is None:
                M = Mx
            else:
                M = np.concatenate((M, Mx))

        self.M = M

        # QP matrices
        F, Phi = opt_matrices(A, B, C, n_pred, n_ctl)
        self.F = F; self.Phi = Phi
        
        Ej = Phi.T @ Phi + R_bar
        Ej = Ej + self.fw.quad_cost_static()
        Ej_inv = np.linalg.inv(Ej)
        self.Ej = Ej; self.Ej_inv = Ej_inv


    def gen_dyn_qp_matrices(self, xm, dx, xa, ui, r, du_1):
        """Sets dynamic matrices, to be used later by the optimization.

        """
        n_cnt = self.n_cnt
        F = self.F; Phi = self.Phi
        Rs_bar = self.Rs_bar
       
        u_lim = self.u_lim
        x_lim = self.x_lim

        Fj = -Phi.T @ (Rs_bar @ r.reshape(-1, 1) - F @ xa.reshape(-1, 1))
        #print(Fj)
        #Fj = Fj + self.fw.lin_cost_dyn(du_1)
        Fj = Fj + self.fw.lin_cost_dyn(ui)
        #print(Fj)
        
        # Creates the right-hand side inequality vector, starting first with
        # the control inequality constraints
        y = None
        
        if u_lim is not None:
            u_min = np.tile(-u_lim[0] + ui, n_cnt).reshape(-1, 1)
            u_max = np.tile( u_lim[1] - ui, n_cnt).reshape(-1, 1)

            y = np.concatenate((u_min, u_max))

        # Now, the state inequality constraints
        if x_lim is not None:
            Mx_aux = self.Mx_aux
            Cx = self.Cx
            Fx = self.Fx
            M_Fx = Mx_aux @ Fx
            x_min = np.tile(-x_lim[0] + Cx @ xm, n_cnt).reshape(-1, 1) + M_Fx @ dx.reshape(-1, 1)
            x_max = np.tile( x_lim[1] - Cx @ xm, n_cnt).reshape(-1, 1) - M_Fx @ dx.reshape(-1, 1)

            if y is None:
                y = np.concatenate((x_min, x_max))
            else:
                y = np.concatenate((y, x_min, x_max))

        return (Fj, y)


    def opt_unc(self, dx, y, r, ui, du_1):

        du = - self.Ky @ (y - r) - self.Kx @ dx
        #du_freq = self.fw.du_unc(du_1)
        du_freq = self.fw.du_unc(ui)
        du = du + du_freq

        if self._du_0 is None:
            self._du_0 = du
            self._u_0 = np.tri(du.shape[0]) @ du
                
        return du
    

    def opt(self, xm, dx, xa, ui, du_1, r, solver='hild'):

        nu = ui.shape[0]
        
        Fj, y = self.gen_dyn_qp_matrices(xm, dx, xa, ui, r, du_1)

        du, n_iters = self.qp(Fj, y, solver=solver)

        if self._du_0 is None:
            self._du_0 = du
            self._u_0 = np.tri(du.shape[0]) @ du
            
        return (du[:nu], n_iters)

    
    def qp(self, Fj, y, solver='hild'):
        r"""Solves the QP problem given by:

        .. :math:

            J = \Delta U^T E_J \Delta U^T +  \Delta U^T F_j,

        subject to:

        .. :math:

            M \Delta U \leq y.
            
        """
        Ej = self.Ej; Ej_inv = self.Ej_inv
        M = self.M

        if solver == 'hild':
            F = self.F; Phi = self.Phi
            
            Hj = self.Hj
            Kj = y + M @ Ej_inv @ Fj

            lm, n_iters = pyctl.qp.hild(Hj, Kj, n_iter=250, ret_n_iter=True)
            lm = lm.reshape(-1, 1)
            du_opt = -Ej_inv @ (Fj + M.T @ lm)
            du_opt = du_opt.reshape(-1)

        elif solver == 'cvx':
            du_opt = qps.cvxopt_solve_qp(Ej, Fj.reshape(-1), M, y.reshape(-1))
            n_iters = 0

        elif solver == 'quadprog':
            du_opt = qps.solve_qp(Ej, Fj.reshape(-1), M, y.reshape(-1))
            n_iters = 0

        else:
            du_opt = 0
            n_iters = 0

        return (du_opt, n_iters)


    def hild_matrices(self, ref='constant'):

        if self.Bm.ndim == 1:
            m = 1
        else:
            m = self.Bm.shape[1]
        
        if ref == 'constant':
            Fj1 = -self.Phi.T @ self.Rs_bar
        else:
            Fj1 = -self.Phi.T
        Fj2 = self.Phi.T @ self.F

        Fj3 = self.fw.lin_cost_matrix()

        Kj1 = self.M @ self.Ej_inv

        if self.x_lim is None:
            Fx = np.zeros((1,1))
        else:
            Fx = self.Mx_aux @ self.Fx

        Hj = np.zeros(self.Hj.shape, dtype=self.Hj.dtype)
        Hj[:] = self.Hj[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        #Hj_fxp = (Hj * (2 ** qbase)).astype(np.int64)
        
        DU1 = (-self.Ej_inv)[:m, :]
        DU2 = (-self.Ej_inv @ self.M.T)[:m, :]

        return (Fj1, Fj2, Fj3, Fx, Kj1, Hj, DU1, DU2)
    

    def sim(self, xi, ui, r, n, Bd=None, ud=None, solver='hild'):
        """Simulates closed-loop system with the predictive controller.

        Parameters
        ----------
        xi : :class:`int`, :class:`float`, :class:`list`, :class:`np.array`
            Initial state conditions. Can be passed as a single value, list
            or array, where each element corresponds the initial condition
            of each state. If there are multiple states, and `xi` is a
            single element, or a list with a single element, the initial
            value of all states is set to `xi`.

        ui : :class:`np.array`
            Initial conditions for the control inputs. Can be passed as a
            single value, list or array, where each element corresponds the
            initial value of each input. If there are multiple states, and
            `xi` is a single element, or a list with a single element, the
            initial value of all states is set to `xi`.

        r : :class:`float`, :class:`np.array`
            The set-point. If a single value, it is assumed constant for the
            whole simulation. If a vector, each row is used at each step of
            the simulation.

        n : :class:`int`
            Number of points for the simulation.

        Bd : :class:`np.array`
            An (p, p) numpy matrix, where `p` is the number of disturbances.
            By default, it is `None`.

        ud : :class:`np.array`
            An (p, 1) or (p, n) numpy matrix, where `p` is the number of
            disturbances. If the second dimension is 1, the disturbance is
            considered to be constant during the entire period. Otherwise,
            it must contain `n` values to be used during the entire
            simulation. By default, it is `None`.

        solver : :class:`str`
            Solver to use for constrained problems. Supported options are
            `hild`, `quadprog`, and `cvx`.
            
        Returns
        -------
        data : :class:`dict`
            A dictionary containing the simulation results. The key `u`
            contains the control actions, the key `x_m` contains the states
            and the key `y` contains the output.

        """
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        A = self.A; B = self.B; C = self.C

        if type(xi) is int or type(xi) is float or type(xi) is list:
            xi = np.array(xi).reshape(-1)
        elif type(xi) is np.ndarray:
            xi = np.array(xi).reshape(-1)

        if type(ui) is int or type(ui) is float or type(ui) is list:
            ui = np.array(ui).reshape(-1)
        elif type(xi) is np.ndarray:
            ui = np.array(ui).reshape(-1)
            
        if type(r) is int or type(r) is float:
            r = r * np.ones((n, 1))
        elif type(r) is list:
            r = np.array(r)
        
        if type(r) is np.ndarray and r.ndim == 1:
            r = np.tile(r, (n, 1))
        
        n_xm = Am.shape[0]
        nx = A.shape[0]
        ny = C.shape[0]
        nu = B.shape[1]
        
        xm = np.zeros((n, n_xm))
        x = np.zeros((n, nx))
        y = np.zeros((n, ny))
        xa = np.zeros((A.shape[0], 1))

        u = np.zeros((n, nu))
        du = np.zeros((n, nu))
        n_iters = np.zeros(n)
        
        xm[0] = xi
        dx = xm[0]
        u[0] = ui

        Ky = self.Ky
        Kx = self.Kx

        if Bd is None:
            Bd = np.zeros(Bm.shape)
            ud = np.zeros(u.shape)
        else:
            if type(ud) is int or type(ud) is float or type(ud) is list:
                ud = np.array(ud)
            if ud.ndim == 1:
                ud = np.tile(ud, (n, 1))

        # Simulation
        for i in range(n - 1):
            # Updates the output and dx
            y[i] = Cm @ xm[i]
            dx = xm[i] - dx

            # Computes the control law for sampling instant i
            if (self.u_lim is None) and (self.x_lim is None):
                _du = self.opt_unc(dx, y[i], r[i], u[i], du[i])
            else:
                xa[:n_xm, 0] = dx
                xa[n_xm:, 0] = y[i]
                _du, n_it = self.opt(xm[i], dx, xa, u[i], du[i], r[i], solver=solver) 
                n_iters[i] = n_it
                
            du[i] = _du
            u[i] = u[i] + du[i]
            
            # Applies the control law
            xm[i + 1] = Am @ xm[i] + Bm @ u[i] + Bd @ ud[i]

            # Update variables for next iteration
            dx = xm[i]
            u[i + 1] = u[i]
            du[i + 1] = du[i]

        # Updates last value of y
        y[n - 1] = Cm @ xm[n - 1]

        n_iters[n - 1] = n_iters[n - 2]

        results = {}
        results['u'] = u
        results['xm'] = xm
        results['y'] = y
        results['n_iters'] = n_iters
        results['du'] = du 

        return results


    def _gen(self, scaling=1.0, Bd=None, ref='constant', ftype='src', prefix=None):

        u_lim = self.u_lim
        x_lim = self.x_lim
        
        # Matrices for Hildreth's QP procedure
        if (u_lim is not None) or (x_lim is not None):
            (Fj1, Fj2, Fj3, Fx, Kj1, Hj, DU1, DU2) = self.hild_matrices(ref=ref)
        
        header = self.c_gen.header(ftype=ftype, prefix=prefix)

        includes, end = self.c_gen.includes(ftype=ftype, prefix=prefix)

        if u_lim is not None:
            u_lim = u_lim / scaling
        in_cnt = self.c_gen.cnt(u_lim, self.u_lim_idx, cnt='input', ftype=ftype, prefix=prefix)

        if x_lim is not None:
            x_lim = x_lim / scaling

        st_cnt = self.c_gen.cnt(x_lim, self.x_lim_idx, cnt='state', ftype=ftype, prefix=prefix)
        
        out_idx = self.c_gen.output_idx(self.y_idx, ftype=ftype, prefix=prefix)

        pred_matrices = self.c_gen.Am_Bm_matrices_pred(self.Am, self.Bm, Bd=Bd, ftype=ftype, prefix=prefix)

        kx_ky_gains = self.c_gen.Kx_Ky_gains(self.Kx, self.Ky, ftype=ftype, prefix=prefix)
        
        if (u_lim is not None) or (x_lim is not None):
            qp_matrices = self.c_gen.qp_matrices(self.Ej, self.M, ftype=ftype, prefix=prefix)
            hild_matrices = self.c_gen.hild_matrices(Fj1, Fj2, Fj3, Fx, Kj1, Hj, DU1, DU2, ftype=ftype, prefix=prefix)
        else:
            qp_matrices = ''
            hild_matrices = ''
        
        txt = header + includes + in_cnt + st_cnt +\
              out_idx + pred_matrices + kx_ky_gains +\
              qp_matrices + hild_matrices +\
              end

        return txt
    

    def _gen_defs(self, scaling=1.0, Bd=None, prefix=None):

        x_lim = self.x_lim
        u_lim = self.u_lim
        
        n_xm = self.Am.shape[0]
        n_xa = self.A.shape[0]
        n_pred = self.n_pred
        n_ctl = self.n_ctl

        if (x_lim is not None) or (u_lim is not None):
            n_cnt = self.n_cnt
            n_lambda = self.M.shape[0]
        else:
            n_cnt = 0
            n_lambda = 0

        if self.Cm.ndim == 1:
            ny = 1
        else:
            ny = self.Cm.shape[0]
        
        if self.Bm.ndim == 1:
            nu = 1
        else:
            nu = self.Bm.shape[1]

        if Bd is None:
            nd = 0
        else:
            if Bd.ndim == 1:
                nd = 1
            else:
                nd = Bd.shape[1]

        n_in_cnt = 0
        if self.u_lim_idx is not None:
            n_in_cnt = self.u_lim_idx.shape[0]

        n_st_cnt = 0
        if self.x_lim_idx is not None:
            n_st_cnt = self.x_lim_idx.shape[0]

        lp = self.lp
        
        defs = self.c_gen.defs_header(n_xm, n_xa, ny, nu, nd,
                               n_pred, n_ctl, n_cnt, n_lambda,
                               n_in_cnt, n_st_cnt,
                               lp = lp,
                               scaling=scaling,
                               prefix=prefix)

        return defs

    
    def export(self, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant'):
        
        if prefix is None:
            file_prefix = ''
        else:
            file_prefix = prefix.lower() + '_'

        np.set_printoptions(floatmode='unique', threshold=sys.maxsize)

        src_txt = self._gen(scaling=scaling, Bd=Bd, ref=ref, ftype='src', prefix=prefix)
        header_txt = self._gen(scaling=scaling, Bd=Bd, ref=ref, ftype='header', prefix=prefix)
        defs_txt = self._gen_defs(scaling=scaling, Bd=Bd, prefix=prefix)

        if file_path is not None:
            with open(file_path + file_prefix + 'dmpc_matrices.c', 'w') as efile:
                efile.write(src_txt)
            with open(file_path + file_prefix + 'dmpc_matrices.h', 'w') as efile:
                efile.write(header_txt)
            with open(file_path + file_prefix + 'dmpc_defs.h', 'w') as efile:
                efile.write(defs_txt)
                
        np.set_printoptions(floatmode='fixed', threshold=1000)


class c_gen:

    def __init__(self):
        pass
    
    def _export_np_array_to_c(self, arr, arr_name, fill=True):

        if arr.ndim == 1:
            n = arr.shape[0]
            m = 1
        else:
            if (arr.shape[0] == 1) or (arr.shape[1] == 1):
                arr = arr.flatten()
                n = arr.shape[0]
                m = 1
            else:
                n, m = arr.shape

        arr_str = np.array2string(arr, separator=',')
        arr_str = arr_str.replace('[', '{')
        arr_str = arr_str.replace(']', '}')

        if m == 1:
            arr_txt = '{:}[{:}];'.format(arr_name, n)
        else:
            arr_txt = '{:}[{:}][{:}];'.format(arr_name, n, m)

        if fill is True:
            arr_txt = arr_txt[:-1] + ' = {:};'.format(arr_str)
            
        return arr_txt
    

    def header(self, ftype='src', prefix=None):

        header = '/**\n'\
         ' * @file {:}\n'\
         ' * @brief Header with data to run the DMPC algorithm.\n'\
         ' *\n'\
         ' * This file is generated automatically and should not be modified.\n'\
         ' *\n'\
         ' * The Hj matrix is already generated by flipping the sign and inverting its\n'\
         ' * diagonal elements, so that Hildreth\'s algorithm does not require any \n'\
         ' * divisions.\n'\
         ' *\n'\
         ' */\n'

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '_'
        
        if ftype == 'src':
            file = 'dmpc_matrices.c'
        else:
            file = 'dmpc_matrices.h'
            
        header = header.format(prefix + file)

        return header


    def includes(self, ftype='src', prefix=None):

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '_'

        file = 'dmpc_matrices'

        if ftype == 'src':
            txt = '\n#include "{:}"\n'.format(prefix + file + '.h')
            guard_end = ''
        else:
            def_guard = prefix.upper() + file.upper() + '_H_'
            if prefix is not None: def_guard = def_guard

            def_guard_txt = '#ifndef {:}\n'\
                        '#define {:}\n'
            def_guard_txt = def_guard_txt.format(def_guard, def_guard)   

            include_header = '\n#include "stdint.h"\n'

            txt = '\n' + def_guard_txt + include_header

            guard_end = '\n#endif /* {:} */\n'.format(def_guard)

        return txt, guard_end


    def cnt(self, lim, idx, scaling=1.0, cnt='input', ftype='src', prefix=None):

        if lim is None:
            return ''
        
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        extern = ''
        if ftype != 'src':
            fill = False
            extern = 'extern '

        cnt_txt = 'DMPC_CONFIG_U'
        comment = '/* Input constraints */\n'
        if cnt != 'input':
            cnt_txt = 'DMPC_CONFIG_XM'
            comment = '/* State constraints */\n'
        
        min_txt = extern + 'float {:}{:}_MIN'.format(prefix, cnt_txt)
        max_txt = extern + 'float {:}{:}_MAX'.format(prefix, cnt_txt)
        lim_idx_txt = extern + 'uint32_t {:}{:}_LIM_IDX'.format(prefix, cnt_txt)

        min_txt = self._export_np_array_to_c(lim[0] / scaling, min_txt, fill=fill) + '\n'
        max_txt = self._export_np_array_to_c(lim[1] / scaling, max_txt, fill=fill) + '\n'
        lim_idx_txt = self._export_np_array_to_c(idx, lim_idx_txt, fill=fill) + '\n'

        txt = '\n' + comment + min_txt + max_txt + lim_idx_txt
        
        return txt


    def output_idx(self, idx, ftype='src', prefix=None):

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        extern = ''
        if ftype != 'src':
            fill = False
            extern = 'extern '

        comment = '/* Index of ouputs */\n'
        y_idx_txt = extern + 'uint32_t {:}DMPC_CONFIG_Y_IDX'.format(prefix)

        idx_txt = self._export_np_array_to_c(idx, y_idx_txt, fill=fill) + '\n'

        txt = '\n' + comment + idx_txt
        
        return txt


    def Am_Bm_matrices_pred(self, Am, Bm, Bd=None, ftype='src', prefix=None):

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        nl = '\n'
        extern = ''
        if ftype != 'src':
            fill = False
            extern = 'extern '
            nl = ''

        if Bd is not None:
            B = np.concatenate((Bm, Bd), axis=1)
        else:
            B = Bm
        
        comment = '/*\n'\
                  ' * A and B matrices for prediction.\n'\
                  ' * A corresponds to model matrix Am.\n'\
                  ' * B corresponds to model matrix Bm concatenated with Bd, if Bd exists.\n'\
                  '*/\n'

        A_txt = extern + 'float {:}DMPC_M_A'.format(prefix)
        B_txt = extern + 'float {:}DMPC_M_B'.format(prefix)

        A_txt = self._export_np_array_to_c(Am, A_txt, fill=fill) + '\n'
        B_txt = self._export_np_array_to_c(B, B_txt, fill=fill) + '\n'

        txt = '\n' + comment + A_txt + nl + B_txt
        
        return txt


    def Kx_Ky_gains(self, Kx, Ky, ftype='src', prefix=None):

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        nl = '\n'
        extern = ''
        if ftype != 'src':
            fill = False
            extern = 'extern '
            nl = ''
        
        comment = '/* Optimal Kx and Ky for unconstrained problems */\n'

        Kx_txt = extern + 'float {:}DMPC_Kx'.format(prefix)
        Ky_txt = extern + 'float {:}DMPC_Ky'.format(prefix)

        Kx_txt = self._export_np_array_to_c(Kx, Kx_txt, fill=fill) + '\n'
        Ky_txt = self._export_np_array_to_c(Ky, Ky_txt, fill=fill) + '\n'

        txt = '\n' + comment + Kx_txt + nl + Ky_txt
        
        return txt
    

    def qp_matrices(self, Ej, M, ftype='src', prefix=None):

        n = Ej.shape[0]
        m = M.shape[0]
        
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        extern = ''
        nl = '\n'
        if ftype != 'src':
            fill = False
            extern = 'extern '
            nl = ''

        comment = '\n/*\n * Matrices for QP solvers \n'\
                   ' *\n'\
                   ' * The matrices were generated considering the following problem:\n'\
                   ' *\n'\
                   ' * min (1/2) * DU\' * Ej * DU + DU\' * Fj\n'\
                   ' * DU\n'\
                   ' *\n'\
                   ' * s.t. M * DU <= gam\n'\
                   ' *\n'\
                   ' * The (1/2) term in from of DU\' * Ej * DU needs to be considered in the QP\n'\
                   ' * solver selected, or the solution will appear to be inconsistent.\n'\
                   ' * Note that the Fj and gam matrices are usually updated online, while Ej\n'\
                   ' * and M are static.\n'\
                   ' */\n'

        Ej_txt = extern + 'float {:}DMPC_M_Ej'.format(prefix)
        Fj_txt = nl + extern + 'float {:}DMPC_M_Fj'.format(prefix)

        M_txt = nl + extern + 'float {:}DMPC_M_M'.format(prefix)
        gam_txt = nl + extern + 'float {:}DMPC_M_gam'.format(prefix)

        # Generates dummy matrices for Fj and gam, since they are updated
        # at every iteration.
        Fj = np.zeros(n)
        gam = np.zeros(m)
        
        Ej_txt = self._export_np_array_to_c(Ej, Ej_txt, fill=fill) + '\n'
        Fj_txt = self._export_np_array_to_c(Fj, Fj_txt, fill=False) + '\n'

        M_txt = self._export_np_array_to_c(M, M_txt, fill=fill) + '\n'
        gam_txt = self._export_np_array_to_c(gam, gam_txt, fill=False) + '\n'

        txt = comment + Ej_txt + Fj_txt + M_txt + gam_txt

        return txt
    

    def hild_matrices(self, Fj1, Fj2, Fj3, Fx, Kj1, Hj, DU1, DU2, ftype='src', prefix=None):
        
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix.upper() + '_'

        fill = True
        extern = ''
        nl = '\n'
        if ftype != 'src':
            fill = False
            extern = 'extern '
            nl = ''

        comments = '\n/* Matrices for Hildreth\'s QP procedure */\n'
        
        Fj1_txt = extern + 'float {:}DMPC_M_Fj_1'.format(prefix)
        Fj1_txt = self._export_np_array_to_c(Fj1, Fj1_txt, fill=fill) + '\n'
        
        Fj2_txt = nl + extern + 'float {:}DMPC_M_Fj_2'.format(prefix)
        Fj2_txt = self._export_np_array_to_c(Fj2, Fj2_txt, fill=fill) + '\n'

        if Fj3 is not None:
            Fj3_txt = nl + extern + 'float {:}DMPC_M_Fj_3'.format(prefix)
            Fj3_txt = self._export_np_array_to_c(Fj3, Fj3_txt, fill=fill) + '\n'
        else:
            Fj3_txt = ''
        
        Fx_txt = nl + extern + 'float {:}DMPC_M_Fx'.format(prefix)
        Fx_txt = self._export_np_array_to_c(Fx, Fx_txt, fill=fill) + '\n'
        
        Kj1_txt = nl + extern + 'float {:}DMPC_M_Kj_1'.format(prefix)
        Kj1_txt = self._export_np_array_to_c(Kj1, Kj1_txt, fill=fill) + '\n'
        
        Hj_txt = nl + extern + 'float {:}DMPC_M_Hj'.format(prefix)
        Hj_txt = self._export_np_array_to_c(Hj, Hj_txt, fill=fill) + '\n'
        
        DU1_txt = nl + extern + 'float {:}DMPC_M_DU_1'.format(prefix)
        DU1_txt = self._export_np_array_to_c(DU1, DU1_txt, fill=fill) + '\n'
        
        DU2_txt = nl + extern + 'float {:}DMPC_M_DU_2'.format(prefix)
        DU2_txt = self._export_np_array_to_c(DU2, DU2_txt, fill=fill) + '\n'

        txt = comments + \
              Fj1_txt + Fj2_txt + Fj3_txt + Fx_txt +\
              Kj1_txt + Hj_txt + DU1_txt + DU2_txt
        
        return txt


    def defs_header(self, n_xm, n_xa, ny, nu, nd, n_pred, n_ctl, n_cnt, n_lambda, nu_cnt, n_st_cnt, lp, scaling=1.0, prefix=None):

        header = '/**\n'\
         ' * @file {:}\n'\
         ' * @brief Header with definitions to aid the DMPC algorithm.\n'\
         ' *\n'\
         ' * This file is generated automatically and should not be modified.\n'\
         ' *\n'\
         ' */\n'

        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '_'

        tab_size = len(prefix.upper() + 'DMPC_CONFIG_NLAMBDA')
        tab = '{{:<{:}}}'.format(tab_size + 4)
        
        file = prefix + 'dmpc_defs'
        
        header = header.format(file + '.h')

        def_guard = file.upper() + '_H_'

        def_guard_txt = '\n#ifndef {:}\n'\
                    '#define {:}\n'
        def_guard_txt = def_guard_txt.format(def_guard, def_guard)

        guard_end_txt = '\n#endif /* {:} */\n'.format(def_guard)

        scale_def = (tab + '{:f}f').format(prefix.upper() + 'DMPC_CONFIG_SCALE', scaling)
        scale_txt = '\n/* Scaling factor */\n'\
                     '#define {:}\n'.format(scale_def)

        n_xm_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NXM', n_xm)
        n_xa_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NXA', n_xa)
        n_states_txt = '\n/* Number of model states and augmented states */\n'+\
                       '#define {:}\n'.format(n_xm_def)+\
                       '#define {:}\n'.format(n_xa_def)

        n_pred_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NP', n_pred)
        n_ctl_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NC', n_ctl)
        n_cnt_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NR', n_cnt)
        n_lambda_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NLAMBDA', n_lambda)

        n_hor_txt = '\n/* Prediction, control and constraint horizon */\n'+\
                    '#define {:}\n'.format(n_pred_def)+\
                    '#define {:}\n'.format(n_ctl_def)+\
                    '#define {:}\n'.format(n_cnt_def)+\
                    '#define {:}\n'.format(n_lambda_def)


        n_in_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NU', nu)
        n_out_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NY', ny)
        n_dist_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_ND', nd)
        n_in_out_txt = '\n/* Number of inputs, outputs, and disturbances */\n'+\
                       '#define {:}\n'.format(n_in_def)+\
                       '#define {:}\n'.format(n_out_def)+\
                       '#define {:}\n'.format(n_dist_def)

        n_ctl_def = prefix.upper() + 'DMPC_CONFIG_NC'
        nu_def = prefix.upper() + 'DMPC_CONFIG_NU'
        size_u = tab.format(prefix.upper() + 'DMPC_CONFIG_NC_x_NU') + '({:} * {:})'.format(n_ctl_def, nu_def)
        n_size_u_txt = '\n/* Size of control vector */\n'+\
                       '#define {:}\n'.format(size_u)

        n_input_cnt_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NU_CTR', nu_cnt)
        n_input_cnt_txt = '\n/* Input constraints */\n'+\
                          '#define {:}\n'.format(n_input_cnt_def)

        n_st_cnt_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_NXM_CTR', n_st_cnt)
        n_st_cnt_txt = '\n/* State constraints */\n'+\
                          '#define {:}\n'.format(n_st_cnt_def)

        lp_def = (tab + '{:}').format(prefix.upper() + 'DMPC_CONFIG_LP', lp)
        lp_txt = '\n/* Past samples for freq. penalization */\n'+\
                          '#define {:}\n'.format(lp_def)

        defs_txt = header + def_guard_txt +\
                   scale_txt +\
                   n_states_txt + n_hor_txt + n_in_out_txt +\
                   n_size_u_txt + n_input_cnt_txt + n_st_cnt_txt + lp_txt+\
                   guard_end_txt
        
        return defs_txt

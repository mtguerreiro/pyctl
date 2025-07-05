import numpy as np
import scipy

import pyctl
import qpsolvers as qps


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


def opt_matrices(A, B, C, lp, lc):
    r"""Computes the :math:`F` and :math:`Phi` matrices.

    Parameters
    ----------
    A : :class:`np.array`
        The `A` matrix of the augmented model. An (n, n) numpy matrix.

    B : :class:`np.array`
        The `B` matrix of the augmented model. An (n, m) numpy matrix.

    C : :class:`np.array`
        The `B` matrix of the augmented model. An (q, n) numpy matrix.

    lp : :class:`int`
        Length of prediction horizon.

    lc : :class:`int`
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
    
    F = np.zeros((lp * q, A.shape[1]))
    F[:q, :] = C @ A
    for i in range(1, lp):
        F[q * i : q * (i + 1), :] = F[q * (i - 1) : q * i, :] @ A

    Phi = np.zeros((lp * q, lc * m))
    Phi[:q, :m] = C @ B
    for i in range(1, lp):
        A_p = np.linalg.matrix_power(A, i)
        Phi[ (q * i) : ( q * (i + 1) ), :m] = C @ A_p @ B
        for j in range(lc - 1):
            Phi[ (q * i) : ( q * (i + 1) ), m * (j + 1) : m * (j + 2)] = Phi[ ( q * (i - 1) ) : (q * i), m * j : m * (j + 1)]

    return (F, Phi)


def control_weighting_matrix(r_w, lc):
    r"""Computes the :math:`\bar{R}` matrix.

    Parameters
    ----------
    r_w : :class:`int`, :class:`list`, :class:`np.array`
        The weighting coefficients, as a 1-d numpy array, an integer or a
        list.

    lc : :class:`int`
        Length of the control window.

    Returns
    -------
    R_bar : :class:`np.array`
        An (lc * m, lc * m) matrix, where `m` is the number of coefficients
        (system input signals).
    
    """
    if type(r_w) is int or type(r_w) is float:
        r_w = np.array([r_w])
    elif type(r_w) is list:
        r_w = np.array(r_w)

    m = r_w.shape[0]
    
    R_bar = np.zeros((lc * m, lc * m))
    for i in range(lc):
        R_bar[m * i : m * (i + 1), m * i : m * (i + 1)] = np.diag(r_w)

    return R_bar


def reference_matrix(q, lp):
    r"""Computes the :math:`\bar{R_s}` matrix.

    Parameters
    ----------
    q : :class:`int`
        Number of references (system outputs).
        
    lp : :class:`int`
        Length of prediction horizon.

    Returns
    -------
    R_s_bar : :class:`np.array`
        An (q, lp * q) matrix, where `q` is the number of references
        (system outputs).
    
    """
    R_s_bar = np.tile(np.eye(q), (lp, 1))

    return R_s_bar


def opt_unc_gains(A, B, C, l_pred, l_ctl, rw):
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

    l_pred : :class:`int`
        Length of prediction horizon.

    l_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `l_ctl` is less than or equal to `l_pred`. For
        `l_ctl` less than `l_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `l_ctl = l_pred` is
        assumed.

    n_ctn : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints are enforced.
        Note that `n_ctn` is less than or equal to `l_ctl`. If set to `None`,
        `n_ctn = l_ctl` is assumed.
    
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

    Rs_bar = reference_matrix(q, l_pred)

    R = control_weighting_matrix(rw, l_ctl)

    F, Phi = opt_matrices(A, B, C, l_pred, l_ctl)
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

    l_pred : :class:`int`
        Length of prediction horizon.

    l_ctl : :class:`NoneType`, :class:`int`
        Length of the prediction horizon where the control input increments
        can be set. Note that `l_ctl` is less than or equal to `l_pred`. For
        `l_ctl` less than `l_pred`, the increments are set zero to for the
        remaining prediction steps. If set to `None`, `l_ctl = l_pred` is
        assumed.

    l_u_cnt : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints on the input
        are enforced. Note that `l_u_cnt` is less than or equal to `l_ctl`.
        If set to `None`, `l_u_cnt = l_ctl` is assumed.

    l_x_cnt : :class:`NoneType`, :class:`int`
        Lenght of the prediction horizon where the constraints on the states
        are enforced. Note that `l_x_cnt` is less than or equal to `l_ctl`.
        If set to `None`, `l_x_cnt = l_ctl` is assumed.
        
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
        constraints and set on the input signals. At the moment, constraints
        can set either on all inputs or none of them.

    """
    def __init__(self,
             Am, Bm, Cm,
             l_pred, l_ctl=None, l_u_cnt=None, l_x_cnt=None,
             rw=None, q=None,
             x_lim=None, u_lim=None):

        # System model and augmented model
        self.Am = Am; self.Bm = Bm; self.Cm = Cm
        
        self.A, self.B, self.C = aug(Am, Bm, Cm)

        # Prediction horizon
        self.l_pred = l_pred

        # Control horizon
        if l_ctl is None:
            self.l_ctl = l_pred
        else:
            self.l_ctl = l_ctl

        # Constraints horizon
        if l_u_cnt is None:
            self.l_u_cnt = self.l_ctl
        else:
            self.l_u_cnt = l_u_cnt

        if l_x_cnt is None:
            self.l_x_cnt = self.l_ctl
        else:
            self.l_x_cnt = l_x_cnt
            
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

        # Spectrum weighting factor
        if type(q) is None:
            q = 0.0
        
        if type(q) is int or type(q) is float:
            r_w = np.array([q])

        # Bounds
        if type(x_lim) is list:
            x_lim = np.array(x_lim)
        if type(u_lim) is list:
            u_lim = np.array(u_lim)
        
        self.x_lim = x_lim
        self.u_lim = u_lim

        # Gains for unconstrained problem
        n_xm = Am.shape[0]
        Ky, K_mpc = opt_unc_gains(self.A, self.B, self.C, \
                                  self.l_pred, self.l_ctl, self.rw)
        Kx = K_mpc[:, :n_xm]

        self.Ky = Ky
        self.Kx = Kx

        if x_lim is None:
            self.x_lim_idx = None

        if u_lim is None:
            self.u_lim_idx = None

        self.gen_static_qp_matrices()

        # Creates index vector of outputs
        y_idx = []
        if Cm.ndim != 1:
            Cm = np.sum(Cm, axis=0)
        for i, yi in enumerate(Cm):
            if np.abs(yi) > 0.5: y_idx.append(i)

        self.y_idx = np.array(y_idx)
        
        self.qp = pyctl.qp.QP(self.Ej, self.M, self.Hj, self.F, self.Phi)

    
    def gen_static_qp_matrices(self):
        """Sets constant matrices, to be used later by the optimization.

        """
        A = self.A; B = self.B; C = self.C
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        l_pred = self.l_pred; l_ctl = self.l_ctl;
        l_u_cnt = self.l_u_cnt; l_x_cnt = self.l_x_cnt
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
        
        R_bar = control_weighting_matrix(rw, l_ctl)
        Rs_bar = reference_matrix(q, l_pred)

        self.R_bar = R_bar
        self.Rs_bar = Rs_bar

        # Creates left-hand side inequality constraint, starting first with
        # control inequality constraints
        M = None
        if u_lim is not None:
            M_aux = np.tril( np.tile( np.eye(m), (l_u_cnt, l_ctl) ) )
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
                        
            Fx, Phi_x = opt_matrices(Am, Bm, Cx, l_x_cnt, l_ctl)
            self.Fx = Fx; self.Phi_x = Phi_x
            self.Cx = Cx
            
            self.x_lim = np.array(x_lim_new)
            self.x_lim_idx = np.array(x_lim_idx)

            M_aux = np.tril( np.tile( np.eye(self.x_lim.shape[1]), (l_x_cnt, l_x_cnt) ) )
            self.Mx_aux = M_aux
            Mx = np.concatenate((-M_aux @ Phi_x, M_aux @ Phi_x))

            if M is None:
                M = Mx
            else:
                M = np.concatenate((M, Mx))

        self.M = M

        # QP matrices
        F, Phi = opt_matrices(A, B, C, l_pred, l_ctl)
        self.F = F; self.Phi = Phi

        Ej = Phi.T @ Phi + R_bar
        Ej_inv = np.linalg.inv(Ej)
        self.Ej = Ej; self.Ej_inv = Ej_inv

        if (x_lim is not None) or (u_lim is not None):
            # Creates Hildreth's static matrix        
            self.Hj = self.M @ self.Ej_inv @ self.M.T
        else:
            self.Hj = None


    def gen_dyn_qp_matrices(self, xm, dx, xa, ui, r):
        """Sets dynamic matrices, to be used later by the optimization.

        """
        l_u_cnt = self.l_u_cnt; l_x_cnt = self.l_x_cnt
        F = self.F; Phi = self.Phi
        Rs_bar = self.Rs_bar
       
        u_lim = self.u_lim
        x_lim = self.x_lim

        Fj = -Phi.T @ (Rs_bar @ r.reshape(-1, 1) - F @ xa.reshape(-1, 1))

        # Creates the right-hand side inequality vector, starting first with
        # the control inequality constraints
        y = None
        
        if u_lim is not None:
            u_min = np.tile(-u_lim[0] + ui, l_u_cnt).reshape(-1, 1)
            u_max = np.tile( u_lim[1] - ui, l_u_cnt).reshape(-1, 1)

            y = np.concatenate((u_min, u_max))

        # Now, the state inequality constraints
        if x_lim is not None:
            Mx_aux = self.Mx_aux
            Cx = self.Cx
            Fx = self.Fx
            M_Fx = Mx_aux @ Fx
            x_min = np.tile(-x_lim[0] + Cx @ xm, l_x_cnt).reshape(-1, 1) + M_Fx @ dx.reshape(-1, 1)
            x_max = np.tile( x_lim[1] - Cx @ xm, l_x_cnt).reshape(-1, 1) - M_Fx @ dx.reshape(-1, 1)

            if y is None:
                y = np.concatenate((x_min, x_max))
            else:
                y = np.concatenate((y, x_min, x_max))

        return (Fj, y)


    def opt(self, xm, dx, xa, ui, r, solver='hild'):

        nu = ui.shape[0]

        Fj, y = self.gen_dyn_qp_matrices(xm, dx, xa, ui, r)
        du, n_iters = self.qp.solve(xm, dx, xa, ui, r, Fj, y, solver=solver)

        return (du[:nu], n_iters)


    def sim(self, x0, u0, r, n, Bd=None, ud=None, solver='hild'):
        """Simulates closed-loop system with the predictive controller.

        Parameters
        ----------
        x0 : :class:`int`, :class:`float`, :class:`list`, :class:`np.array`
            Initial state conditions. Can be passed as a single value, list
            or array, where each element corresponds the initial condition
            of each state. If there are multiple states, and `x0` is a
            single element, or a list with a single element, the initial
            value of all states is set to `x0`.

        u0 : :class:`np.array`
            Initial conditions for the control inputs. Can be passed as a
            single value, list or array, where each element corresponds the
            initial value of each input. If there are multiple states, and
            `u0` is a single element, or a list with a single element, the
            initial value of all states is set to `u0`.

        r : :class:`float`, :class:`np.array`
            The setpoint. If a single value, it is assumed constant for the
            whole simulation. If a vector, each row is used at each step of
            the simulation.

        n : :class:`int`
            Number of points for the simulation. The system is simulated for
            `n - 1` points, since the initial condition is assumed to be one
            simulation point as well.

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
            Solver to use for constrained problems. See `System.qp.solvers` for
            a list of available solvers on your system.
            
        Returns
        -------
        data : :class:`dict`
            A dictionary containing the simulation results. The key `u`
            contains the control actions, the key `xm` contains the states
            and the key `y` contains the output.

        """
        Am = self.Am; Bm = self.Bm; Cm = self.Cm
        A = self.A; B = self.B; C = self.C

        if type(x0) is int or type(x0) is float or type(x0) is list:
            x0 = np.array(x0).reshape(-1)
        elif type(x0) is np.ndarray:
            x0 = np.array(x0).reshape(-1)

        if type(u0) is int or type(u0) is float or type(u0) is list:
            u0 = np.array(u0).reshape(-1)
        elif type(xi) is np.ndarray:
            u0 = np.array(u0).reshape(-1)
            
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

        n_iters = np.zeros((n, 1))

        # Initial conditions
        u[0] = u0
        xm[0] = x0
        y[0] = Cm @ xm[0]

        # System change due to initial conditions
        xm[1] = Am @ xm[0] + Bm @ u[0]
        y[1] = Cm @ xm[1]

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
        for i in range(1, n - 1):
            # Updates the output and dx
            dx = xm[i] - xm[i - 1]

            # Computes the control law for sampling instant i
            if (self.u_lim is None) and (self.x_lim is None):
                du = -Ky @ (y[i] - r[i]) + -Kx @ dx
                n_iter = 0
            else:
                xa[:n_xm, 0] = dx
                xa[n_xm:, 0] = y[i]
                du, n_iter = self.opt(xm[i], dx, xa, u[i - 1], r[i], solver=solver)
            
            u[i] = u[i - 1] + du
            n_iters[i] = n_iter

            # Applies the control law
            xm[i + 1] = Am @ xm[i] + Bm @ u[i] + Bd @ ud[i]
            y[i + 1] = Cm @ xm[i]


        # Populates last value of u and n_iters, otherwise they would be zero.
        # This is just for plotting effects, and the last value of u and n_iters
        # do not have any meaning
        u[n - 1] = u[n - 2]
        n_iters[n - 1] = n_iters[n - 2]

        results = {}
        results['u'] = u
        results['xm'] = xm
        results['y'] = y
        results['n_iters'] = n_iters

        return results

    
    def export(self, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant', gen_dll=False, solver_settings=None):
        
        model = self._get_code_gen_model()
        pyctl.code_gen.gen(
            model,
            file_path=file_path, prefix=prefix,
            scaling=scaling, Bd=Bd,
            ref=ref,
            solver_settings=solver_settings
        )

        if gen_dll:
            dll_path = pyctl.code_gen.gen_py_cdmpc_dll(file_path)
            self.qp.set_dll(dll_path)


    def _get_code_gen_model(self):

        model = pyctl.code_gen.CodeGenData

        model.A = self.A
        model.B = self.B
        model.C = self.C
        
        model.Am = self.Am
        model.Bm = self.Bm
        model.Cm = self.Cm

        model.l_pred = self.l_pred
        model.l_ctl = self.l_ctl
        model.l_u_cnt = self.l_u_cnt
        model.l_x_cnt = self.l_x_cnt
        
        model.u_lim = self.u_lim
        model.u_lim_idx = self.u_lim_idx

        model.x_lim = self.x_lim
        model.x_lim_idx = self.x_lim_idx

        model.y_idx = self.y_idx

        model.R_bar = self.R_bar
        model.Rs_bar = self.Rs_bar

        model.M = self.M

        if self.x_lim is not None:
            model.Mx_aux = self.Mx_aux

            model.Fx = self.Fx
            model.Phi_x = self.Phi_x

        model.F = self.F
        model.Phi = self.Phi

        model.Ej = self.Ej
        model.Hj = self.Hj

        model.Kx = self.Kx
        model.Ky = self.Ky

        return model

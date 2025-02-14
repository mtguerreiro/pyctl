import numpy as np
import scipy
import scipy.signal

import sys
import os
from shutil import copytree, ignore_patterns

from dataclasses import dataclass

import osqp


def gen(model, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant', copy_cdmpc_src=False):

    _hild = Hildreth(model)
    _hild.gen(file_path=file_path, prefix=prefix, scaling=scaling, Bd=Bd, ref=ref, copy_cdmpc_src=False)

    _osqp = OSQP(model)
    _osqp.gen(file_path=file_path, scaling=scaling)


class Hildreth:

    def __init__(self, model):

        self.model = model


    def gen(self, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant', copy_cdmpc_src=False):
        
        if prefix is None:
            file_prefix = ''
        else:
            file_prefix = prefix.lower() + '_'

        np.set_printoptions(floatmode='unique', threshold=sys.maxsize)

        src_txt = self._gen(scaling=scaling, Bd=Bd, ref=ref, ftype='src', prefix=prefix)
        header_txt = self._gen(scaling=scaling, Bd=Bd, ref=ref, ftype='header', prefix=prefix)
        defs_txt = self._gen_defs(scaling=scaling, Bd=Bd, prefix=prefix)

        if file_path is not None:
            if copy_cdmpc_src is True:
                self.copy_static_sources(path=file_path)
                
            with open(file_path + file_prefix + 'dmpc_matrices.c', 'w') as efile:
                efile.write(src_txt)
            with open(file_path + file_prefix + 'dmpc_matrices.h', 'w') as efile:
                efile.write(header_txt)
            with open(file_path + file_prefix + 'dmpc_defs.h', 'w') as efile:
                efile.write(defs_txt)
                
        np.set_printoptions(floatmode='fixed', threshold=1000)
        
        
    def _gen(self, scaling=1.0, Bd=None, ref='constant', ftype='src', prefix=None):

        u_lim = self.model.u_lim
        x_lim = self.model.x_lim
        
        # Matrices for Hildreth's QP procedure
        if (u_lim is not None) or (x_lim is not None):
            (Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2) = self.hild_matrices(ref=ref)
        
        header = self.header(ftype=ftype, prefix=prefix)

        includes, end = self.includes(ftype=ftype, prefix=prefix)

        if u_lim is not None:
            u_lim = u_lim / scaling
        in_cnt = self.cnt(u_lim, self.model.u_lim_idx, cnt='input', ftype=ftype, prefix=prefix)

        if x_lim is not None:
            x_lim = x_lim / scaling

        st_cnt = self.cnt(x_lim, self.model.x_lim_idx, cnt='state', ftype=ftype, prefix=prefix)
        
        out_idx = self.output_idx(self.model.y_idx, ftype=ftype, prefix=prefix)

        pred_matrices = self.Am_Bm_matrices_pred(self.model.Am, self.model.Bm, Bd=Bd, ftype=ftype, prefix=prefix)

        kx_ky_gains = self.Kx_Ky_gains(self.model.Kx, self.model.Ky, ftype=ftype, prefix=prefix)
        
        if (u_lim is not None) or (x_lim is not None):
            qp_matrices = self.qp_matrices(self.model.Ej, self.model.M, ftype=ftype, prefix=prefix)
            hild_matrices = self.hild_matrices_txt(Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2, ftype=ftype, prefix=prefix)
        else:
            qp_matrices = ''
            hild_matrices = ''
        
        txt = header + includes + in_cnt + st_cnt +\
              out_idx + pred_matrices + kx_ky_gains +\
              qp_matrices + hild_matrices +\
              end

        return txt


    def _gen_defs(self, scaling=1.0, Bd=None, prefix=None):

        x_lim = self.model.x_lim
        u_lim = self.model.u_lim
        
        n_xm = self.model.Am.shape[0]
        n_xa = self.model.A.shape[0]
        n_pred = self.model.n_pred
        n_ctl = self.model.n_ctl

        if (x_lim is not None) or (u_lim is not None):
            n_cnt = self.model.n_cnt
            n_lambda = self.model.M.shape[0]
        else:
            n_cnt = 0
            n_lambda = 0

        if self.model.Cm.ndim == 1:
            ny = 1
        else:
            ny = self.model.Cm.shape[0]
        
        if self.model.Bm.ndim == 1:
            nu = 1
        else:
            nu = self.model.Bm.shape[1]

        if Bd is None:
            nd = 0
        else:
            if Bd.ndim == 1:
                nd = 1
            else:
                nd = Bd.shape[1]

        n_in_cnt = 0
        if self.model.u_lim_idx is not None:
            n_in_cnt = self.model.u_lim_idx.shape[0]

        n_st_cnt = 0
        if self.model.x_lim_idx is not None:
            n_st_cnt = self.model.x_lim_idx.shape[0]
        
        defs = self.defs_header(
            n_xm, n_xa, ny, nu, nd,
            n_pred, n_ctl, n_cnt, n_lambda,
            n_in_cnt, n_st_cnt,
            scaling=scaling,
            prefix=prefix
        )

        return defs


    def hild_matrices(self, ref='constant'):

        Ej_inv = np.linalg.inv(self.model.Ej)
        
        if self.model.Bm.ndim == 1:
            m = 1
        else:
            m = self.model.Bm.shape[1]
        
        if ref == 'constant':
            Fj1 = -self.model.Phi.T @ self.model.Rs_bar
        else:
            Fj1 = -self.model.Phi.T
        Fj2 = self.model.Phi.T @ self.model.F

        Kj1 = self.model.M @ Ej_inv

        if self.model.x_lim is None:
            Fx = np.zeros((1,1))
        else:
            Fx = self.model.Mx_aux @ self.model.Fx

        Hj = np.zeros(self.model.Hj.shape, dtype=self.model.Hj.dtype)
        Hj[:] = self.model.Hj[:]
        Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        DU1 = (-Ej_inv)[:m, :]
        DU2 = (-Ej_inv @ self.model.M.T)[:m, :]

        return (Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2)


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

        min_txt = _export_np_array_to_c(lim[0] / scaling, min_txt, fill=fill) + '\n'
        max_txt = _export_np_array_to_c(lim[1] / scaling, max_txt, fill=fill) + '\n'
        lim_idx_txt = _export_np_array_to_c(idx, lim_idx_txt, fill=fill) + '\n'

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

        idx_txt = _export_np_array_to_c(idx, y_idx_txt, fill=fill) + '\n'

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

        A_txt = _export_np_array_to_c(Am, A_txt, fill=fill) + '\n'
        B_txt = _export_np_array_to_c(B, B_txt, fill=fill) + '\n'

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

        Kx_txt = _export_np_array_to_c(Kx, Kx_txt, fill=fill) + '\n'
        Ky_txt = _export_np_array_to_c(Ky, Ky_txt, fill=fill) + '\n'

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
        
        Ej_txt = _export_np_array_to_c(Ej, Ej_txt, fill=fill) + '\n'
        Fj_txt = _export_np_array_to_c(Fj, Fj_txt, fill=False) + '\n'

        M_txt = _export_np_array_to_c(M, M_txt, fill=fill) + '\n'
        gam_txt = _export_np_array_to_c(gam, gam_txt, fill=False) + '\n'

        txt = comment + Ej_txt + Fj_txt + M_txt + gam_txt

        return txt
    

    def hild_matrices_txt(self, Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2, ftype='src', prefix=None):
        
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
        Fj1_txt = _export_np_array_to_c(Fj1, Fj1_txt, fill=fill) + '\n'
        
        Fj2_txt = nl + extern + 'float {:}DMPC_M_Fj_2'.format(prefix)
        Fj2_txt = _export_np_array_to_c(Fj2, Fj2_txt, fill=fill) + '\n'
        
        Fx_txt = nl + extern + 'float {:}DMPC_M_Fx'.format(prefix)
        Fx_txt = _export_np_array_to_c(Fx, Fx_txt, fill=fill) + '\n'
        
        Kj1_txt = nl + extern + 'float {:}DMPC_M_Kj_1'.format(prefix)
        Kj1_txt = _export_np_array_to_c(Kj1, Kj1_txt, fill=fill) + '\n'
        
        Hj_txt = nl + extern + 'float {:}DMPC_M_Hj'.format(prefix)
        Hj_txt = _export_np_array_to_c(Hj, Hj_txt, fill=fill) + '\n'
        
        DU1_txt = nl + extern + 'float {:}DMPC_M_DU_1'.format(prefix)
        DU1_txt = _export_np_array_to_c(DU1, DU1_txt, fill=fill) + '\n'
        
        DU2_txt = nl + extern + 'float {:}DMPC_M_DU_2'.format(prefix)
        DU2_txt = _export_np_array_to_c(DU2, DU2_txt, fill=fill) + '\n'

        txt = comments + \
              Fj1_txt + Fj2_txt + Fx_txt +\
              Kj1_txt + Hj_txt + DU1_txt + DU2_txt
        
        return txt


    def defs_header(self, n_xm, n_xa, ny, nu, nd, n_pred, n_ctl, n_cnt, n_lambda, nu_cnt, n_st_cnt, scaling=1.0, prefix=None):

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

        defs_txt = header + def_guard_txt +\
                   scale_txt +\
                   n_states_txt + n_hor_txt + n_in_out_txt +\
                   n_size_u_txt + n_input_cnt_txt + n_st_cnt_txt +\
                   guard_end_txt
        
        return defs_txt


class OSQP:

    def __init__(self, model):
        
        self.model = model


    def gen(self, file_path = '', scaling=1.0):

        (P, q, A, l, u) = self.gen_osqp_matrices(scaling=scaling)

        prob = osqp.OSQP()

        prob.setup(
            P, q, A, l, u,
            scaled_termination=False,
            check_termination=0,
            max_iter=40,
            warm_start=False,
            #eps_abs=1e-5, eps_rel=1e-5,
            scaling=100,
            adaptive_rho=False
        )

        osqp_src_gen = file_path + r'/osqp_code_gen'
        osqp_src_copy = file_path + r'/osqp'

        prob.codegen(
            osqp_src_gen,
            parameters='vectors',
            force_rewrite=True,
            FLOAT=True, LONG=False,
            compile_python_ext=False
        )

        copytree(
            osqp_src_gen + r'/include', osqp_src_copy,
            dirs_exist_ok=True, ignore=ignore_patterns('*qdldl_types.h')
        )

        copytree(
            osqp_src_gen + r'/src/osqp', osqp_src_copy,
            dirs_exist_ok=True
        )

        
    def gen_osqp_matrices(self, scaling=1.0):

        n_cnt = self.model.n_cnt
        nu = self.model.Bm.shape[1]
        nx_cnt = self.model.x_lim.shape[1] if self.model.x_lim is not None else 0
        
        bounds_size = round( self.model.M.shape[0] / 2 )
        lin_cost_size = self.model.Ej.shape[0]

        P = self.model.Ej
        P = scipy.sparse.csc_matrix(P)

        A = np.zeros([bounds_size, lin_cost_size])
        A[:(nu * n_cnt), :] = self.model.M[ nu * n_cnt : 2 * (nu * n_cnt), : ]
        A[(nu * n_cnt):, :] = self.model.M[ (2 * nu + nx_cnt) * n_cnt :, :]
        A = scipy.sparse.csc_matrix(A)

        l = np.zeros(bounds_size)
        l[:(nu * n_cnt)] = self.model.u_lim[0, 0]
        l[(nu * n_cnt):] = self.model.x_lim[0, 0]

        u = np.zeros(bounds_size)
        u[:(nu * n_cnt)] = self.model.u_lim[1, 0]
        u[(nu * n_cnt):] = self.model.x_lim[1, 0]

        q = -self.model.Phi.T @ self.model.Rs_bar @ np.ones(nu) / 1000

        return (P, q, A, l, u)
        
def _export_np_array_to_c(arr, arr_name, fill=True):

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


def _copy_cdmpc_static_sources(self, path=''):

    src = os.path.dirname( os.path.dirname(pyctl.__file__) )
    src = src + '/cdmpc/'
    copytree(src, path, dirs_exist_ok=True)

    
@dataclass
class CodeGenData:

    Am : np.ndarray
    Bm : np.ndarray
    Cm : np.ndarray

    A : np.ndarray
    B : np.ndarray
    C : np.ndarray
    
    n_pred : int
    n_ctl : int
    n_cnt : int

    u_lim = np.ndarray
    u_lim_idx = np.ndarray

    x_lim = np.ndarray
    x_lim_idx = np.ndarray

    y_idx = np.ndarray
    
    R_bar : np.ndarray
    Rs_bar : np.ndarray

    M : np.ndarray
    Mx_aux : np.ndarray

    Fx : np.ndarray
    Phi_x : np.ndarray

    F : np.ndarray    
    Phi : np.ndarray

    Ej : np.ndarray
    Hj : np.ndarray

    Kx : np.ndarray
    Ky : np.ndarray

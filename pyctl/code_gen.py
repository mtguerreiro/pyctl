import numpy as np
import scipy
import scipy.signal

import pyctl
import osqp

from dataclasses import dataclass, field

import shutil

import sys, os
import subprocess
import platform


def _np_array_to_c(a):

    a_cstr = np.array2string(a, separator=',')
    a_cstr = a_cstr.replace('[', '{')
    a_cstr = a_cstr.replace(']', '}')

    return a_cstr

def gen_dmpc_data_defs(
    n_xm, n_xa, ny, nu, nd,
    l_pred, l_ctl, l_u_cnt, l_x_cnt, n_lambda,
    n_u_cnt, n_x_cnt, aux_size
    ):

    txt = f"""#ifndef DMPC_DATA_H_
#define DMPC_DATA_H_

#include "stdint.h"

/* Solver settings */
#define DMPC_CONFIG_HILD_TOL           1e-06
#define DMPC_CONFIG_HILD_N_ITER        200
#define DMPC_CONFIG_HILD_FIXED_ITER    1

#if !defined(DMPC_CONFIG_SOLVER_HILD) && !defined(DMPC_CONFIG_SOLVER_OSQP)
#define DMPC_CONFIG_SOLVER_HILD
#endif

typedef struct{{
    uint32_t n_xm;
    uint32_t n_xa;
    uint32_t nu;
    uint32_t nd;
    uint32_t ny;
    uint32_t n_lambda;
    uint32_t l_u_cnt;
    uint32_t n_x_cnt;
    uint32_t l_x_cnt;
    uint32_t u_size;
    float u_min[{n_u_cnt}];
    float u_max[{n_u_cnt}];
    uint32_t u_cnt_idx[{n_u_cnt}];
    float x_min[{n_x_cnt}];
    float x_max[{n_x_cnt}];
    uint32_t x_cnt_idx[{n_x_cnt}];
    uint32_t y_idx[{ny}];
    float A[{n_xm}][{n_xm}];
    float B[{n_xm}][{nu + nd}];
    float Kx[{ny}][{n_xm}];
    float Ky[{ny}][{ny}];
    float Ej[{l_ctl * nu}][{l_ctl * nu}];
    float Fj[{l_ctl * nu}];
    float M[{2 * (l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt)}][{l_ctl * nu}];
    float gam[{2 * (l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt)}];
    float Fj_1[{l_ctl * nu}][{ny}];
    float Fj_2[{l_ctl * nu}][{n_xa}];
    float Fx[{l_x_cnt * n_xm}][{n_xm}];
    float Kj_1[{n_lambda}][{l_ctl * nu}];
    float Hj[{n_lambda}][{n_lambda}];
    float Kj[{n_lambda}];
    float lambda[{n_lambda}];
    float DU_1[{nu}][{l_ctl * nu}];
    float DU_2[{nu}][{n_lambda}];
    float xa[{n_xa}];
    float dx[{n_xm}];
    float e[{ny}];
    float auxm1[{aux_size}];
    float auxm2[{aux_size}];
    float ldata[{l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt}];
    float udata[{l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt}];
}}dmpc_data_t;

extern dmpc_data_t dmpc_data;

#endif /* DMPC_DATA_H_ */
    """

    return txt

def gen_dmpc_data_src(
    n_xm, n_xa, ny, nu, nd, n_lambda,
    l_ctl, l_u_cnt,
    l_x_cnt, n_x_cnt,
    u_min, u_max, u_cnt_idx,
    x_min, x_max, x_cnt_idx,
    y_idx,
    A, B, Kx, Ky, Ej, M,
    Fj_1, Fj_2, Fx, Kj_1, Hj, DU_1, DU_2
    ):
    
    txt = f"""#include "dmpc_data.h"

/*
 * Matrices for QP solvers
 *
 * The matrices were generated considering the following problem:
 *
 * min (1/2) * DU' * Ej * DU + DU' * Fj
 * DU
 *
 * s.t. M * DU <= gam
 *
 * The (1/2) term in from of DU' * Ej * DU needs to be considered in the QP
 * solver selected, or the solution will appear to be inconsistent.
 * Note that the Fj and gam matrices are usually updated online, while Ej
 * and M are static.
 */

dmpc_data_t dmpc_data = {{
  .n_xm = {n_xm},
  .n_xa = {n_xa},
  .nu = {nu},
  .nd = {nd},
  .ny = {ny},
  .n_lambda = {n_lambda},
  .l_u_cnt = {l_u_cnt},
  .n_x_cnt = {n_x_cnt},
  .l_x_cnt = {l_x_cnt},
  .u_size = {l_ctl * nu},
  .u_min = {_np_array_to_c(u_min)},
  .u_max = {_np_array_to_c(u_max)},
  .u_cnt_idx = {_np_array_to_c(u_cnt_idx)},
  .x_min = {_np_array_to_c(x_min)},
  .x_max = {_np_array_to_c(x_max)},
  .x_cnt_idx = {_np_array_to_c(x_cnt_idx)},
  .y_idx = {_np_array_to_c(y_idx)},
  .A = {_np_array_to_c(A)},
  .B = {_np_array_to_c(B)},
  .Kx = {_np_array_to_c(Kx)},
  .Ky = {_np_array_to_c(Ky)},
  .Ej = {_np_array_to_c(Ej)},
  .Fj = {{0}},
  .M = {_np_array_to_c(M)},
  .gam = {{0}},
  .Fj_1 = {_np_array_to_c(Fj_1)},
  .Fj_2 = {_np_array_to_c(Fj_2)},
  .Fx = {_np_array_to_c(Fx)},
  .Kj_1 = {_np_array_to_c(Kj_1)},
  .Hj = {_np_array_to_c(Hj)},
  .DU_1 = {_np_array_to_c(DU_1)},
  .DU_2 = {_np_array_to_c(DU_2)}
}};
    """

    return txt
    
    
def gen(model, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant', solver_settings=None):

    if solver_settings is None:
        solver_settings = Solver_Settings()
    
    _hild = Hildreth(model, settings=solver_settings.hild)
    _hild.gen(file_path=file_path, prefix=prefix, scaling=scaling, Bd=Bd, ref=ref)

    _osqp = OSQP(model, settings=solver_settings.osqp)
    _osqp.gen(file_path=file_path, scaling=scaling)


def gen_py_cdmpc_dll(source_path):

    pyctl_root = os.path.dirname( os.path.dirname(pyctl.__file__) )
    cdmpc_py_dir = pyctl_root + r'/cdmpc_py/cdmpc/'
    cdmpc_py_build_dir = pyctl_root + r'/cdmpc_py/build/'

    plat = platform.system()
    
    if not os.path.exists(cdmpc_py_dir):
        os.makedirs(cdmpc_py_dir)

    if not os.path.exists(cdmpc_py_build_dir):
        os.makedirs(cdmpc_py_build_dir)

    shutil.copytree(source_path, cdmpc_py_dir, dirs_exist_ok=True)

    plat = platform.system()

    subprocess.run(['cmake', '..', '-G', 'Ninja'], cwd=cdmpc_py_build_dir, check=True)

    if plat == 'Linux':
        dll_path = cdmpc_py_build_dir + r'libcdmpc_py.so'
    elif plat == 'Windows':
        dll_path = cdmpc_py_build_dir + r'libcdmpc_py.dll'
    else:
        raise ValueError('Platform not supported for code generation.')

    subprocess.run(['ninja'], cwd=cdmpc_py_build_dir, check=True)

    return dll_path


@dataclass
class Hildreth_Solver_Settings:
    tol : float = 1e-6
    max_iter : int = 200
    fixed_iter : bool = True
    normalize_h: bool = False

class Hildreth:

    def __init__(self, model, settings=None):

        if settings is None:
            settings = Hildreth_Solver_Settings()

        self.settings = settings
        self.model = model


    def gen(self, file_path='', prefix=None, scaling=1.0, Bd=None, ref='constant', solver_settings=None):

        if solver_settings is None:
            solver_settings = self.settings
        
        pyctl_root = os.path.dirname( os.path.dirname(pyctl.__file__) )
        cdmpc_path = pyctl_root + r'/cdmpc/'
        shutil.copytree(
            cdmpc_path, file_path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.git', '.gitignore')
            )
            
        if prefix is None:
            file_prefix = ''
        else:
            file_prefix = prefix.lower() + '_'

        np.set_printoptions(floatmode='unique', threshold=sys.maxsize)

        src_txt, defs_txt = self._gen(scaling=scaling, Bd=Bd, ref=ref, ftype='src', prefix=prefix, normalize=solver_settings.normalize_h)

        if file_path is not None:                
            with open(file_path + file_prefix + 'dmpc_data.c', 'w') as efile:
                efile.write(src_txt)
            with open(file_path + file_prefix + 'dmpc_data.h', 'w') as efile:
                efile.write(defs_txt)
                
        np.set_printoptions(floatmode='fixed', threshold=1000)
        
        
    def _gen(self, scaling=1.0, Bd=None, ref='constant', ftype='src', prefix=None, normalize=False):

        u_lim = self.model.u_lim
        x_lim = self.model.x_lim

        n_xm = self.model.Am.shape[0]
        n_xa = self.model.A.shape[0]
        l_pred = self.model.l_pred
        l_ctl = self.model.l_ctl

        if (x_lim is not None) or (u_lim is not None):
            l_u_cnt = self.model.l_u_cnt
            l_x_cnt = self.model.l_x_cnt
            n_lambda = self.model.M.shape[0]
        else:
            l_u_cnt = 0
            l_x_cnt = 0
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
            B = self.model.Bm
            nd = 0
        else:
            B = np.concatenate((self.model.Bm, Bd), axis=1)
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
        
        # Matrices for Hildreth's QP procedure
        if (u_lim is not None) or (x_lim is not None):
            (Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2) = self.hild_matrices(ref=ref, normalize=normalize)

        aux_size = max(n_xm, nu, n_lambda, Fx.shape[0], Fj1.shape[0])

        src_txt = gen_dmpc_data_src(
            n_xm, n_xa, ny, nu, nd, n_lambda,
            l_ctl, l_u_cnt,
            l_x_cnt, n_st_cnt,
            u_lim[0], u_lim[1], self.model.u_lim_idx,
            x_lim[0], x_lim[1], self.model.x_lim_idx,
            self.model.y_idx,
            self.model.Am, B, self.model.Kx, self.model.Ky, self.model.Ej, self.model.M,
            Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2
        )

        defs_txt = gen_dmpc_data_defs(
            n_xm, n_xa, ny, nu, nd,
            l_pred, l_ctl, l_u_cnt, l_x_cnt, n_lambda,
            n_in_cnt, n_st_cnt,
            aux_size
        )
        
        return src_txt, defs_txt


    def hild_matrices(self, ref='constant', normalize=False):

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

        if normalize == True:
            Hj = np.zeros(self.Hj.shape, dtype=self.Hj.dtype)
            Hj[:] = self.Hj[:]
            Hj_aux = Hj.copy()
            np.fill_diagonal(Hj_aux, 1)
            Hj = np.linalg.inv(-np.diag(np.diag(Hj))) @ Hj_aux
        else:
            Hj = np.zeros(self.model.Hj.shape, dtype=self.model.Hj.dtype)
            Hj[:] = self.model.Hj[:]
            Hj[np.eye(Hj.shape[0],dtype=bool)] = -1 / Hj[np.eye(Hj.shape[0],dtype=bool)]

        DU1 = (-Ej_inv)[:m, :]
        DU2 = (-Ej_inv @ self.model.M.T)[:m, :]

        return (Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2)


@dataclass
class OSQP_Solver_Settings:
    scaled_termination : bool = False    
    check_termination : int = 0
    max_iter : int = 40
    warm_start : bool = False
    scaling : int = 100
    adaptive_rho : bool = False
    #eps_abs : float = 1e-5
    #eps_rel : float = 1e-5

class OSQP:

    def __init__(self, model, settings=None):

        if settings is None:
            settings = OSQP_Solver_Settings()

        self.settings = settings
        self.model = model


    def gen(self, file_path = '', scaling=1.0, settings=None):

        if not settings:
            settings = self.settings

        (P, q, A, l, u) = self.gen_osqp_matrices(scaling=scaling)

        prob = osqp.OSQP()

        prob.setup(
            P, q, A, l, u,
            scaled_termination=settings.scaled_termination,
            check_termination=settings.check_termination,
            max_iter=settings.max_iter,
            warm_start=settings.warm_start,
            #eps_abs=1e-5, eps_rel=1e-5,
            scaling=settings.scaling,
            adaptive_rho=settings.adaptive_rho
        )

        osqp_src_gen = f'{file_path}/osqp_code_gen'
        osqp_src_copy = f'{file_path}/osqp'

        prob.codegen(
            osqp_src_gen,
            parameters='vectors',
            force_rewrite=True,
            use_float=True
        )
        
        shutil.copytree(
            f'{osqp_src_gen}/inc/private', osqp_src_copy,
            dirs_exist_ok=True
        )

        shutil.copytree(
            f'{osqp_src_gen}/inc/public', osqp_src_copy,
            dirs_exist_ok=True
        )
        
        shutil.copytree(
            f'{osqp_src_gen}/src', osqp_src_copy,
            dirs_exist_ok=True
        )

        shutil.copy(f'{osqp_src_gen}/osqp_configure.h', osqp_src_copy)
        shutil.copy(f'{osqp_src_gen}/workspace.h', osqp_src_copy)        
        shutil.copy(f'{osqp_src_gen}/workspace.c', osqp_src_copy)

        
    def gen_osqp_matrices(self, scaling=1.0):

        nu = self.model.Bm.shape[1]
        
        P = self.model.Ej
        P = scipy.sparse.csc_matrix(P)

        q = -self.model.Phi.T @ self.model.Rs_bar @ np.ones(nu) / 1000
        
        if (self.model.x_lim is None) and (self.model.u_lim is None):
            A = np.eye(self.model.Ej.shape[0])
            l = -np.inf * np.ones(A.shape[0])
            u =  np.inf * np.ones(A.shape[0])
            A = scipy.sparse.csc_matrix(A)
        
        else:
            l_u_cnt = self.model.l_u_cnt
            l_x_cnt = self.model.l_x_cnt
            nx_cnt = self.model.x_lim.shape[1] if self.model.x_lim is not None else 0
            
            bounds_size = round( self.model.M.shape[0] / 2 )
            lin_cost_size = self.model.Ej.shape[0]            
            A = np.zeros([bounds_size, lin_cost_size])
            A[:(nu * l_u_cnt), :] = self.model.M[ nu * l_u_cnt : 2 * (nu * l_u_cnt), : ]
            A[(nu * l_u_cnt):, :] = self.model.M[ (2 * nu * l_u_cnt + nx_cnt * l_x_cnt):, : ]
            A = scipy.sparse.csc_matrix(A)

            l = np.zeros(bounds_size)
            u = np.zeros(bounds_size)

            if self.model.x_lim is not None:
                l[(nu * l_x_cnt):] = self.model.x_lim[0, 0]
                u[(nu * l_x_cnt):] = self.model.x_lim[1, 0]

            if self.model.u_lim is not None:
                l[:(nu * l_u_cnt)] = self.model.u_lim[0, 0]
                u[:(nu * l_u_cnt)] = self.model.u_lim[1, 0]

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

    
@dataclass
class CodeGenData:

    Am : np.ndarray
    Bm : np.ndarray
    Cm : np.ndarray

    A : np.ndarray
    B : np.ndarray
    C : np.ndarray
    
    l_pred : int
    l_ctl : int
    l_u_cnt : int
    l_x_cnt : int

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


@dataclass
class Solver_Settings:
    hild : Hildreth_Solver_Settings = field(default_factory=Hildreth_Solver_Settings)
    osqp : OSQP_Solver_Settings = field(default_factory=OSQP_Solver_Settings)

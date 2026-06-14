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

"""
Documentation todo:
- explain the problem that is solved, e.g.

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
 
"""

def _np_array_to_c(a):

    a_cstr = np.array2string(a, separator=',')
    a_cstr = a_cstr.replace('[', '{')
    a_cstr = a_cstr.replace(']', '}')

    return a_cstr

def gen_dmpc_data_defs(inst=None):

    if inst is None:
        inst = 'inst'
    
    txt = f"""#ifndef DMPC_{inst.upper()}_DATA_
#define DMPC_{inst.upper()}_DATA_

#include "dmpc_data.h"

extern dmpc_inst_t {inst};

#endif /* DMPC_{inst.upper()}_DATA_H_ */
    """

    return txt

def gen_dmpc_prob_data(model, Bd=None, ref='constant'):

    if ref == 'constant':
        Fj1 = -model.Phi.T @ model.Rs_bar
    else:
        Fj1 = -self.model.Phi.T
    Fj2 = model.Phi.T @ model.F

    Fx = model.Mx_aux @ model.Fx

    u_lim = model.u_lim
    x_lim = model.x_lim

    n_xm = model.Am.shape[0]
    n_xa = model.A.shape[0]
    l_pred = model.l_pred
    l_ctl = model.l_ctl

    if (x_lim is not None) or (u_lim is not None):
        l_u_cnt = model.l_u_cnt
        l_x_cnt = model.l_x_cnt
    else:
        l_u_cnt = 0
        l_x_cnt = 0

    if model.Cm.ndim == 1:
        ny = 1
    else:
        ny = model.Cm.shape[0]
    
    if model.Bm.ndim == 1:
        nu = 1
    else:
        nu = model.Bm.shape[1]

    if Bd is None:
        B = model.Bm
        nd = 0
    else:
        B = np.concatenate((model.Bm, Bd), axis=1)
        if Bd.ndim == 1:
            nd = 1
        else:
            nd = Bd.shape[1]

    n_u_cnt = 0
    if model.u_lim_idx is not None:
        n_u_cnt = model.u_lim_idx.shape[0]

    n_x_cnt = 0
    if model.x_lim_idx is not None:
        n_x_cnt = model.x_lim_idx.shape[0]

    aux_size = max(n_xm, nu, Fx.shape[0], Fj1.shape[0])
            
    txt = f"""
static float x[{n_xm}] = {{0.0f}};
static float x_1[{n_xm}] = {{0.0f}};
static float r[{ny}] = {{0.0f}};
static float u_1[{nu+nd}] = {{0.0f}};
static float du[{nu+nd}] = {{0.0f}};

static float u_min[{n_u_cnt}] = {_np_array_to_c(model.u_lim[0])};
static float u_max[{n_u_cnt}] = {_np_array_to_c(model.u_lim[1])};
static uint32_t u_cnt_idx[{n_u_cnt}] = {_np_array_to_c(model.u_lim_idx)};

static float x_min[{n_x_cnt}] = {_np_array_to_c(model.x_lim[0])};
static float x_max[{n_x_cnt}] = {_np_array_to_c(model.x_lim[1])};
static uint32_t x_cnt_idx[{n_x_cnt}] = {_np_array_to_c(model.x_lim_idx)};

static uint32_t y_idx[{ny}] = {_np_array_to_c(model.y_idx)};

static float A[{n_xm}][{n_xm}] = {_np_array_to_c(model.Am)};
static float B[{n_xm}][{nu + nd}] = {_np_array_to_c(B)};
static float Kx[{ny}][{n_xm}] = {_np_array_to_c(model.Kx)};
static float Ky[{ny}][{ny}] = {_np_array_to_c(model.Ky)};
static float Ej[{l_ctl * nu}][{l_ctl * nu}] = {_np_array_to_c(model.Ej)};
static float Fj[{l_ctl * nu}] = {{0.0f}};
static float M[{2 * (l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt)}][{l_ctl * nu}] = {_np_array_to_c(model.M)};
static float gam[{2 * (l_u_cnt * n_u_cnt + l_x_cnt * n_x_cnt)}] = {{0.0f}};
static float Fj_1[{l_ctl * nu}][{ny}] = {_np_array_to_c(Fj1)};
static float Fj_2[{l_ctl * nu}][{n_xa}] = {_np_array_to_c(Fj2)};
static float Fx[{l_x_cnt * n_xm}][{n_xm}] = {_np_array_to_c(Fx)};
static float xa[{n_xa}] = {{0.0f}};
static float dx[{n_xm}] = {{0.0f}};
static float e[{ny}] = {{0.0f}};
static float auxm1[{aux_size}] = {{0.0f}};
static float auxm2[{aux_size}] = {{0.0f}};

dmpc_data_t dmpc_data = {{
  .x = x,
  .x_1 = x_1,
  .r = r,
  .u_1 = u_1,
  .du = du,
  .n_xm = {n_xm},
  .n_xa = {n_xa},
  .nu = {nu},
  .nd = {nd},
  .ny = {ny},
  .l_u_cnt = {l_u_cnt},
  .n_x_cnt = {n_x_cnt},
  .l_x_cnt = {l_x_cnt},
  .u_size = {l_ctl * nu},
  .u_min = u_min,
  .u_max = u_max,
  .u_cnt_idx = u_cnt_idx,
  .x_min = x_min,
  .x_max = x_max,
  .x_cnt_idx = x_cnt_idx,
  .y_idx = y_idx,
  .A = (float *)A,
  .B = (float *)B,
  .Kx = (float *)Kx,
  .Ky = (float *)Ky,
  .Ej = (float *)Ej,
  .Fj = Fj,
  .M = (float *)M,
  .gam = gam,
  .Fj_1 = (float *)Fj_1,
  .Fj_2 = (float *)Fj_2,
  .Fx = (float *)Fx,
  .xa = xa,
  .dx = dx,
  .e = e,
  .auxm1 = auxm1,
  .auxm2 = auxm2
}};
"""
    return txt

def gen_dmpc_solver_data(model, Kj_1, Hj, DU_1, DU_2):

    n_lambda = model.M.shape[0]
    l_ctl = model.l_ctl
    if model.Bm.ndim == 1:
        nu = 1
    else:
        nu = model.Bm.shape[1]
        
    txt = f"""
static float Kj_1[{n_lambda}][{l_ctl * nu}] = {_np_array_to_c(Kj_1)};
static float Hj[{n_lambda}][{n_lambda}] = {_np_array_to_c(Hj)};
static float Kj[{n_lambda}] = {{0.0f}};
static float lambda[{n_lambda}] = {{0.0f}};
static float DU_1[{nu}][{l_ctl * nu}] = {_np_array_to_c(DU_1)};
static float DU_2[{nu}][{n_lambda}] = {_np_array_to_c(DU_2)};
static float aux[{n_lambda}] = {{0.0f}};

dmpc_hild_data_t dmpc_hild_data = {{
  .fixed_iter = 1,
  .n_iter = 200,
  .tol = 1e-6,
  .max_iter = 200,
  .n_lambda = {n_lambda},
  .Kj_1 = (float *)Kj_1,
  .Hj = (float *)Hj,
  .Kj = Kj,
  .lambda = lambda,
  .DU_1 = (float *)DU_1,
  .DU_2 = (float *)DU_2,
  .aux = aux
}};
"""
    return txt


def gen_dmpc_src(prob_txt, solver_txt):
    
    txt = f"""#include "dmpc_inst_data.h"
#include "dmpc_hild.h"

{prob_txt}

{solver_txt}

dmpc_inst_t inst = {{
    .prob_data = &dmpc_data,
    .solver_data = (void *)&dmpc_hild_data,
    .solve = dmpc_hild_solve
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
            with open(file_path + file_prefix + 'dmpc_inst_data.c', 'w') as efile:
                efile.write(src_txt)
            with open(file_path + file_prefix + 'dmpc_inst_data.h', 'w') as efile:
                efile.write(defs_txt)
                
        np.set_printoptions(floatmode='fixed', threshold=1000)
        
        
    def _gen(self, scaling=1.0, Bd=None, ref='constant', ftype='src', prefix=None, normalize=False):

        
        # Matrices for Hildreth's QP procedure
        if (self.model.u_lim is not None) or (self.model.x_lim is not None):
            (Kj1, Hj, DU1, DU2) = self.hild_matrices(ref=ref, normalize=normalize)

        prob_txt = gen_dmpc_prob_data(self.model, Bd=Bd)
        solver_txt = gen_dmpc_solver_data(self.model, Kj1, Hj, DU1, DU2)

        src_txt = gen_dmpc_src(prob_txt, solver_txt)
        
##        src_txt = gen_dmpc_data_src(
##            n_xm, n_xa, ny, nu, nd, n_lambda,
##            l_pred, l_ctl, l_u_cnt, l_x_cnt,
##            n_in_cnt, n_st_cnt,
##            u_lim[0], u_lim[1], self.model.u_lim_idx,
##            x_lim[0], x_lim[1], self.model.x_lim_idx,
##            self.model.y_idx,
##            self.model.Am, B, self.model.Kx, self.model.Ky, self.model.Ej, self.model.M,
##            Fj1, Fj2, Fx, Kj1, Hj, DU1, DU2,
##            aux_size
##        )

        defs_txt = gen_dmpc_data_defs()
        
        return src_txt, defs_txt


    def hild_matrices(self, ref='constant', normalize=False):

        Ej_inv = np.linalg.inv(self.model.Ej)
        
        if self.model.Bm.ndim == 1:
            m = 1
        else:
            m = self.model.Bm.shape[1]
        
        Kj1 = self.model.M @ Ej_inv

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

        return (Kj1, Hj, DU1, DU2)


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

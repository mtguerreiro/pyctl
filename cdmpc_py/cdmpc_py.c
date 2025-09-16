
#include <stdint.h>
#include "cdmpc/dmpc.h"


int cdmpc_py_step(
    float *xm, float *xm_1,
    float *r, float *u_1,
    float *D, float *Xp,
    float *du){
    
    uint32_t n_iters;

    /* Optimization */
    dmpcOpt(xm, xm_1, r, u_1, D, Xp, &n_iters, du);

    return n_iters;
}


#include <stdint.h>
#include "cdmpc/dmpc.h"
#include "cdmpc/dmpc_data_hild.h"

int cdmpc_py_step(float *xm, float *xm_1, 
                  float *r, float *u_1,
                  float *du){
    
    int status;

    inst.prob_data->x = xm;
    inst.prob_data->x_1 = xm_1;
    inst.prob_data->r = r;
    inst.prob_data->u_1 = u_1;
    inst.prob_data->du = du;

    /* Optimization */
    status = dmpcOpt(&inst);

    if( status == 0 ) status = (int) inst.prob_data->n_iters;

    return status;
}

cmake_minimum_required(VERSION 3.15)

set(PROJ "cdmpc_py")

project(${PROJ})

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

add_compile_options(
    -Wall
    -Wextra
    -O1
)

add_library(
    ${PROJ}
    SHARED
    cdmpc_py.c
    cdmpc/dmpc.c
    cdmpc/dmpc_data.c
    cdmpc/dmpc_matrices.c
    cdmpc/mvops.c
    cdmpc/qp.c
    cdmpc/osqp/auxil.c
    cdmpc/osqp/algebra_libs.c
    cdmpc/osqp/csc_math.c
    cdmpc/osqp/csc_utils.c
    cdmpc/osqp/error.c
    cdmpc/osqp/kkt.c
    cdmpc/osqp/matrix.c
    cdmpc/osqp/osqp_api.c
    cdmpc/osqp/qdldl.c
    cdmpc/osqp/qdldl_interface.c
    cdmpc/osqp/scaling.c
    cdmpc/osqp/util.c
    cdmpc/osqp/vector.c
    cdmpc/osqp/workspace.c
)

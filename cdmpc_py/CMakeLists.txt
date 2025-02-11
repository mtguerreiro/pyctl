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
        cdmpc/dmpc_matrices.c
        cdmpc/mvops.c
        cdmpc/qp.c
        cdmpc/osqp/auxil.c
        cdmpc/osqp/error.c
        cdmpc/osqp/lin_alg.c
        cdmpc/osqp/osqp.c
        cdmpc/osqp/proj.c
        cdmpc/osqp/qdldl.c
        cdmpc/osqp/qdldl_interface.c
        cdmpc/osqp/scaling.c
        cdmpc/osqp/util.c
        cdmpc/osqp/workspace.c
)


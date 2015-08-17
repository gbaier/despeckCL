set(KERNEL_PATH ${despeckCL_SOURCE_DIR}/src/compute/nlinsar/cl_kernels/)
FILE(READ ${despeckCL_SOURCE_DIR}/include/compute/nlinsar/sim_measures.h SIM_MEASURES)

configure_file (
    ${KERNEL_PATH}precompute_similarities_1st_pass.cl.in
    ${KERNEL_PATH}precompute_similarities_1st_pass.cl
)

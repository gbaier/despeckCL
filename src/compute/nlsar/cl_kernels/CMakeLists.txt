FILE(READ ${despeckCL_SOURCE_DIR}/src/compute/nlsar/sim_measures.h SIM_MEASURES)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pixel_similarities_2x2.cl.in
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pixel_similarities_2x2.cl
)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pixel_similarities_3x3.cl.in
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pixel_similarities_3x3.cl
)

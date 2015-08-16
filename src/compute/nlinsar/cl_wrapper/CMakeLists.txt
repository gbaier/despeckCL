include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper)

add_library(nlinsar_opencl_wrapper precompute_similarities.cpp 
                                   precompute_patch_similarities.cpp
                                   compute_weights.cpp
                                   compute_number_of_looks.cpp
                                   transpose.cpp
                                   precompute_filter_values.cpp
                                   compute_insar.cpp
                                   )

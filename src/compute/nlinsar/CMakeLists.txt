add_subdirectory(cl_kernels)
add_subdirectory(cl_wrapper)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

add_library(nlinsar_lib nlinsar.cpp
                        insarsim_simu.cpp
                        nlinsar_sub_image.cpp
                        smoothing)

target_link_libraries(nlinsar_lib nlinsar_opencl_wrapper
                                  common_opencl_wrappers
                                  insar_data
                                  clcfg)

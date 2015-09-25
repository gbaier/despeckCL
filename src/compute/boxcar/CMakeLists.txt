add_subdirectory(cl_kernels)
add_subdirectory(cl_wrapper)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper)

add_library(boxcar_lib boxcar.cpp boxcar_sub_image.cpp)
target_link_libraries(boxcar_lib boxcar_opencl_wrapper insar_data clcfg)

include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)

add_library(boxcar_opencl_wrapper boxcar_wrapper.cpp)
target_link_libraries(boxcar_opencl_wrapper insar_data ${OpenCL_LIBRARIES} )

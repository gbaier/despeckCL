include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)

set(infile      "${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper/boxcar_wrapper.h.in")
set(kernel_file "${despeckCL_SOURCE_DIR}/src/compute/boxcar/cl_kernels/boxcar_kernel.cl")
set(outfile     "${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper/boxcar_wrapper.h")
FILE(READ ${kernel_file} KERNEL_SOURCE_RAW)
STRING(REGEX REPLACE "\\\\" "\\\\\\\\" KERNEL_SOURCE ${KERNEL_SOURCE_RAW})
STRING(REPLACE "\n" " \\\n" CMAKE_KERNEL_SOURCE ${KERNEL_SOURCE})
configure_file(${infile} ${outfile})

add_library(boxcar_opencl_wrapper boxcar_wrapper.cpp)
target_link_libraries(boxcar_opencl_wrapper clcfg ${OpenCL_LIBRARIES})

include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/routines)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

add_library(nlsar_routines routines.cpp)

target_link_libraries(nlsar_routines ${OpenCL_LIBRARIES} nlsar_cl_wrappers)

add_subdirectory(cl_kernels)
add_subdirectory(cl_wrapper)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrapper)

add_library(nlsar_lib nlsar.cpp
                      nlsar_sub_image.cpp
                      stats.cpp
                      get_dissims.cpp)

# Testing Code
add_executable(sim_measures_test sim_measures_test.cpp)

add_test(NAME NLSAR_sim_measures
         COMMAND sim_measures_test)

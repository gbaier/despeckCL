add_subdirectory(cl_kernels)
add_subdirectory(cl_wrappers)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

add_library(nlsar_lib nlsar.cpp
                      nlsar_filter_sub_image.cpp
                      stats.cpp
                      get_dissims.cpp)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()

target_link_libraries(nlsar_lib nlsar_cl_wrappers
                                common_opencl_wrappers
                                insar_data
                                clcfg
                                logging
                                best_params_lib
                                best_weights_copy_lib
                                ${GSL_LIBRARIES})

add_library(best_params_lib best_params.cpp)
add_library(best_weights_copy_lib best_weights_copy.cpp)


# Testing Code
add_executable(sim_measures_test sim_measures_test.cpp)

add_executable(best_params_test best_params_test.cpp)
target_link_libraries(best_params_test best_params_lib) 

add_executable(best_weights_copy_test best_weights_copy_test.cpp)
target_link_libraries(best_weights_copy_test best_weights_copy_lib) 

add_test(NAME NLSAR_sim_measures
         COMMAND sim_measures_test)

add_test(NAME NLSAR_best_params
         COMMAND best_params_test)

add_test(NAME NLSAR_best_weights_copy
         COMMAND best_weights_copy_test)

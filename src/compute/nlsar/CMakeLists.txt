add_subdirectory(cl_kernels)
add_subdirectory(cl_wrappers)
add_subdirectory(routines)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/routines)

add_library(nlsar_lib OBJECT
            nlsar.cpp
            nlsar_filter_sub_image.cpp
            stats.cpp
            get_dissims.cpp
            best_params.cpp
            best_weights_copy.cpp
            best_alpha_copy.cpp
            routines/routines.cpp
            tile_size.cpp
           )

# Testing Code
add_executable(sim_measures_test sim_measures_test.cpp)
add_executable(best_params_test best_params_test.cpp best_params.cpp)
add_executable(best_weights_copy_test best_weights_copy_test.cpp best_weights_copy.cpp)
add_executable(stats_test stats_test.cpp stats.cpp)
target_link_libraries(stats_test ${GSL_LIBRARIES})

add_test(NAME NLSAR_sim_measures
         COMMAND sim_measures_test)

add_test(NAME NLSAR_best_params
         COMMAND best_params_test)

add_test(NAME NLSAR_best_weights_copy
         COMMAND best_weights_copy_test)

add_test(NAME NLSAR_stats
         COMMAND stats_test)

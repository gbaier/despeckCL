add_subdirectory(cl_kernels)
add_subdirectory(cl_wrappers)
add_subdirectory(routines)
add_subdirectory(training)


include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/common/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlsar/routines)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlsar/training)

add_library(nlsar_lib OBJECT
            nlsar.cpp
            nlsar_filter_sub_image.cpp
            stats.cpp
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

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()
target_link_libraries(stats_test ${GSL_LIBRARIES})

add_executable(nlsar_unit_tests nlsar_unit_test.cpp
  cl_wrappers/covmat_rescale_test.cpp
  cl_wrappers/covmat_spatial_avg_test.cpp
  cl_wrappers/covmat_create_test.cpp
  cl_wrappers/compute_pixel_similarities_2x2_test.cpp
  cl_wrappers/compute_patch_similarities_test.cpp
  )

set_property(TARGET nlsar_unit_tests PROPERTY CXX_STANDARD 14)

target_link_libraries(nlsar_unit_tests
                      despeckcl
                      clcfg
                      ${OpenCL_LIBRARIES}
                      pthread
                      gtest
                      gtest_main
                      gmock
                      gmock_main)

add_test(NAME NLSAR_unit_tests
         COMMAND nlsar_unit_tests)

add_test(NAME NLSAR_sim_measures
         COMMAND sim_measures_test)

add_test(NAME NLSAR_best_params
         COMMAND best_params_test)

add_test(NAME NLSAR_best_weights_copy
         COMMAND best_weights_copy_test)

add_test(NAME NLSAR_stats
         COMMAND stats_test)

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
add_executable(nlsar_unit_tests nlsar_unit_test.cpp
  cl_wrappers/covmat_rescale_test.cpp
  cl_wrappers/covmat_spatial_avg_test.cpp
  cl_wrappers/covmat_create_test.cpp
  cl_wrappers/compute_pixel_similarities_2x2_test.cpp
  cl_wrappers/compute_patch_similarities_test.cpp
  cl_wrappers/compute_nl_statistics_test.cpp
  cl_wrappers/compute_alphas_test.cpp
  cl_wrappers/compute_enls_nobias_test.cpp
  cl_wrappers/copy_best_weights_test.cpp
  cl_wrappers/covmat_decompose_test.cpp
  cl_wrappers/weighted_means_test.cpp
  cl_wrappers/copy_symm_weights_test.cpp
  sim_measures_test.cpp
  best_params_test.cpp
  best_weights_copy_test.cpp
  stats_test.cpp
)

set_property(TARGET nlsar_unit_tests PROPERTY CXX_STANDARD 14)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()
target_link_libraries(nlsar_unit_tests
                      despeckcl
                      clcfg
                      ${OpenCL_LIBRARIES}
                      ${GSL_LIBRARIES}
                      pthread
                      gtest
                      gtest_main
                      gmock
                      gmock_main)

add_test(NAME NLSAR_unit_tests
         COMMAND nlsar_unit_tests)

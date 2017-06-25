include_directories(${despeckCL_SOURCE_DIR}/src)
include_directories(${despeckCL_SOURCE_DIR}/src/data)
include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/utils)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/goldstein)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/goldstein/cl_wrappers)
include_directories(${CMAKE_BINARY_DIR}/ext/clfft/src/src/include)
include_directories(${CMAKE_BINARY_DIR}/ext/clfft/build/include)

add_library(goldstein_lib OBJECT
            goldstein.cpp
            goldstein_filter_sub_image.cpp
            tile_size.cpp
           )

target_include_directories(goldstein_lib PUBLIC ${CMAKE_BINARY_DIR}/ext/clfft/src/src/include)
target_include_directories(goldstein_lib PUBLIC ${CMAKE_BINARY_DIR}/ext/clfft/build/include)
set(CLFFT_LIBRARIES "${CMAKE_BINARY_DIR}/ext/clfft/build/library/libclFFT.a")

add_dependencies(goldstein_lib clfft)

add_subdirectory(cl_wrappers)

# testing code
add_executable(goldstein_unit_tests goldstein_unit_test.cpp
  cl_wrappers/goldstein_patch_ft_test.cpp
  cl_wrappers/patches_unpack_test.cpp
  cl_wrappers/patches_pack_test.cpp
  cl_wrappers/patches_packing_test.cpp
  )

target_link_libraries(goldstein_unit_tests
                      despeckcl
                      clcfg
                      ${OpenCL_LIBRARIES}
                      ${CLFFT_LIBRARIES}
                      pthread
                      gtest
                      gtest_main
                      gmock
                      gmock_main)

add_test(NAME GOLDSTEIN_unit_tests
         COMMAND goldstein_unit_tests)

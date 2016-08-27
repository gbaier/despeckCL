include_directories(${despeckCL_SOURCE_DIR}/include/utils)

include_directories(${gtest_SOURCE_DIR}/include)

add_executable(utils_unit_tests
               optimal_tiling.cpp
               optimal_tiling_test.cpp)

target_link_libraries(utils_unit_tests
                      gtest_main)

enable_testing()
add_test(UTILS_UNIT_TESTS utils_unit_tests)

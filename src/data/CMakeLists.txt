include_directories(${despeckCL_SOURCE_DIR}/include/data)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${gmock_SOURCE_DIR}/include)

add_executable(data_unit_tests
               tile_test.cpp)

target_link_libraries(data_unit_tests
                      gtest
                      gtest_main
                      gmock
                      gmock_main)

add_test(DATA_UNIT_TESTS data_unit_tests)

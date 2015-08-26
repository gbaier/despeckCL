add_subdirectory(compute)
add_subdirectory(clcfg)
add_subdirectory(data)
add_subdirectory(utils)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)

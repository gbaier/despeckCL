add_subdirectory(cl_wrappers)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/goldstein)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/goldstein/cl_wrappers)

add_library(goldstein_lib OBJECT
            goldstein.cpp
            goldstein_filter_sub_image.cpp
           )

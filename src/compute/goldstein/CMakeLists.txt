add_subdirectory(cl_wrappers)

include_directories(${despeckCL_SOURCE_DIR}/src)
include_directories(${despeckCL_SOURCE_DIR}/src/data)
include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/utils)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/goldstein)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/goldstein/cl_wrappers)

add_library(goldstein_lib OBJECT
            goldstein.cpp
            goldstein_filter_sub_image.cpp
            tile_size.cpp
           )

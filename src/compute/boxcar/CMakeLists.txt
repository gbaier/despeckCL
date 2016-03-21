add_subdirectory(cl_wrapper)

add_library(boxcar_lib OBJECT
            boxcar.cpp
            boxcar_sub_image.cpp
           )

target_include_directories(boxcar_lib PUBLIC
                           ${despeckCL_SOURCE_DIR}/include/compute/boxcar
                           ${despeckCL_SOURCE_DIR}/include/compute/boxcar/cl_wrapper)

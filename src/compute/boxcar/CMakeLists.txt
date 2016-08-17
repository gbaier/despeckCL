add_subdirectory(cl_wrapper)

add_library(boxcar_lib OBJECT
            boxcar.cpp
            boxcar_sub_image.cpp
           )

target_include_directories(boxcar_lib PUBLIC)

add_subdirectory(cl_kernels)
add_subdirectory(cl_wrappers)

add_library(nlinsar_lib OBJECT
            nlinsar.cpp
            insarsim_simu.cpp
            nlinsar_sub_image.cpp
            ${despeckCL_SOURCE_DIR}/src/clcfg/clcfg.cpp
            smoothing)


target_include_directories(nlinsar_lib PUBLIC 
                           ${despeckCL_SOURCE_DIR}/include/compute/nlinsar
                           ${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrappers
                           ${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

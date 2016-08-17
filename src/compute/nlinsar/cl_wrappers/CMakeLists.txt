include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlinsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/common/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlinsar)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/src/compute/nlinsar/cl_wrappers/*.in")

foreach(infile ${infiles})
    get_filename_component(file_name_we ${infile} NAME_WE)
    get_filename_component(file_dir ${infile} DIRECTORY)
    set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/${file_name_we}.cl)
    set(outfile ${file_dir}/${file_name_we}.h)
    FILE(READ ${kernel_file} KERNEL_SOURCE)
    STRING(REPLACE "\\" "\\\\" KERNEL_SOURCE_EDIT "${KERNEL_SOURCE}")
    STRING(REPLACE "\n" "\\n\"\n\"" CMAKE_KERNEL_SOURCE "${KERNEL_SOURCE_EDIT}")
    SET(CMAKE_KERNEL_SOURCE "${CMAKE_KERNEL_SOURCE}\\n")
    configure_file(${infile} ${outfile})
endforeach()

add_library(nlinsar_opencl_wrapper OBJECT
            cl_wrappers.cpp
            precompute_similarities_1st_pass.cpp 
            precompute_similarities_2nd_pass.cpp 
            precompute_patch_similarities.cpp
            compute_weights.cpp
            transpose.cpp
            precompute_filter_values.cpp
            compute_insar.cpp
           )

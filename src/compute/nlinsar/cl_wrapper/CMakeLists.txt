include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper/*.in")

foreach(infile ${infiles})
    get_filename_component(file_name_we ${infile} NAME_WE)
    get_filename_component(file_dir ${infile} DIRECTORY)
    set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/${file_name_we}.cl)
    set(outfile ${file_dir}/${file_name_we}.h)
    FILE(READ ${kernel_file} KERNEL_SOURCE_RAW)
    STRING(REGEX REPLACE "\\\\" "\\\\\\\\" KERNEL_SOURCE ${KERNEL_SOURCE_RAW})
    STRING(REPLACE "\n" " \\\n" CMAKE_KERNEL_SOURCE ${KERNEL_SOURCE})
    configure_file(${infile} ${outfile})
endforeach()

add_library(nlinsar_opencl_wrapper precompute_similarities_1st_pass.cpp 
                                   precompute_similarities_2nd_pass.cpp 
                                   precompute_patch_similarities.cpp
                                   compute_weights.cpp
                                   compute_number_of_looks.cpp
                                   transpose.cpp
                                   precompute_filter_values.cpp
                                   compute_insar.cpp
                                   )

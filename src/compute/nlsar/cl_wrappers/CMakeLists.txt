include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/utils)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/nlsar/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/common/cl_wrappers)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/src/compute/nlsar/cl_wrappers/*.in")

function(read_clfile kernel_file)
    FILE(READ ${kernel_file} KERNEL_SOURCE)
    STRING(REPLACE "\\" "\\\\" KERNEL_SOURCE_EDIT "${KERNEL_SOURCE}")
    STRING(REPLACE "\n" "\\n\"\n\"" CMAKE_KERNEL_SOURCE "${KERNEL_SOURCE_EDIT}")
    SET(CMAKE_KERNEL_SOURCE "${CMAKE_KERNEL_SOURCE}\\n" PARENT_SCOPE)
endfunction(read_clfile)

set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/patch_similarities_row_pass.cl)
read_clfile(${kernel_file})
SET(CMAKE_KERNEL_SOURCE_ROW_PASS "${CMAKE_KERNEL_SOURCE}\\n")

set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/patch_similarities_col_pass.cl)
read_clfile(${kernel_file})
SET(CMAKE_KERNEL_SOURCE_COL_PASS "${CMAKE_KERNEL_SOURCE}\\n")

foreach(infile ${infiles})
    get_filename_component(file_name_we ${infile} NAME_WE)
    get_filename_component(file_dir ${infile} DIRECTORY)
    set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/${file_name_we}.cl)
    set(outfile ${file_dir}/${file_name_we}.h)
    read_clfile(${kernel_file})
    configure_file(${infile} ${outfile})
endforeach()

add_library(nlsar_cl_wrappers OBJECT
            cl_wrappers.cpp
            covmat_spatial_avg.cpp
            covmat_rescale.cpp
            covmat_create.cpp
            compute_pixel_similarities_2x2.cpp
            compute_pixel_similarities_3x3.cpp
            compute_patch_similarities.cpp
            compute_weights.cpp
            compute_nl_statistics.cpp
            compute_alphas.cpp
            compute_enls_nobias.cpp
            copy_symm_weights.cpp
            copy_best_weights.cpp
            covmat_decompose.cpp
            weighted_means.cpp)

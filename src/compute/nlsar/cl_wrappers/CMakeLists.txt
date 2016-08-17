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
            compute_patch_similarities.cpp
            compute_weights.cpp
            compute_nl_statistics.cpp
            compute_alphas.cpp
            compute_enls_nobias.cpp
            copy_symm_weights.cpp
            copy_best_weights.cpp
            covmat_decompose.cpp
            weighted_means.cpp
                             )
# Testing Code
add_executable(covmat_rescale_test covmat_rescale_test.cpp covmat_rescale.cpp)
target_link_libraries(covmat_rescale_test clcfg ${OpenCL_LIBRARIES})

add_executable(covmat_spatial_avg_test covmat_spatial_avg_test.cpp covmat_spatial_avg.cpp)
target_link_libraries(covmat_spatial_avg_test clcfg ${OpenCL_LIBRARIES})

add_executable(covmat_create_test covmat_create_test.cpp covmat_create.cpp)
target_link_libraries(covmat_create_test clcfg ${OpenCL_LIBRARIES})

add_executable(compute_pixel_similarities_2x2_test compute_pixel_similarities_2x2_test.cpp compute_pixel_similarities_2x2.cpp)
target_link_libraries(compute_pixel_similarities_2x2_test clcfg ${OpenCL_LIBRARIES})

add_executable(compute_patch_similarities_test compute_patch_similarities_test.cpp compute_patch_similarities.cpp)
target_link_libraries(compute_patch_similarities_test clcfg ${OpenCL_LIBRARIES})

add_executable(compute_nl_statistics_test compute_nl_statistics_test.cpp compute_nl_statistics.cpp)
target_link_libraries(compute_nl_statistics_test clcfg ${OpenCL_LIBRARIES})

add_executable(compute_alphas_test compute_alphas_test.cpp compute_alphas.cpp)
target_link_libraries(compute_alphas_test clcfg ${OpenCL_LIBRARIES})

add_executable(compute_enls_nobias_test compute_enls_nobias_test.cpp compute_enls_nobias.cpp)
target_link_libraries(compute_enls_nobias_test clcfg ${OpenCL_LIBRARIES})

add_executable(copy_best_weights_test copy_best_weights_test.cpp copy_best_weights.cpp)
target_link_libraries(copy_best_weights_test clcfg ${OpenCL_LIBRARIES})

add_executable(covmat_decompose_test covmat_decompose_test.cpp covmat_decompose.cpp covmat_create.cpp)
target_link_libraries(covmat_decompose_test clcfg ${OpenCL_LIBRARIES})

add_executable(weighted_means_test weighted_means_test.cpp weighted_means.cpp)
target_link_libraries(weighted_means_test clcfg ${OpenCL_LIBRARIES})

add_executable(copy_symm_weights_test copy_symm_weights_test.cpp copy_symm_weights.cpp)
target_link_libraries(copy_symm_weights_test clcfg ${OpenCL_LIBRARIES})

add_test(NAME NLSAR_covmat_rescale
         COMMAND covmat_rescale_test)

add_test(NAME NLSAR_covmat_spatial_avg
         COMMAND covmat_spatial_avg_test)

add_test(NAME NLSAR_covmat_create
         COMMAND covmat_create_test)

add_test(NAME NLSAR_compute_pixel_similarities_2x2
         COMMAND compute_pixel_similarities_2x2_test)

add_test(NAME NLSAR_compute_patch_similarities
         COMMAND compute_patch_similarities_test)

add_test(NAME NLSAR_compute_nl_statistics
         COMMAND compute_nl_statistics_test)

add_test(NAME NLSAR_compute_alphas
         COMMAND compute_alphas_test)

add_test(NAME NLSAR_compute_enls_nobias
         COMMAND compute_enls_nobias_test)

add_test(NAME NLSAR_covmat_decompose
         COMMAND covmat_decompose_test)

add_test(NAME NLSAR_weighted_means
         COMMAND weighted_means_test)

add_test(NAME NLSAR_copy_symm_weights
         COMMAND copy_symm_weights_test)

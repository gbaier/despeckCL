include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrapper)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrapper/*.in")

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

add_library(nlsar_opencl_wrapper covmat_spatial_avg.cpp
                                 covmat_rescale.cpp
                                 covmat_create.cpp
                                 compute_pixel_similarities_2x2.cpp
                                 compute_patch_similarities.cpp
                                 compute_weights.cpp
                                 compute_nl_statistics.cpp
                                 compute_alphas.cpp
                                 covmat_decompose.cpp
                                 weighted_means.cpp
                                 )
target_link_libraries(nlsar_opencl_wrapper clcfg ${OpenCL_LIBRARIES})

# Testing Code
add_executable(covmat_rescale_test     covmat_rescale_test.cpp)
target_link_libraries(covmat_rescale_test nlsar_opencl_wrapper) 

add_executable(covmat_spatial_avg_test covmat_spatial_avg_test.cpp)
target_link_libraries(covmat_spatial_avg_test nlsar_opencl_wrapper) 

add_executable(covmat_create_test covmat_create_test.cpp)
target_link_libraries(covmat_create_test nlsar_opencl_wrapper) 

add_executable(compute_pixel_similarities_2x2_test compute_pixel_similarities_2x2_test.cpp)
target_link_libraries(compute_pixel_similarities_2x2_test nlsar_opencl_wrapper) 

add_executable(compute_patch_similarities_test compute_patch_similarities_test.cpp)
target_link_libraries(compute_patch_similarities_test nlsar_opencl_wrapper) 

add_executable(compute_nl_statistics_test compute_nl_statistics_test.cpp)
target_link_libraries(compute_nl_statistics_test nlsar_opencl_wrapper) 

add_executable(compute_alphas_test compute_alphas_test.cpp)
target_link_libraries(compute_alphas_test nlsar_opencl_wrapper) 

add_executable(covmat_decompose_test covmat_decompose_test.cpp)
target_link_libraries(covmat_decompose_test nlsar_opencl_wrapper) 

add_executable(weighted_means_test weighted_means_test.cpp)
target_link_libraries(weighted_means_test nlsar_opencl_wrapper) 


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

add_test(NAME NLSAR_covmat_decompose
         COMMAND covmat_decompose_test)

add_test(NAME NLSAR_weighted_means
         COMMAND weighted_means_test)

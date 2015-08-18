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

add_library(nlsar_opencl_wrapper covmat_spatial_avg.cpp covmat_rescale.cpp)
target_link_libraries(nlsar_opencl_wrapper clcfg ${OpenCL_LIBRARIES})

add_executable(nlsar_kernel_test covmat_spatial_avg_test.cpp covmat_rescale_test.cpp)
target_link_libraries(nlsar_kernel_test nlsar_opencl_wrapper) 

add_test(NAME NLSAR
         COMMAND nlsar_kernel_test)

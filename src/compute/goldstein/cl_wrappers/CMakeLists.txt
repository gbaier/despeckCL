include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/goldstein/cl_wrappers)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/include/compute/goldstein/cl_wrappers/*.in")

function(read_clfile kernel_file)
    FILE(READ ${kernel_file} KERNEL_SOURCE)
    STRING(REPLACE "\\" "\\\\" KERNEL_SOURCE_EDIT "${KERNEL_SOURCE}")
    STRING(REPLACE "\n" "\\n\"\n\"" CMAKE_KERNEL_SOURCE "${KERNEL_SOURCE_EDIT}")
    SET(CMAKE_KERNEL_SOURCE "${CMAKE_KERNEL_SOURCE}\\n" PARENT_SCOPE)
endfunction(read_clfile)

foreach(infile ${infiles})
    get_filename_component(file_name_we ${infile} NAME_WE)
    get_filename_component(file_dir ${infile} DIRECTORY)
    set(kernel_file ${CMAKE_CURRENT_SOURCE_DIR}/../cl_kernels/${file_name_we}.cl)
    set(outfile ${file_dir}/${file_name_we}.h)
    read_clfile(${kernel_file})
    configure_file(${infile} ${outfile})
endforeach()

add_library(goldstein_cl_wrappers OBJECT
            patches_unpack.cpp
            patches_pack.cpp
           )
# Testing Code
add_executable(patches_packing_test patches_packing_test.cpp)
target_link_libraries(patches_packing_test despeckcl)

add_executable(patches_unpack_test patches_unpack_test.cpp)
target_link_libraries(patches_unpack_test despeckcl)

add_test(NAME GOLDSTEIN_patches_packing
         COMMAND patches_packing_test)

add_test(NAME GOLDSTEIN_patches_unpack
         COMMAND patches_unpack_test)

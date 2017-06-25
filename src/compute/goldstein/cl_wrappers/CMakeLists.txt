include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/utils)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/goldstein/cl_wrappers)
include_directories(${CMAKE_BINARY_DIR}/ext/clfft/src/src/include)
include_directories(${CMAKE_BINARY_DIR}/ext/clfft/build/include)

FILE(GLOB infiles "${despeckCL_SOURCE_DIR}/src/compute/goldstein/cl_wrappers/*.in")

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
            cl_wrappers.cpp
            patches_unpack.cpp
            patches_pack.cpp
            goldstein_patch_ft.cpp
            weighted_multiply.cpp
            raw_interferogram.cpp
            slc2real.cpp
           )

target_include_directories(goldstein_cl_wrappers PUBLIC ${CMAKE_BINARY_DIR}/ext/clfft/src/src/include)
target_include_directories(goldstein_cl_wrappers PUBLIC ${CMAKE_BINARY_DIR}/ext/clfft/build/include)
add_dependencies(goldstein_cl_wrappers clfft)

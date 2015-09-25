add_subdirectory(compute)
add_subdirectory(clcfg)
add_subdirectory(data)
add_subdirectory(utils)

include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)

add_library(despeckcl STATIC)
target_link_libraries(despeckcl boxcar_lib
                                nlinsar_lib
                                nlsar_lib)

INSTALL(TARGETS despeckcl
        DESTINATION "lib"
        )
INSTALL(FILES ../include/despeckcl.h
        DESTINATION "include"
        )

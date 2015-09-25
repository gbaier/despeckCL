add_subdirectory(compute)
add_subdirectory(clcfg)
add_subdirectory(data)
add_subdirectory(utils)

include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()

add_library(despeckcl STATIC 
            $<TARGET_OBJECTS:nlsar_lib>
            $<TARGET_OBJECTS:nlsar_cl_wrappers>
           )

target_link_libraries(despeckcl common_opencl_wrappers
                                insar_data
                                clcfg
                                logging
                                timings
                                ${OpenCL_LIBRARIES}
                                ${GSL_LIBRARIES})

INSTALL(TARGETS despeckcl
        DESTINATION "lib"
        )
INSTALL(FILES ../include/despeckcl.h
        DESTINATION "include"
        )

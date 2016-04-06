include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/include/utils)
include_directories(${despeckCL_SOURCE_DIR}/include/compute)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)

add_subdirectory(compute)
add_subdirectory(clcfg)
add_subdirectory(data)
add_subdirectory(utils)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()

add_library(despeckcl STATIC 
            $<TARGET_OBJECTS:insar_data>
            $<TARGET_OBJECTS:nlsar_lib>
            $<TARGET_OBJECTS:nlsar_training>
            $<TARGET_OBJECTS:nlsar_cl_wrappers>
            $<TARGET_OBJECTS:boxcar_lib>
            $<TARGET_OBJECTS:boxcar_opencl_wrapper>
            $<TARGET_OBJECTS:nlinsar_lib>
            $<TARGET_OBJECTS:nlinsar_opencl_wrapper>
            $<TARGET_OBJECTS:goldstein_lib>
            $<TARGET_OBJECTS:goldstein_cl_wrappers>
            ${despeckCL_SOURCE_DIR}/src/utils/logging.cpp
            ${despeckCL_SOURCE_DIR}/src/utils/checks.cpp
            ${despeckCL_SOURCE_DIR}/src/utils/timings.cpp
            ${despeckCL_SOURCE_DIR}/src/clcfg/clcfg.cpp
            $<TARGET_OBJECTS:common_opencl_wrappers>
           )

target_link_libraries(despeckcl ${OpenCL_LIBRARIES}
                                ${GSL_LIBRARIES}
                                ${CLFFT_LIBRARIES})

INSTALL(TARGETS despeckcl
        DESTINATION "lib"
        )

INSTALL(FILES ../include/despeckcl.h
        DESTINATION "include"
        )

INSTALL(FILES
        ../include/compute/nlsar/parameters.h
        ../include/compute/nlsar/stats.h
        DESTINATION "include/nlsar"
        )

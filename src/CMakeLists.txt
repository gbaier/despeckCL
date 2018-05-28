include_directories(${despeckCL_SOURCE_DIR}/src)
include_directories(${despeckCL_SOURCE_DIR}/src/data)
include_directories(${despeckCL_SOURCE_DIR}/src/clcfg)
include_directories(${despeckCL_SOURCE_DIR}/src/utils)
include_directories(${despeckCL_SOURCE_DIR}/src/compute)
include_directories(${despeckCL_SOURCE_DIR}/src/compute/common/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/compute)

add_subdirectory(compute)
add_subdirectory(clcfg)
add_subdirectory(data)
add_subdirectory(utils)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
endif()

# serialization of NL-SAR statistics use boost
find_package(Boost REQUIRED
    COMPONENTS serialization)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_library(despeckcl SHARED
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
            ${despeckCL_SOURCE_DIR}/src/utils/optimal_tiling.cpp
            ${despeckCL_SOURCE_DIR}/src/data/data.cpp
            ${despeckCL_SOURCE_DIR}/src/data/tile_iterator.cpp
            ${despeckCL_SOURCE_DIR}/src/data/sub_images.cpp
            #${despeckCL_SOURCE_DIR}/src/clcfg/clcfg.cpp
            $<TARGET_OBJECTS:common_opencl_wrappers>
           )

set(CLFFT_LIBRARIES "${CMAKE_BINARY_DIR}/ext/clfft/build/library/libclFFT.a")
add_dependencies(despeckcl clfft)
target_link_libraries(despeckcl ${OpenCL_LIBRARIES}
                                ${Boost_LIBRARIES}
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

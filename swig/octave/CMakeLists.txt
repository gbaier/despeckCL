#find_package(Octave REQUIRED)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/include/octave-3.8.2)

set_source_files_properties(boxcar_octave.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(boxcar_octave.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(boxcar_octave.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

swig_add_module(boxcar_octave octave boxcar_octave.i)
swig_link_libraries(boxcar_octave boxcar_lib)

set_target_properties(boxcar_octave PROPERTIES OUTPUT_NAME boxcar)

INSTALL(TARGETS boxcar_octave DESTINATION bindings)

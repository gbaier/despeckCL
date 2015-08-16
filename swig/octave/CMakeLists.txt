#find_package(Octave REQUIRED)

include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/include/octave-3.8.2)

set_source_files_properties(despeckcl_octave.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(despeckcl_octave.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(despeckcl_octave.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

swig_add_module(despeckcl_octave octave despeckcl_octave.i)
swig_link_libraries(despeckcl_octave boxcar_lib)

set_target_properties(despeckcl_octave PROPERTIES OUTPUT_NAME despeckcl)

INSTALL(TARGETS despeckcl_octave DESTINATION bindings)

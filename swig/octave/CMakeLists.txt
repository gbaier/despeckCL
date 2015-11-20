include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
string(REGEX REPLACE "/octave$" "" OCTAVE_INCLUDE_DIR_PURE ${OCTAVE_INCLUDE_DIRS})
include_directories(${OCTAVE_INCLUDE_DIR_PURE})

set_source_files_properties(despeckcl_octave.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(despeckcl_octave.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(despeckcl_octave.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

swig_add_module(despeckcl_octave octave despeckcl_octave.i)
swig_link_libraries(despeckcl_octave despeckcl)

set_source_files_properties(${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w")
set_target_properties(despeckcl_octave PROPERTIES OUTPUT_NAME despeckcl)

INSTALL(TARGETS despeckcl_octave DESTINATION ${CMAKE_INSTALL_PREFIX}/lib64/octave/${OCTAVE_VERSION_STRING}/site/oct/x86_64-redhat-linux-gnu)

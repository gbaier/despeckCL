include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)

find_package(PythonLibs REQUIRED)
if(PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${NUMPY_INCLUDE_DIR})
    message(STATUS "python include dir:" ${PYTHON_INCLUDE_DIRS})
    message(STATUS "python libraries dir:" ${PYTHON_LIBRARIES})
endif()

string(REGEX MATCH "^[0-9]+\\.[0-9]+" PYTHON_VERSION ${PYTHONLIBS_VERSION_STRING})

set_source_files_properties(boxcar_python.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(boxcar_python.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(boxcar_python.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

# setting SWIG_MODULE_NAME prevents the rebuilding of the wrapper every time
# make is run, even if no changes where made
set_property(SOURCE boxcar_python.i PROPERTY SWIG_MODULE_NAME boxcar)
swig_add_module(boxcar_python python boxcar_python.i)
swig_link_libraries(boxcar_python ${PYTHON_LIBRARIES} boxcar_lib)

set_target_properties(_boxcar_python PROPERTIES OUTPUT_NAME _boxcar)

INSTALL(TARGETS _boxcar_python DESTINATION bindings)
INSTALL(FILES ${CMAKE_BINARY_DIR}/swig/python/boxcar.py  DESTINATION bindings)


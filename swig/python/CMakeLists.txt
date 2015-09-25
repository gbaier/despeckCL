include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/boxcar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlinsar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrapper)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/common/cl_wrappers)
include_directories(${despeckCL_SOURCE_DIR}/include/clcfg)

find_package(PythonLibs REQUIRED)
if(PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${NUMPY_INCLUDE_DIR})
    message(STATUS "python include dir:" ${PYTHON_INCLUDE_DIRS})
    message(STATUS "python libraries dir:" ${PYTHON_LIBRARIES})
endif()

string(REGEX MATCH "^[0-9]+\\.[0-9]+" PYTHON_VERSION ${PYTHONLIBS_VERSION_STRING})

set_source_files_properties(despeckcl_python.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(despeckcl_python.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(despeckcl_python.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

# setting SWIG_MODULE_NAME prevents the rebuilding of the wrapper every time
# make is run, even if no changes where made
set_property(SOURCE despeckcl_python.i PROPERTY SWIG_MODULE_NAME despeckcl)
swig_add_module(despeckcl_python python despeckcl_python.i)
swig_link_libraries(despeckcl_python ${PYTHON_LIBRARIES}
                                     despeckcl)

set_target_properties(_despeckcl_python PROPERTIES OUTPUT_NAME _despeckcl)

INSTALL(TARGETS _despeckcl_python DESTINATION bindings)
INSTALL(FILES ${CMAKE_BINARY_DIR}/swig/python/despeckcl.py DESTINATION bindings)


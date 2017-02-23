include_directories(${despeckCL_SOURCE_DIR}/include)
include_directories(${despeckCL_SOURCE_DIR}/include/data)
include_directories(${despeckCL_SOURCE_DIR}/include/compute)
include_directories(${despeckCL_SOURCE_DIR}/include/compute/nlsar)

#set(PYTHON_LIBRARY /usr/lib64/libpython3.4m.so)
#set(PYTHON_INCLUDE_DIR /usr/include/python3.4m)

find_package(PythonInterp 3.4 REQUIRED)

find_package(PythonLibs REQUIRED)
if(PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

# get platform specific sites_packages and remove prefix
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(plat_specific=True, prefix=''))"
                OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# get numpy include dir
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy as np; print(np.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${NUMPY_INCLUDE_DIR})



set_source_files_properties(despeckcl_python.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(despeckcl_python.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(despeckcl_python.i PROPERTIES SWIG_FLAGS -DSKIP_INCLUDES)

# setting SWIG_MODULE_NAME prevents the rebuilding of the wrapper every time
# make is run, even if no changes where made
set_property(SOURCE despeckcl_python.i PROPERTY SWIG_MODULE_NAME despeckcl)
swig_add_module(despeckcl_python python despeckcl_python.i)
swig_link_libraries(despeckcl_python ${PYTHON_LIBRARIES} despeckcl)

set_source_files_properties(${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w")

set_target_properties(_despeckcl_python PROPERTIES OUTPUT_NAME _despeckcl)

INSTALL(TARGETS _despeckcl_python DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES})
INSTALL(FILES ${CMAKE_BINARY_DIR}/swig/python/despeckcl.py DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES})


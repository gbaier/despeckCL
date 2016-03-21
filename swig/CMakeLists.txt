find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

add_subdirectory(python)

#find_package(Octave)
#if(Octave_FOUND)
#    add_subdirectory(octave)
#endif()

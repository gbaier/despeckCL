find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

add_subdirectory(python)
add_subdirectory(octave)

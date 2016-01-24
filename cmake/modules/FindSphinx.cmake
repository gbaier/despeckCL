# taken from http://ericscottbarr.com/blog/2012/03/sphinx-and-cmake-beautiful-documentation-for-c-projects/
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build sphinx-build-3 sphinx_build-3.4
    HINTS
    $ENV{SPHINX_DIR}
    PATH_SUFFIXES bin
    DOC "Sphinx documentation generator"
)
 
include(FindPackageHandleStandardArgs)
 
find_package_handle_standard_args(Sphinx DEFAULT_MSG
    SPHINX_EXECUTABLE
)
 
mark_as_advanced(SPHINX_EXECUTABLE)

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
 
# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")
 
# HTML output directory. Put everything in docs easy hosting with github
set(SPHINX_HTML_DIR "${CMAKE_SOURCE_DIR}/docs")
 
add_custom_target(despeckCL_docs ALL
    ${SPHINX_EXECUTABLE}
        -q -b html
        -c "${CMAKE_CURRENT_SOURCE_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML documentation with Sphinx"
    DEPENDS _despeckcl_python)

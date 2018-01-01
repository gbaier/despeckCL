include(ExternalProject)
ExternalProject_Add(clfft
  GIT_REPOSITORY    https://github.com/clMathLibraries/clFFT
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/ext/clfft/src"
  SOURCE_SUBDIR     "src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/ext/clfft/build"
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  CMAKE_ARGS        -DBUILD_SHARED_LIBS:BOOL=OFF "-DCMAKE_CXX_FLAGS=-fPIC -w"
  CMAKE_CACHE_ARGS  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

include_directories(${despeckCL_SOURCE_DIR}/include/data)
add_library(insar_data OBJECT
            insar_data.cpp
            sub_images.cpp
            tile.cpp
            tile_iterator.cpp
            bbox.cpp)

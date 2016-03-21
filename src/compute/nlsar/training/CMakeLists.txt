add_library(nlsar_training OBJECT
            nlsar_training.cpp
            get_stats.cpp
            get_dissims.cpp
            patches.cpp
            combinations.cpp
           )

target_include_directories(nlsar_training PUBLIC
                           ${despeckCL_SOURCE_DIR}/include/compute/nlsar
                           ${despeckCL_SOURCE_DIR}/include/compute/nlsar/cl_wrappers
                           ${despeckCL_SOURCE_DIR}/include/compute/nlsar/routines
                           ${despeckCL_SOURCE_DIR}/include/compute/nlsar/training)

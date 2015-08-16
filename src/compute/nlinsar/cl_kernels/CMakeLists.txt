configure_file (
    "precompute_similarities_1st_pass.cl.in"
    "precompute_similarities_1st_pass.cl"
)
set(KERNEL_PATH kernel)
install(DIRECTORY . DESTINATION ${KERNEL_PATH} FILES_MATCHING PATTERN "*.cl")

#include "compute_pixel_similarities_2x2.h"

void nlsar::compute_pixel_similarities_2x2::run(cl::CommandQueue cmd_queue,
                                                cl::Buffer covmat,
                                                cl::Buffer pixel_similarities,
                                                const int height_overlap,
                                                const int width_overlap,
                                                const int dimension,
                                                const int nlooks,
                                                const int search_window_size)
/* computes all pixel similarities in the first and second quadrant
 * of the search window. The third and fourth aren't needed due to
 * their symmetry.
 *
 * The search window is divided up as follows into the first and second quadrant,
 * where x is the center pixel:
 *
 *
 * +-------+
 * |2221111|
 * |2221111|
 * |2221111|
 * |222x   |
 * |       |
 * |       |
 * |       |
 * +-------+
 */
{
    const int wsh = (search_window_size - 1)/2;

    kernel.setArg(0, covmat);
    kernel.setArg(1, pixel_similarities);
    kernel.setArg(2, height_overlap);
    kernel.setArg(3, width_overlap);
    kernel.setArg(4, dimension);
    kernel.setArg(5, nlooks);
    kernel.setArg(6, search_window_size);

    cl::NDRange global_size {(size_t) block_size*( (height_overlap - wsh - 1)/block_size + 1), \
                             (size_t) block_size*( (width_overlap  - wsh - 1)/block_size + 1)};
    cl::NDRange local_size  {block_size, block_size};

    // compute first quadrant of search window
    kernel.setArg(7,  0);                  // hh_start
    kernel.setArg(8,  wsh);                // hh_stop
    kernel.setArg(9,  wsh);                // ww_start
    kernel.setArg(10, search_window_size); // ww_stop
    cl::NDRange first_quadrant_offset  {0, 0};
    cmd_queue.enqueueNDRangeKernel(kernel, first_quadrant_offset, global_size, local_size, NULL, NULL);

    // compute second quadrant of search window
    kernel.setArg(7,  0);     // hh_start
    kernel.setArg(8,  wsh+1); // hh_stop
    kernel.setArg(9,  0);     // ww_start
    kernel.setArg(10, wsh);   // ww_stop
    cl::NDRange second_quadrant_offset  {0, wsh};
    cmd_queue.enqueueNDRangeKernel(kernel, second_quadrant_offset, global_size, local_size, NULL, NULL);
}

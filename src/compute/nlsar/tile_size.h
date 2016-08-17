/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TILE_SIZE_H
#define TILE_SIZE_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>

namespace nlsar {
    int round_down(const int num, const int multiple);

    int tile_size(cl::Context context,
                  const int search_window_size,
                  const std::vector<int>& patch_sizes,
                  const std::vector<int>& scale_sizes);

    class buffer_sizes {
      private:
        const int data_height;
        const int data_width;
        const int data_dimensions;
        const int search_window_size;
        std::vector<int> patch_sizes;
        std::vector<int> scale_sizes;


        size_t height_overlap(void) const;
        size_t width_overlap(void) const;

        size_t height_pixel_sim_symm(void) const;
        size_t width_pixel_sim_symm(void) const;

        size_t height_patch_sim_symm(void) const;
        size_t width_patch_sim_symm(void) const;

        size_t height_ori(void) const;
        size_t width_ori(void) const;

      public:
       buffer_sizes(int data_height,
                    int data_width,
                    int data_dimensions,
                    const int search_window_size,
                    const std::vector<int>& patch_sizes,
                    const std::vector<int>& scale_sizes)
           : data_height(data_height),
             data_width(data_width),
             data_dimensions(data_dimensions),
             search_window_size(search_window_size),
             patch_sizes(patch_sizes),
             scale_sizes(scale_sizes) {};

       size_t io_data(void) const;
       size_t io_data_all(void) const;

       size_t io_covmat(void) const;
       size_t io_covmat_all(void) const;
       size_t work_covmat(void) const;

       size_t weight_kernel_lut(void) const;
       size_t weight_kernel_lut_all(void) const;

       size_t best_idxs(void) const;

       size_t weights(void) const;
       size_t weights_all(void) const;

       size_t alphas(void) const;

       size_t pixel_similarities(void) const;
       size_t patch_similarities(void) const;

       size_t equivalent_number_of_looks(void) const;
       size_t intensities_nl(void) const;
       size_t variances_nl(void) const;
       size_t weight_sums(void) const;

       size_t all(void) const;
    };
}
#endif

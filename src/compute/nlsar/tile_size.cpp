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

#include "tile_size.h"

#include <algorithm>
#include <omp.h>
#include <cmath>

#include "logging.h"
#include "easylogging++.h"

int nlsar::round_down(const int num, const int multiple)
{
     int remainder = num % multiple;
     return num - remainder;
}

int nlsar::tile_size(cl::Context context,
                     const int search_window_size,
                     const std::vector<int>& patch_sizes,
                     const std::vector<int>& scale_sizes)
{
    const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());

    const int overlap = search_window_size + patch_size_max + scale_size_max - 3;

    const int n_params = patch_sizes.size() * scale_sizes.size();
    VLOG(0) << "number of parameters = " << n_params;

    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    cl::Device dev = devices[0];

    int long global_mem_size;
    dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
    VLOG(0) << "global memory size = " << global_mem_size;

    int long max_mem_alloc_size;
    dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
    VLOG(0) << "maximum memory allocation size = " << max_mem_alloc_size;

    const int n_threads = omp_get_max_threads();
    VLOG(0) << "number of threads = " << n_threads;

    // Most the most memory is required for storing weights, so only this is taken into account.
    // required bytes per pixel = reg_bpp
    const int req_bpp = 4 * search_window_size * search_window_size * n_params;
    const int n_pixels_global = global_mem_size    / (req_bpp * n_threads);
    const int n_pixels_alloc  = max_mem_alloc_size /  req_bpp;
    const int n_pixels = std::min(n_pixels_global, n_pixels_alloc);

    const int tile_size_fit        = std::sqrt(n_pixels);
    const int tile_size_fit_rounded = round_down(tile_size_fit, 64);

    VLOG(0) << "tile_size_fit = "         << tile_size_fit;
    VLOG(0) << "tile_size_fit_rounded = " << tile_size_fit_rounded;

    const float safety_factor = 0.90;
    int safe_tile_size = 0;
    if (tile_size_fit_rounded < safety_factor*tile_size_fit) {
        safe_tile_size = tile_size_fit_rounded;
    } else {
        safe_tile_size = tile_size_fit_rounded - 64;
    }
    VLOG(0) << "safe_sub_image_size = " << safe_tile_size;
    safe_tile_size += overlap;
    VLOG(0) << "safe_sub_image_size with overlap = " << safe_tile_size;


    return safe_tile_size;
}

size_t nlsar::buffer_sizes::width_overlap(void) const{
  const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());
  return data_width - scale_size_max + 1;
}

size_t nlsar::buffer_sizes::height_overlap(void) const{
  const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());
  return data_height - scale_size_max + 1;
}

size_t nlsar::buffer_sizes::height_pixel_sim_symm(void) const{
  const int wsh = (search_window_size - 1) / 2;
  return height_overlap() - wsh;
}

size_t nlsar::buffer_sizes::width_pixel_sim_symm(void) const{
  return width_overlap();
}

size_t nlsar::buffer_sizes::height_patch_sim_symm(void) const{
  const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
  return height_pixel_sim_symm() - patch_size_max + 1;
}

size_t nlsar::buffer_sizes::width_patch_sim_symm(void) const{
  const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
  return width_pixel_sim_symm() - patch_size_max + 1;
}

size_t nlsar::buffer_sizes::height_ori(void) const{
  const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
  return height_overlap() - patch_size_max - search_window_size + 2;
}

size_t nlsar::buffer_sizes::width_ori(void) const{
  const int patch_size_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
  return width_overlap() - patch_size_max - search_window_size + 2;
}

size_t nlsar::buffer_sizes::io_data(void) const{
  return data_height*data_width*sizeof(float);
}

size_t nlsar::buffer_sizes::io_data_all(void) const{
/* input: amplitude master, amplitude slave, interferometric phase
 * output: reflectivity, interferometric phase, coherence */
  return 6*io_data();
}

size_t nlsar::buffer_sizes::io_covmat(void) const{
/* dim x dim Matrix and real and imaginary part */
  return 2*data_dimensions*data_dimensions*io_data();
}

size_t nlsar::buffer_sizes::io_covmat_all(void) const{
  // input, rescaled, and output covariance matrices
  return 3*io_covmat();
}

size_t nlsar::buffer_sizes::work_covmat(void) const{
    const int scale_size_max = *std::max_element(scale_sizes.begin(), scale_sizes.end());
    return 2*data_dimensions*data_dimensions*(data_height-scale_size_max+1)*(data_width+scale_size_max+1)*sizeof(float);
}

// FIXME too small and insignificant
//size_t nlsar::buffer_sizes::weight_kernel_lut(void) const;
//size_t nlsar::buffer_sizes::weight_kernel_lut_all(void) const;

size_t nlsar::buffer_sizes::best_idxs(void) const{
  return height_ori()*width_ori()*sizeof(int);
}

size_t nlsar::buffer_sizes::weights(void) const{
  return search_window_size * search_window_size * height_ori() * width_ori() * sizeof(float);
}

size_t nlsar::buffer_sizes::weights_all(void) const{
  return patch_sizes.size()*scale_sizes.size()*weights();
}

size_t nlsar::buffer_sizes::alphas(void) const{
  return height_ori()*width_ori()*sizeof(float);
}

size_t nlsar::buffer_sizes::pixel_similarities(void) const{
  const int wsh = (search_window_size - 1)/2;
  return (wsh+search_window_size*wsh) * height_pixel_sim_symm() * width_pixel_sim_symm() * sizeof(float);
}

size_t nlsar::buffer_sizes::patch_similarities(void) const{
  const int wsh = (search_window_size - 1)/2;
  return (search_window_size*wsh + wsh) * height_patch_sim_symm() * width_patch_sim_symm() * sizeof(float);
}

size_t nlsar::buffer_sizes::equivalent_number_of_looks(void) const{
  return height_ori() * width_ori() * sizeof(float);
}

size_t nlsar::buffer_sizes::intensities_nl(void) const{
  return data_dimensions * height_ori() * width_ori() * sizeof(float);
}

size_t nlsar::buffer_sizes::variances_nl(void) const{
  return data_dimensions * height_ori() * width_ori() * sizeof(float);
}

size_t nlsar::buffer_sizes::weight_sums(void) const{
  return height_ori() * width_ori() * sizeof(float);
}

size_t nlsar::buffer_sizes::all(void) const{
  return io_data_all() + \
         io_covmat_all() + \
         work_covmat() + \
         best_idxs() + \
         pixel_similarities() + \
         patch_similarities() + \
         weights() + \
         weights_all() + \
         alphas() + \
         equivalent_number_of_looks() + \
         intensities_nl() + \
         variances_nl() + \
         weight_sums();
}

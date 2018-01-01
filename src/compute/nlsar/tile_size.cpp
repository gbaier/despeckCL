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
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "logging.h"
#include "easylogging++.h"
#include "optimal_tiling.h"

std::pair<int, int> nlsar::tile_size(cl::Context context,
                                     const int img_height,
                                     const int img_width,
                                     const int dimensions,
                                     const int search_window_size,
                                     const std::vector<int>& patch_sizes,
                                     const std::vector<int>& scale_sizes)
{
  const int patch_size_max =
      *std::max_element(patch_sizes.begin(), patch_sizes.end());
  const int scale_size_max =
      *std::max_element(scale_sizes.begin(), scale_sizes.end());

  // overlap consists of:
  // - (patch_size_max - 1)/2 + (search_window_size - 1)/2 for similarities
  // - (window_width - 1)/2 for spatial averaging of covariance matrices
  const int overlap = (patch_size_max - 1) / 2 + (search_window_size - 1) / 2 +
                      (scale_size_max - 1) / 2;
  VLOG(0) << "Getting device memory characteristics";

  std::vector<cl::Device> devices;
  context.getInfo(CL_CONTEXT_DEVICES, &devices);
  cl::Device dev = devices[0];

  size_t global_mem_size;
  dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
  VLOG(0) << "global memory size = " << global_mem_size;

  size_t max_mem_alloc_size;
  dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);
  VLOG(0) << "maximum memory allocation size = " << max_mem_alloc_size;


#ifdef _OPENMP
  const unsigned int n_threads = omp_get_max_threads();
#else
  const unsigned int n_threads = 1;
#endif
  VLOG(0) << "number of threads = " << n_threads;
  constexpr int step = 16;
  constexpr int nitems = 128;

  std::vector<int> dims;
  for(auto x : range<nitems>(step)) {
    dims.push_back(x);
  }

  auto pairs = all_pairs(dims);

  // pairs that fit into memory
  std::vector<std::pair<int, int>> pairs_fit;

  for(const auto p : pairs) {
    const int tile_height = p.first;
    const int tile_width = p.second;

    const buffer_sizes bs{tile_height,
                          tile_width,
                          dimensions,
                          search_window_size,
                          patch_sizes,
                          scale_sizes};

    size_t req_mem = bs.all();

    const float safety_factor = 0.8;

    if(req_mem > safety_factor*max_mem_alloc_size || req_mem > safety_factor*(global_mem_size/n_threads) ) {
      continue;
    } else {
      pairs_fit.push_back(p);
    }
  }

  // non wasteful pairs
  std::vector<std::pair<int, int>> nwp =
      retain_small_offcut_tiles(pairs_fit,
                                img_height,
                                img_width,
                                overlap,
                                1.2);

  nwp = sort_by_offcut(pairs_fit, img_height, img_width, overlap);

  // stable sort by scale factor so that tile sizes (a*b) and (b*a) result in
  // the one with the lower offcut to get chosen
  std::stable_sort(nwp.begin(), nwp.end(), [] (auto p1, auto p2) {return scale_factor(p1) > scale_factor(p2);});

  return nwp[0];
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

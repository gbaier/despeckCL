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

#include "patches.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <complex>
#include <iostream>
#include <cassert>

#include "sim_measures.h"
#include "tile_iterator.h"
#include "checks.h"

std::vector<covmat_data> nlsar::training::get_all_patches(const covmat_data& training_data, const int patch_size)
{
  std::vector<covmat_data> all_patches;
  for (auto t : tile_iterator(training_data.height(),
                              training_data.width(),
                              patch_size, patch_size, 0, 0)) {
            all_patches.push_back(tileget(training_data, t));
    }
    return all_patches;
}

float
nlsar::training::dissimilarity(const covmat_data& first,
                               const covmat_data& second)
{
  if (first.height() != second.height() || first.width() != second.width() || first.dim() != second.dim()) {
    std::cout << "heights: " << first.height() << ", " << second.height() << std::endl;
    std::cout << "width:   " << first.width() << ", " << second.width() << std::endl;
    std::cout << "dim:     " << first.dim() << ", " << second.dim() << std::endl;
    throw std::out_of_range("patch dimensions do not match");
  }
  float sum                    = 0.0f;
  const float* const firstptr  = first.data();
  const float* const secondptr = second.data();
  const int offset             = first.height() * first.width();

  for (int i = 0; i < offset; i++) {
      if (first.dim() == 2) {
        sum += dissimilarity_2x2(firstptr + i, secondptr + i, offset);
      }
      else if (first.dim() == 3) {
        sum += dissimilarity_3x3(firstptr + i, secondptr + i, offset);
      }
      else {
        throw std::runtime_error("currently only 2x2 or 3x3 covariance matrices are supported");
      }
  }
  return sum;
}

float nlsar::training::dissimilarity_2x2(const float * const first_pix, const float * const second_pix, const int offset) {
  const float el_00_p1     = *first_pix;
  const float el_01real_p1 = *(first_pix + 2 * offset);
  const float el_01imag_p1 = *(first_pix + 3 * offset);
  const float el_11_p1     = *(first_pix + 6 * offset);

  const float el_00_p2     = *second_pix;
  const float el_01real_p2 = *(second_pix + 2 * offset);
  const float el_01imag_p2 = *(second_pix + 3 * offset);
  const float el_11_p2     = *(second_pix + 6 * offset);

  const int nlooks = 1;

  return pixel_similarity_2x2(el_00_p1,
                              el_01real_p1,
                              el_01imag_p1,
                              el_11_p1,
                              el_00_p2,
                              el_01real_p2,
                              el_01imag_p2,
                              el_11_p2,
                              nlooks);
}

float nlsar::training::dissimilarity_3x3(const float * const first_pix, const float * const second_pix, const int offset) {
  const float p1_a_00      = *first_pix;
  const float p1_a_11      = *(first_pix +  8 * offset);
  const float p1_a_22      = *(first_pix + 16 * offset);
  const float p1_a_01_real = *(first_pix +  2 * offset);
  const float p1_a_01_imag = *(first_pix +  3 * offset);
  const float p1_a_02_real = *(first_pix +  4 * offset);
  const float p1_a_02_imag = *(first_pix +  5 * offset);
  const float p1_a_12_real = *(first_pix + 10 * offset);
  const float p1_a_12_imag = *(first_pix + 11 * offset);
  const float p2_a_00      = *second_pix;
  const float p2_a_11      = *(second_pix +  8 * offset);
  const float p2_a_22      = *(second_pix + 16 * offset);
  const float p2_a_01_real = *(second_pix +  2 * offset);
  const float p2_a_01_imag = *(second_pix +  3 * offset);
  const float p2_a_02_real = *(second_pix +  4 * offset);
  const float p2_a_02_imag = *(second_pix +  5 * offset);
  const float p2_a_12_real = *(second_pix + 10 * offset);
  const float p2_a_12_imag = *(second_pix + 11 * offset);

  const int nlooks = 1;

  pixel_similarity_3x3(p1_a_00,
                       p1_a_11,
                       p1_a_22,
                       p1_a_01_real,
                       p1_a_01_imag,
                       p1_a_02_real,
                       p1_a_02_imag,
                       p1_a_12_real,
                       p1_a_12_imag,
                       p2_a_00,
                       p2_a_11,
                       p2_a_22,
                       p2_a_01_real,
                       p2_a_01_imag,
                       p2_a_02_real,
                       p2_a_02_imag,
                       p2_a_12_real,
                       p2_a_12_imag,
                       nlooks);
}

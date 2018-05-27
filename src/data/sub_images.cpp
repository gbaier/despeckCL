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

#include "sub_images.h"

#include <memory>

std::unique_ptr<float[]>
get_sub_image(const float* image,
              const int height,
              const int width,
              const int dim,
              const int h_low,
              const int w_low,
              const int sub_img_height,
              const int sub_img_width)
{
    auto sub_image = std::make_unique<float[]>(dim*sub_img_height*sub_img_width);
    for (int d = 0; d < dim; d++) {
      for (int h = 0; h < sub_img_height; h++) {
        for (int w = 0; w < sub_img_width; w++) {
          const int h_img = std::min(height - 1, std::max(0, h + h_low));
          const int w_img = std::min(width - 1, std::max(0, w + w_low));
          sub_image[d * sub_img_height * sub_img_width + h * sub_img_width +
                    w]    = image[d * height * width + h_img * width + w_img];
        }
      }
    }
    return sub_image;
}

void write_sub_image(float * image,
                     const int height,
                     const int width,
                     const int dim,
                     float * sub_image,
                     const int h_low,
                     const int w_low,
                     const int sub_img_height,
                     const int sub_img_width,
                     const int overlap)
{
  for (int d = 0; d < dim; d++) {
    for (int h = overlap; h < sub_img_height - overlap; h++) {
      for (int w = overlap; w < sub_img_width - overlap; w++) {
        const int h_img = std::min(height - 1, std::max(0, h + h_low));
        const int w_img = std::min(width - 1, std::max(0, w + w_low));
        image[d * height * width + h_img * width + w_img] =
            sub_image[d * sub_img_height * sub_img_width + h * sub_img_width +
                      w];
      }
    }
  }
}

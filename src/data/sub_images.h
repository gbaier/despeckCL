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

#ifndef SUB_IMAGES_H
#define SUB_IMAGES_H

#include <algorithm>
#include <memory>

std::unique_ptr<float[]>
get_sub_image(const float* image,
              int height,
              int width,
              int dim,
              int h_low,
              int w_low,
              int sub_img_height,
              int sub_img_width);

void write_sub_image(float * image,
                     int height,
                     int width,
                     int dim,
                     float * sub_image,
                     int h_low,
                     int w_low,
                     int sub_img_height,
                     int sub_img_width,
                     int overlap);

#endif

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

#include "insar_data.h"
#include "sub_images.h"

#include <algorithm>

insar_data::insar_data(std::unique_ptr<float[]> data,
                       int height,
                       int width) 
            : _data(std::move(data)),
              height(height),
              width(width)
        {
        }

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * ref_filt,
                       float * phi_filt,
                       float * coh_filt,
                       int height,
                       int width) : height(height), width(width)
{
    const size_t size = height*width;

    this->_data = std::make_unique<float[]>(dim*size);

    std::copy(a1, a1+size, this->ampl_master());
    std::copy(a2, a2+size, this->ampl_slave());
    std::copy(dp, dp+size, this->phase());
    std::copy(ref_filt, ref_filt+size, this->ref_filt());
    std::copy(phi_filt, phi_filt+size, this->phase_filt());
    std::copy(coh_filt, coh_filt+size, this->coh_filt());
}

insar_data::insar_data(insar_data &&other) noexcept
{
    std::swap(height, other.height);
    std::swap(width, other.width);
    std::swap(_data, other._data);
}

insar_data& insar_data::operator=(insar_data &&other) noexcept {
    std::swap(height, other.height);
    std::swap(width, other.width);
    std::swap(_data, other._data);
    return *this;
}

insar_data tileget(const insar_data& img_data, tile<2> sub) {
  auto data_sub = get_sub_image(img_data.data(),
                                img_data.height,
                                img_data.width,
                                img_data.dim,
                                sub[0].start,
                                sub[1].start,
                                sub[0].stop - sub[0].start,
                                sub[1].stop - sub[1].start);
  return insar_data{std::move(data_sub),
                    sub[0].stop - sub[0].start,
                    sub[1].stop - sub[1].start};
}

// copy img_tile to img_data defined by sub
// akin to memcpy
void tilecpy(insar_data& img_data, const insar_data& img_tile, tile<2> sub) {
  write_sub_image(img_data.data(),
                  img_data.height,
                  img_data.width,
                  img_data.dim,
                  img_tile.data(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
}

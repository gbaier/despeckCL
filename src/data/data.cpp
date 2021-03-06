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

#include "data.h"
#include <cmath>

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * ref_filt,
                       float * phi_filt,
                       float * coh_filt,
                       int height,
                       int width) : _cont(std::make_unique<float[]>(6 * (size_t)height*width), height, width, 6)
{
    const size_t single_img_size = height*width;

    std::copy(a1, a1+single_img_size, this->ampl_master());
    std::copy(a2, a2+single_img_size, this->ampl_slave());
    std::copy(dp, dp+single_img_size, this->phase());
    std::copy(ref_filt, ref_filt+single_img_size, this->ref_filt());
    std::copy(phi_filt, phi_filt+single_img_size, this->phase_filt());
    std::copy(coh_filt, coh_filt+single_img_size, this->coh_filt());
}

ampl_data::ampl_data(float* ampl, float* ref_filt, int height, int width)
    : _cont(std::make_unique<float[]>(2 * (size_t)height * width), height, width, 2)
{
    const size_t single_img_size = (size_t)height*width;

    std::copy(ampl, ampl+single_img_size, this->ampl());
    std::copy(ref_filt, ref_filt+single_img_size, this->ref_filt());
}


covmat_data::covmat_data(float* covmat_raw, float* covmat_filt, int height, int width, int dim)
    : _cont(std::make_unique<float[]>(2 * 2 * dim * dim * (size_t)height * width), height, width, 2*2*dim*dim), _dim(dim)
{
  const size_t single_img_size = (size_t)height * width * 2 * dim * dim;

  std::copy(covmat_raw, covmat_raw + single_img_size, this->covmat_raw());
  std::copy(covmat_filt, covmat_filt + single_img_size, this->covmat_filt());
}

covmat_data::covmat_data(insar_data data)
    : _cont(std::make_unique<float[]>(2 * 2 * data.dim() * data.dim() *
                                      (size_t)data.height() * data.width()),
            data.height(),
            data.width(),
            2 * 2 * data.dim() * data.dim()),
      _dim(data.dim())
{
    const size_t single_img_size = (size_t)height()*width();
    // diagonal elements
    std::transform(data.ampl_master(), data.ampl_master() + single_img_size, _cont.data(), [] (float a) {return a*a;});
    std::fill(_cont.data() + single_img_size, _cont.data() + 2*single_img_size, 0);

    std::transform(data.ampl_slave(), data.ampl_slave() + single_img_size, _cont.data() + 6*single_img_size, [] (float a) {return a*a;});
    std::fill(_cont.data() + 7 * single_img_size, _cont.data() + 8*single_img_size, 0);

    // off diagonal elements
    float * upper_real = _cont.data() + 2*single_img_size;
    float * upper_imag = _cont.data() + 3*single_img_size;
    float * lower_real = _cont.data() + 4*single_img_size;
    float * lower_imag = _cont.data() + 5*single_img_size;
    for(size_t i = 0; i<single_img_size; i++) {
        const float real = data.ampl_master()[i] * data.ampl_slave()[i] * std::cos(data.phase()[i]);
        const float imag = data.ampl_master()[i] * data.ampl_slave()[i] * std::sin(data.phase()[i]);
        *upper_real++ = real;
        *upper_imag++ = imag;
        *lower_real++ = real;
        *lower_imag++ = -imag;
    }
}

covmat_data
tileget(const covmat_data& img_data, tile<2> sub)
{
  auto data_sub = get_sub_image(img_data.data(),
                                img_data.height(),
                                img_data.width(),
                                img_data.size(),
                                sub[0].start,
                                sub[1].start,
                                sub[0].stop - sub[0].start,
                                sub[1].stop - sub[1].start);
  return covmat_data{std::move(data_sub),
                     sub[0].stop - sub[0].start,
                     sub[1].stop - sub[1].start,
                     img_data.dim()};
}

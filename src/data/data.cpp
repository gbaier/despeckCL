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

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * ref_filt,
                       float * phi_filt,
                       float * coh_filt,
                       int height,
                       int width) : _cont(std::make_unique<float[]>(6*height*width), height, width)
{
    const size_t size = height*width;

    std::copy(a1, a1+size, this->ampl_master());
    std::copy(a2, a2+size, this->ampl_slave());
    std::copy(dp, dp+size, this->phase());
    std::copy(ref_filt, ref_filt+size, this->ref_filt());
    std::copy(phi_filt, phi_filt+size, this->phase_filt());
    std::copy(coh_filt, coh_filt+size, this->coh_filt());
}

ampl_data::ampl_data(float* ampl, float* ref_filt, int height, int width)
    : _cont(std::make_unique<float[]>(2 * height * width), height, width)
{
    const size_t size = height*width;

    std::copy(ampl, ampl+size, this->ampl());
    std::copy(ref_filt, ref_filt+size, this->ref_filt());
}

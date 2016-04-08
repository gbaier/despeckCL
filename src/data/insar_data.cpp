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

#include <string.h> // for memset
#include <iostream>
#include <algorithm>

insar_data_shared::insar_data_shared(float * a1,
                                     float * a2,
                                     float * dp,
                                     float * amp_filt,
                                     float * phi_filt,
                                     float * coh_filt,
                                     const int height,
                                     const int width) : a1(a1),
                                                        a2(a2),
                                                        dp(dp),
                                                        amp_filt(amp_filt),
                                                        phi_filt(phi_filt),
                                                        coh_filt(coh_filt),
                                                        height(height),
                                                        width(width) {}

insar_data_shared::insar_data_shared(const insar_data_shared &data) : a1(data.a1),
                                                                      a2(data.a2),
                                                                      dp(data.dp),
                                                                      amp_filt(data.amp_filt),
                                                                      phi_filt(data.phi_filt),
                                                                      coh_filt(data.coh_filt),
                                                                      height(data.height),
                                                                      width(data.width) {}

insar_data_shared::~insar_data_shared() {}

insar_data_shared& insar_data_shared::operator=(const insar_data_shared &data)
{
    const size_t bytesize = height*width*sizeof(float);

    memcpy(a1, data.a1, bytesize);
    memcpy(a2, data.a2, bytesize);
    memcpy(dp, data.dp, bytesize);

    memcpy(amp_filt, data.amp_filt, bytesize);
    memcpy(phi_filt, data.phi_filt, bytesize);
    memcpy(coh_filt, data.coh_filt, bytesize);

    return *this;
}

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * amp_filt,
                       float * phi_filt,
                       float * coh_filt,
                       const int height,
                       const int width) : insar_data_shared(a1, a2, dp, amp_filt, phi_filt, coh_filt, height, width)
{
    const size_t bytesize = height*width*sizeof(float);

    this->a1 = (float *) malloc(bytesize);
    memcpy(this->a1, a1, bytesize);

    this->a2 = (float *) malloc(bytesize);
    memcpy(this->a2, a2, bytesize);

    this->dp = (float *) malloc(bytesize);
    memcpy(this->dp, dp, bytesize);

    this->amp_filt = (float *) malloc(bytesize);
    memcpy(this->amp_filt, amp_filt, bytesize);

    this->phi_filt = (float *) malloc(bytesize);
    memcpy(this->phi_filt, phi_filt, bytesize);

    this->coh_filt = (float *) malloc(bytesize);
    memcpy(this->coh_filt, coh_filt, bytesize);
}

insar_data::insar_data(const insar_data &data) : insar_data_shared(data)
{
    copy(data);
}
insar_data::insar_data(const insar_data_shared &data) : insar_data_shared(data)
{
    copy(data);
}



insar_data::~insar_data()
{
    free(a1);
    free(a2);
    free(dp);
    free(amp_filt);
    free(phi_filt);
    free(coh_filt);
}


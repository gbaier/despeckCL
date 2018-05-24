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

#include <string.h> // for memset
#include <iostream>
#include <algorithm>

insar_data_shared::insar_data_shared(float * a1,
                                     float * a2,
                                     float * dp,
                                     float * ref_filt,
                                     float * phi_filt,
                                     float * coh_filt,
                                     const int height,
                                     const int width) : a1(a1),
                                                        a2(a2),
                                                        dp(dp),
                                                        ref_filt(ref_filt),
                                                        phi_filt(phi_filt),
                                                        coh_filt(coh_filt),
                                                        height(height),
                                                        width(width) {}

insar_data_shared::insar_data_shared(const insar_data_shared &data) : a1(data.a1),
                                                                      a2(data.a2),
                                                                      dp(data.dp),
                                                                      ref_filt(data.ref_filt),
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

    memcpy(ref_filt, data.ref_filt, bytesize);
    memcpy(phi_filt, data.phi_filt, bytesize);
    memcpy(coh_filt, data.coh_filt, bytesize);

    return *this;
}

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * ref_filt,
                       float * phi_filt,
                       float * coh_filt,
                       const int height,
                       const int width) : insar_data_shared(a1, a2, dp, ref_filt, phi_filt, coh_filt, height, width)
{
    const size_t bytesize = height*width*sizeof(float);

    this->a1 = (float *) malloc(bytesize);
    memcpy(this->a1, a1, bytesize);

    this->a2 = (float *) malloc(bytesize);
    memcpy(this->a2, a2, bytesize);

    this->dp = (float *) malloc(bytesize);
    memcpy(this->dp, dp, bytesize);

    this->ref_filt = (float *) malloc(bytesize);
    memcpy(this->ref_filt, ref_filt, bytesize);

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
    free(ref_filt);
    free(phi_filt);
    free(coh_filt);
}


insar_data tileget(const insar_data& img_data, tile<2> sub) {
    float * const a1_sub       = get_sub_image(img_data.a1,       img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    float * const a2_sub       = get_sub_image(img_data.a2,       img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    float * const dp_sub       = get_sub_image(img_data.dp,       img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    float * const ref_filt_sub = get_sub_image(img_data.ref_filt, img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    float * const phi_filt_sub = get_sub_image(img_data.phi_filt, img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    float * const coh_filt_sub = get_sub_image(img_data.coh_filt, img_data.height, img_data.width, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start);
    insar_data sub_image{a1_sub, a2_sub, dp_sub, ref_filt_sub, phi_filt_sub, coh_filt_sub, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start};
    free(a1_sub);
    free(a2_sub);
    free(dp_sub);
    free(ref_filt_sub);
    free(phi_filt_sub);
    free(coh_filt_sub);
    return sub_image;
}

// copy img_tile to img_data defined by sub
// akin to memcpy
void tilecpy(insar_data& img_data, const insar_data& img_tile, tile<2> sub) {
    write_sub_image(img_data.ref_filt, img_data.height, img_data.width, img_tile.ref_filt, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start, 0);
    write_sub_image(img_data.phi_filt, img_data.height, img_data.width, img_tile.phi_filt, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start, 0);
    write_sub_image(img_data.coh_filt, img_data.height, img_data.width, img_tile.coh_filt, sub[0].start, sub[1].start, sub[0].stop-sub[0].start, sub[1].stop-sub[1].start, 0);
}

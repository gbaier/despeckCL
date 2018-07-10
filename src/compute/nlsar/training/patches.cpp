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
#include "checks.h"

nlsar::training::data::data(const float * covmats,
                            const uint32_t height,
                            const uint32_t width,
                            const uint32_t dimension) : height(height),
                                                        width(width),
                                                        dimension(dimension),
                                                        covmats((float*) malloc(2*height*width*dimension*dimension*sizeof(float)))
{
    memcpy(this->covmats, covmats, 2*height*width*dimension*dimension*sizeof(float));
}

nlsar::training::data::data(const data& other) : height(other.get_height()),
                                                 width(other.get_width()),
                                                 dimension(other.get_dimension()),
                                                 covmats((float *) malloc(2*height*width*dimension*dimension*sizeof(float)))
{
    memcpy(this->covmats, other.get_covmats(), 2*height*width*dimension*dimension*sizeof(float));
}

nlsar::training::data::~data()
{
    free(covmats);
}

uint32_t nlsar::training::data::get_height(void) const
{
    return height;
}

uint32_t nlsar::training::data::get_width(void) const
{
    return width;
}

uint32_t nlsar::training::data::get_dimension(void) const
{
    return dimension;
}

float * nlsar::training::data::get_covmats(void) const
{
    return covmats;
}

nlsar::training::data nlsar::training::get_patch(const data& training_data,
                                                 const uint32_t upper_h,
                                                 const uint32_t left_w,
                                                 const uint32_t patch_size)
{
    assert(odd(patch_size) && (patch_size > 0));
    if (upper_h + patch_size > training_data.get_height() || left_w + patch_size > training_data.get_width()) {
        throw std::out_of_range("patch does not lie inside data");
    }

    float* temp = (float*)malloc(2 * patch_size * patch_size *
                                 training_data.get_dimension() *
                                 training_data.get_dimension() * sizeof(float));
    for(uint32_t d=0; d<2*training_data.get_dimension()*training_data.get_dimension(); d++) {
        for(uint32_t h=0; h<patch_size; h++) {
            for(uint32_t w=0; w<patch_size; w++) {
                const uint32_t odx = d*patch_size*patch_size + h*patch_size + w;
                const uint32_t idx =
                    d * training_data.get_height() * training_data.get_width() +
                    (upper_h + h) * training_data.get_width() + left_w + w;
                temp[odx] = training_data.get_covmats()[idx];
            }
        }
    }
    data patch(temp, patch_size, patch_size, training_data.get_dimension());
    free(temp);
    return patch;
}

std::vector<nlsar::training::data> nlsar::training::get_all_patches(const data& training_data, const uint32_t patch_size)
{
    std::vector<data> all_patches;
    for(uint32_t h=0; h<training_data.get_height()-patch_size; h++) {
        for(uint32_t w=0; w<training_data.get_width()-patch_size; w++) {
            all_patches.push_back(get_patch(training_data, h, w, patch_size));
        }
    }
    return all_patches;
}

float nlsar::training::dissimilarity(const data& first, const data& second)
{
    if (first.get_height() != second.get_height() || first.get_width() != second.get_width()) {
        std::cout << first.get_height() << ", " << second.get_height() << std::endl;
        std::cout << first.get_width()  << ", " << second.get_width()  << std::endl;
        throw std::out_of_range("patch sizes do not match");
    }
    float sum = 0.0f;
    const float * const firstptr  = first.get_covmats();
    const float * const secondptr = second.get_covmats();
    const size_t offset = first.get_height()*first.get_width();

    for(uint32_t i=0; i<offset; i++) {
        const float el_00_p1     = firstptr[i];
        const float el_01real_p1 = firstptr[i + 2*offset];
        const float el_01imag_p1 = firstptr[i + 3*offset];
        const float el_11_p1     = firstptr[i + 6*offset];

        const float el_00_p2     = secondptr[i];
        const float el_01real_p2 = secondptr[i + 2*offset];
        const float el_01imag_p2 = secondptr[i + 3*offset];
        const float el_11_p2     = secondptr[i + 6*offset];

        const int nlooks = 1;

        sum += pixel_similarity_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1, \
                                    el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2, \
                                    nlooks);
    }
    return sum;
}

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

nlsar::training::data nlsar::training::data::get_patch(const uint32_t upper_h,
                                                       const uint32_t left_w,
                                                       const uint32_t patch_size)
{
    assert(odd(patch_size) && (patch_size > 0));
    if (upper_h + patch_size > height || left_w + patch_size > width) {
        throw std::out_of_range("patch does not lie inside data");
    }

    float * temp = (float*) malloc(2*patch_size*patch_size*dimension*dimension*sizeof(float));
    for(uint32_t d=0; d<2*dimension*dimension; d++) {
        for(uint32_t h=0; h<patch_size; h++) {
            for(uint32_t w=0; w<patch_size; w++) {
                const uint32_t odx = d*patch_size*patch_size + h*patch_size + w;
                const uint32_t idx = d*height*width + (upper_h+h)*width + left_w+w;
                temp[odx] = covmats[idx];
            }
        }
    }
    data patch(temp, patch_size, patch_size, dimension);
    free(temp);
    return patch;
}

std::vector<nlsar::training::data> nlsar::training::data::get_all_patches(const uint32_t patch_size)
{
    std::vector<data> all_patches;
    for(uint32_t h=0; h<height-patch_size; h++) {
        for(uint32_t w=0; w<width-patch_size; w++) {
            all_patches.push_back(get_patch(h, w, patch_size));
        }
    }
    return all_patches;
}

float nlsar::training::data::dissimilarity(const data& other)
{
    if (height != other.get_height() || width != other.get_width()) {
        std::cout << height << ", " << other.get_height() << std::endl;
        std::cout << width << ", " << other.get_width() << std::endl;
        throw std::out_of_range("patch sizes do not match");
    }
    float sum = 0.0f;
    const float * const otherptr = other.get_covmats();
    for(uint32_t i=0; i<height*width; i++) {
        const float el_00_p1     = covmats[i];
        const float el_01real_p1 = covmats[i + 2*height*width];
        const float el_01imag_p1 = covmats[i + 3*height*width];
        const float el_11_p1     = covmats[i + 6*height*width];

        const float el_00_p2     = otherptr[i];
        const float el_01real_p2 = otherptr[i + 2*height*width];
        const float el_01imag_p2 = otherptr[i + 3*height*width];
        const float el_11_p2     = otherptr[i + 6*height*width];

        const int nlooks = 1;

        sum += pixel_similarity_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1, \
                                    el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2, \
                                    nlooks);
    }
    return sum;
}

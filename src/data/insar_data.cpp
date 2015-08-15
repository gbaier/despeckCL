#include "insar_data.h"
#include "sub_images.h"

#include <string.h> // for memset
#include <iostream>
#include <algorithm>

void pad_2d_array(float ** array, const int height, const int width, const int overlap)
{
    const int height_new = height + 2*overlap;
    const int width_new  = width  + 2*overlap;

    float * old_array = *array;
    float * new_array = (float *) malloc( height_new*width_new*sizeof(float) );

    for(int h = 0; h<height_new; h++) {
        for(int w = 0; w<width_new; w++) {
            const int rel_h = std::min( std::max(h - overlap, 0), height - 1);
            const int rel_w = std::min( std::max(w - overlap, 0), width  - 1);
            new_array[h*width_new + w] = old_array[rel_h*width + rel_w];
        }
    }
    free(old_array);
    *array = new_array;
}

void unpad_2d_array(float ** array, const int height, const int width, const int overlap)
{
    const int height_new = height - 2*overlap;
    const int width_new  = width  - 2*overlap;

    float * old_array = *array;
    float * new_array = (float *) malloc( height_new*width_new*sizeof(float) );

    for(int h = 0; h<height_new; h++) {
        for(int w = 0; w<width_new; w++) {
            const int rel_h = h + overlap;
            const int rel_w = w + overlap;
            new_array[h*width_new + w] = old_array[rel_h*width+rel_w];
        }
    }
    free(old_array);
    *array = new_array;
}

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * amp_filt,
                       float * phi_filt,
                       float * coh_filt,
                       const int height,
                       const int width) : height(height), width(width)
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

insar_data::insar_data(const insar_data &data) : height(data.height), width(data.width)
{
    const size_t bytesize = height*width*sizeof(float);

    a1 = (float *) malloc(bytesize);
    memcpy(a1, data.a1, bytesize);

    a2 = (float *) malloc(bytesize);
    memcpy(a2, data.a2, bytesize);

    dp = (float *) malloc(bytesize);
    memcpy(dp, data.dp, bytesize);

    amp_filt = (float *) malloc(bytesize);
    memcpy(amp_filt, data.amp_filt, bytesize);

    phi_filt = (float *) malloc(bytesize);
    memcpy(phi_filt, data.phi_filt, bytesize);

    coh_filt = (float *) malloc(bytesize);
    memcpy(coh_filt, data.coh_filt, bytesize);
}

insar_data& insar_data::operator=(const insar_data &data)
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

insar_data::~insar_data()
{
    free(a1);
    free(a2);
    free(dp);
    free(amp_filt);
    free(phi_filt);
    free(coh_filt);
}

insar_data insar_data::get_sub_insar_data(bbox boundaries)
{
    float * const a1_sub       = get_sub_image(a1,       height, width, boundaries);
    float * const a2_sub       = get_sub_image(a2,       height, width, boundaries);
    float * const dp_sub       = get_sub_image(dp,       height, width, boundaries);
    float * const amp_filt_sub = get_sub_image(amp_filt, height, width, boundaries);
    float * const phi_filt_sub = get_sub_image(phi_filt, height, width, boundaries);
    float * const coh_filt_sub = get_sub_image(coh_filt, height, width, boundaries);
    insar_data sub_image{a1_sub, a2_sub, dp_sub, amp_filt_sub, phi_filt_sub, coh_filt_sub, boundaries.h_up - boundaries.h_low, boundaries.w_up - boundaries.w_low};
    free(a1_sub);
    free(a2_sub);
    free(dp_sub);
    free(amp_filt_sub);
    free(phi_filt_sub);
    free(coh_filt_sub);
    return sub_image;
}

void insar_data::write_sub_insar_data(insar_data& sub_insar_data,
                                      const int overlap,
                                      bbox boundaries)
{
    write_sub_image(amp_filt, sub_insar_data.amp_filt, overlap, height, width, boundaries);
    write_sub_image(phi_filt, sub_insar_data.phi_filt, overlap, height, width, boundaries);
    write_sub_image(coh_filt, sub_insar_data.coh_filt, overlap, height, width, boundaries);
}

void insar_data::pad(const int overlap)
{
    pad_2d_array(&a1, height, width, overlap);
    pad_2d_array(&a2, height, width, overlap);
    pad_2d_array(&dp, height, width, overlap);
    pad_2d_array(&amp_filt, height, width, overlap);
    pad_2d_array(&phi_filt, height, width, overlap);
    pad_2d_array(&coh_filt, height, width, overlap);
    height = height + 2*overlap;
    width = width + 2*overlap;
}

void insar_data::unpad(const int overlap)
{
    unpad_2d_array(&a1, height, width, overlap);
    unpad_2d_array(&a2, height, width, overlap);
    unpad_2d_array(&dp, height, width, overlap);
    unpad_2d_array(&amp_filt, height, width, overlap);
    unpad_2d_array(&phi_filt, height, width, overlap);
    unpad_2d_array(&coh_filt, height, width, overlap);
    height = height - 2*overlap;
    width = width - 2*overlap;
}

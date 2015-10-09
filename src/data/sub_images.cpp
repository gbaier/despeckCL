#include "sub_images.h"
#include <iostream>

#include <stdlib.h>


float* get_sub_image(const float * image,
                     const int height,
                     const int width,
                     bbox boundaries)
{
    if (!boundaries.valid(height, width)) {
        throw std::logic_error("bounding box dimensions too big for image");
    }
    const int h_low = boundaries.h_low;
    const int h_up = boundaries.h_up;
    const int w_low = boundaries.w_low;
    const int w_up = boundaries.w_up;
    float* sub_image = (float *) malloc((h_up-h_low)*(w_up-w_low)*sizeof(float));
    const int siw = w_up - w_low;
    for(int h = h_low; h < h_up; h++) {
        for(int w = w_low; w < w_up; w++) {
            const int h_rel = h-h_low;
            const int w_rel = w-w_low;
            sub_image[h_rel * siw + w_rel] = image[h*width + w];
        }
    }
    return sub_image;
}

void write_sub_image(float * image,
                     float * sub_image,
                     const int overlap,
                     const int height,
                     const int width,
                     bbox boundaries)
{
    if (!boundaries.valid(height, width)) {
        throw std::logic_error("bounding box dimensions too big for image");
    }
    const int h_low = boundaries.h_low;
    const int h_up = boundaries.h_up;
    const int w_low = boundaries.w_low;
    const int w_up = boundaries.w_up;
    const int siw = w_up - w_low;
    for(int h = h_low+overlap; h < h_up-overlap; h++) {
        for(int w = w_low+overlap; w < w_up-overlap; w++) {
            const int h_rel = h-h_low;
            const int w_rel = w-w_low;
            image[h*width + w] = sub_image[h_rel * siw + w_rel];
        }
    }
}

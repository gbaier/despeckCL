#ifndef SUB_IMAGES_H
#define SUB_IMAGES_H

#include <tuple>
#include <algorithm>

float* get_sub_image(const float * image,
                     const int height,
                     const int width,
                     const int h_low,
                     const int w_low,
                     const int sub_img_size);

void write_sub_image(float * image,
                     const int height,
                     const int width,
                     float * sub_image,
                     const int h_low,
                     const int w_low,
                     const int sub_img_size,
                     const int overlap);

#endif

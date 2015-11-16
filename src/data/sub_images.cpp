#include "sub_images.h"

#include <stdlib.h>

float* get_sub_image(const float * image,
                     const int height,
                     const int width,
                     const int h_low,
                     const int w_low,
                     const int sub_img_size)
{
    float* sub_image = (float *) malloc(sub_img_size*sub_img_size*sizeof(float));
    for(int h = 0; h < sub_img_size; h++) {
        for(int w = 0; w < sub_img_size; w++) {
            const int h_img = std::min(height-1, std::max(0, h+h_low));
            const int w_img = std::min(width-1,  std::max(0, w+w_low));
            sub_image[h * sub_img_size + w] = image[h_img*width + w_img];
        }
    }
    return sub_image;
}

void write_sub_image(float * image,
                     const int height,
                     const int width,
                     float * sub_image,
                     const int h_low,
                     const int w_low,
                     const int sub_img_size,
                     const int overlap)
{
    for(int h = overlap; h < sub_img_size-overlap; h++) {
        for(int w = overlap; w < sub_img_size-overlap; w++) {
            const int h_img = std::min(height-1, std::max(0, h+h_low));
            const int w_img = std::min(width-1,  std::max(0, w+w_low));
            image[h_img*width + w_img] = sub_image[h*sub_img_size + w];
        }
    }
}

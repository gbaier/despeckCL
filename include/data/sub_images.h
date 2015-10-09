#ifndef SUB_IMAGES_H
#define SUB_IMAGES_H

#include <tuple>
#include <algorithm>

#include "bbox.h"

float* get_sub_image(const float * image,
                     const int height,
                     const int width,
                     bbox boundaries);

void write_sub_image(float * image,
                     float * sub_image,
                     const int overlap,
                     const int height,
                     const int width,
                     bbox boundaries);

class gen_sub_images
{
    protected:
        const int height;
        const int width;

        const int sub_image_size;
        const int overlap;

        const int step_size;

        bbox boundaries;


    public:
        gen_sub_images( const int height,
                        const int width,
                        const int sub_image_size,
                        const int overlap) : height(height),
                                             width(width),
                                             sub_image_size(sub_image_size),
                                             overlap(overlap),
                                             step_size(sub_image_size - 2*overlap),
                                             boundaries{0, 0, std::min(sub_image_size, height), std::min(sub_image_size, width)} {}

        // Iterator functions
        bool operator!=(const gen_sub_images&) const
        {
            return !((boundaries.w_up - boundaries.w_low) <= 2*overlap && boundaries.w_up == width);
        }

        void operator++()
        {
            boundaries.h_low += step_size;
            boundaries.h_up = std::min(boundaries.h_up + step_size, height);
            if ((boundaries.h_up - boundaries.h_low) <= 2*overlap && boundaries.h_up == height) {
                boundaries.h_low = 0;
                boundaries.h_up = std::min(sub_image_size, height);
                boundaries.w_low += step_size;
                boundaries.w_up = std::min(boundaries.w_up + step_size, width);
            }
        }

        bbox operator*() const
        {
            return boundaries;
        }

        const gen_sub_images& begin() const { return *this; }
        const gen_sub_images& end() const { return *this; }
};

#endif

#ifndef INSAR_DATA_H
#define INSAR_DATA_H

#include "sub_images.h"

void pad_2d_array(float ** array, const int height, const int width, const int overlap);
void unpad_2d_array(float ** array, const int height, const int width, const int overlap);

class insar_data
{
    public:
        float * a1;
        float * a2;
        float * dp;
        float * amp_filt;
        float * phi_filt;
        float * coh_filt;
        int height;
        int width;

        insar_data(float * a1,
                   float * a2,
                   float * dp,
                   float * amp_filt,
                   float * phi_filt,
                   float * coh_filt,
                   const int height,
                   const int width);

        insar_data(const insar_data &data);

        insar_data& operator=(const insar_data &data);

        ~insar_data();

        insar_data get_sub_insar_data(bbox boundaries);

        void write_sub_insar_data(insar_data& sub_insar_data,
                                  const int overlap,
                                  bbox boundaries);

        void pad(const int overlap);
        void unpad(const int overlap);
};
#endif

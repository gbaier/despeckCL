#ifndef INSAR_DATA_H
#define INSAR_DATA_H

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
};
#endif

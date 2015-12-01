#ifndef INSAR_DATA_H
#define INSAR_DATA_H

#include <stdlib.h>
#include <string.h> // for memset, memcpy

class insar_data_shared
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

        insar_data_shared(float * a1,
                          float * a2,
                          float * dp,
                          float * amp_filt,
                          float * phi_filt,
                          float * coh_filt,
                          const int height,
                          const int width);

        insar_data_shared(const insar_data_shared &data);

        ~insar_data_shared();

        insar_data_shared& operator=(const insar_data_shared &data);
};

class insar_data : public insar_data_shared
{
    public:
        insar_data(float * a1,
                   float * a2,
                   float * dp,
                   float * amp_filt,
                   float * phi_filt,
                   float * coh_filt,
                   const int height,
                   const int width);

        insar_data(const insar_data &data);
        insar_data(const insar_data_shared &data);

        ~insar_data();

    private:
        template<class T>
        void copy(const T& data)
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
        };
};

#endif

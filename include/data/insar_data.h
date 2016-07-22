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
        float * ref_filt;
        float * phi_filt;
        float * coh_filt;
        int height;
        int width;

        insar_data_shared(float * a1,
                          float * a2,
                          float * dp,
                          float * ref_filt,
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
                   float * ref_filt,
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

            ref_filt = (float *) malloc(bytesize);
            memcpy(ref_filt, data.ref_filt, bytesize);

            phi_filt = (float *) malloc(bytesize);
            memcpy(phi_filt, data.phi_filt, bytesize);

            coh_filt = (float *) malloc(bytesize);
            memcpy(coh_filt, data.coh_filt, bytesize);
        };
};

#endif

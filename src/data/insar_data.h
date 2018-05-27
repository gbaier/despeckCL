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
#include <memory>

#include "tile.h"


/*
template<typename Type, size_t D>
class sar_data {
    public:
        int height;
        int width;
        int dim = D;

        sar_data(std::unique_ptr<Type> data, int height, int width, int dim);
        sar_data(const sar_data<Type, D>& other);
        sar_data& operator=(const sar_data &other);

        sar_data<Type, D> get_tile(tile t);
        void write_tile(sar_data<Type, D> other, tile t);

    private:
        std::unique_ptr<float> data;
};*/

#include <memory>

class insar_data
{
    private:
        std::unique_ptr<float[]> _data;

    public:
        int height;
        int width;
        const int dim = 6;

        // takes ownership
        insar_data(std::unique_ptr<float[]> data,
                   int height,
                   int width);

        // Allocates memory and copies data.
        // This is for interfacing with C-libraries/programs
        // or Python via SWIG.
        insar_data(float * a1,
                   float * a2,
                   float * dp,
                   float * ref_filt,
                   float * phi_filt,
                   float * coh_filt,
                   int height,
                   int width);

        insar_data(const insar_data& other) = delete;
        insar_data(insar_data&& other) noexcept;
        insar_data& operator=(insar_data &&other) noexcept;

        // pubic interface that abstracts the internel data representation
        float * data()        const { return _data.get(); };
        float * ampl_master() const { return _data.get(); };
        float * ampl_slave()  const { return _data.get() +   height*width; };
        float * phase()       const { return _data.get() + 2*height*width; };
        float * ref_filt()    const { return _data.get() + 3*height*width; };
        float * phase_filt()  const { return _data.get() + 4*height*width; };
        float * coh_filt()    const { return _data.get() + 5*height*width; };

        ~insar_data() {};
};

insar_data tileget(const insar_data& img_data, tile<2> sub);
void tilecpy(insar_data& img_data, const insar_data& img_tile, tile<2> sub);

#endif

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

#ifndef TILE_H
#define TILE_H

#include "insar_data.h"

class tile
{
    public:
        tile(insar_data_shared& img_data,
             const int h_low,
             const int w_low,
             const int tile_size,
             const int overlap);

        void write(insar_data_shared& img_data);
        insar_data& get();

    private:
        const int h_low;
        const int w_low;
        const int overlap;
        insar_data tile_data;
        insar_data copy_tile_data(insar_data_shared& img_data, const int tile_size);
};

#endif

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

#ifndef TILE_ITERATOR_H
#define TILE_ITERATOR_H

#include "tile.h"

class tile_iterator
{
    protected:
        const int max_height;
        const int max_width;
        const int tile_height;
        const int tile_width;
        const int overlap_border;
        const int overlap_tile;

        //tiles lower corner coordinates
        int h_low;
        int w_low;

    public:
        tile_iterator(const int max_height,
                      const int max_width,
                      const int tile_size,
                      const int overlap_border,
                      const int overlap_tile);

        tile_iterator(const int max_height,
                      const int max_width,
                      const int tile_height,
                      const int tile_width,
                      const int overlap_border,
                      const int overlap_tile);

        void operator++();

        bool operator!=(const tile_iterator&) const;

        tile<2> operator*() const;

        const tile_iterator& begin() const;
        const tile_iterator& end() const;

};
#endif

#ifndef TILE_ITERATOR_H
#define TILE_ITERATOR_H

#include "insar_data.h"
#include "tile.h"

class tile_iterator
{
    protected:
        insar_data_shared& data;
        const int tile_size;
        const int overlap_border;
        const int overlap_tile;

        //tiles lower corner coordinates
        int h_low;
        int w_low;

    public:
        tile_iterator(insar_data_shared& data,
                      const int tile_size,
                      const int overlap_border,
                      const int overlap_tile);

        void operator++();

        bool operator!=(const tile_iterator&) const;

        tile operator*() const;

        const tile_iterator& begin() const;
        const tile_iterator& end() const;

};
#endif

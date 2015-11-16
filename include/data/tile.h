#ifndef TILE_H
#define TILE_H

#include "insar_data.h"

class tile
{
    public:
        tile(insar_data& img_data,
             const int h_low,
             const int w_low,
             const int tile_size,
             const int overlap);

        void write(insar_data& img_data);
        insar_data& get();

    private:
        const int h_low;
        const int w_low;
        const int overlap;
        insar_data tile_data;
        insar_data copy_tile_data(insar_data& img_data, const int tile_size);
};

#endif

#include "tile_iterator.h"

tile_iterator::tile_iterator(insar_data_shared& data,
                             const int tile_size,
                             const int overlap_border,
                             const int overlap_tile) : data(data),
                                                       tile_size(tile_size),
                                                       overlap_border(overlap_border),
                                                       overlap_tile(overlap_tile),
                                                       h_low(-overlap_border),
                                                       w_low(-overlap_border) {}

void tile_iterator::operator++()
{
    w_low += tile_size - 2*overlap_tile;
    if (w_low >= data.width) {
        w_low = -overlap_border;
        h_low += tile_size - 2*overlap_tile;
    }
}

bool tile_iterator::operator!=(const tile_iterator&) const
{
    return h_low <= data.height;
}

tile tile_iterator::operator*() const
{
    return tile{data, h_low, w_low, tile_size, overlap_tile};
}

const tile_iterator& tile_iterator::begin() const
{
    return *this;
}

const tile_iterator& tile_iterator::end() const
{
    return *this;
}

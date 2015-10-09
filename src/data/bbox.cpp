#include "bbox.h"

bbox::bbox(const int h_low, const int w_low, const int h_up, const int w_up) : h_low(h_low), w_low(w_low), h_up(h_up), w_up(w_up) {}

bool bbox::valid(const int height, const int width)
{
    if ( h_up > height ||
         w_up > width  ||
         h_low < 0     ||
         w_low < 0 ) {
        return false;
    } else {
        return true;
    }
}

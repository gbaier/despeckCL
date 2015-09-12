#ifndef BEST_PARAMS_H
#define BEST_PARAMS_H

#include <vector>
#include <map>


namespace nlsar {
    struct params {
        const int search_window_size;
        const int patch_size;
        const int scale_size;

        bool operator== (const params& other) const {
            return (search_window_size == other.search_window_size) && \
                           (patch_size == other.patch_size) && \
                           (scale_size == other.scale_size);
        };

        // needed if params is to be used as a key for a map
        bool operator< (const params& other) const {
            return (search_window_size < other.search_window_size) || \
                           (patch_size < other.patch_size) || \
                           (scale_size < other.scale_size);
        };
    };

    std::vector<params> best_params(std::map<params, std::vector<float>> &enl,
                                    const int height,
                                    const int width);
}

#endif

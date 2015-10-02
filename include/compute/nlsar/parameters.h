#ifndef PARAMETERS_H
#define PARAMETERS_H

namespace nlsar {
    struct params {
        int patch_size;
        int scale_size;

        bool operator== (const params& other) const {
            return (patch_size == other.patch_size) && \
                   (scale_size == other.scale_size);
        };

        // needed if params is to be used as a key for a map
        bool operator< (const params& other) const {
            return  (patch_size < other.patch_size) || \
                   ((patch_size == other.patch_size) && (scale_size < other.scale_size));
        };

        params& operator= (const params& other) {
            patch_size = other.patch_size;
            scale_size = other.scale_size;
            return *this;
        };
    };
}

#endif

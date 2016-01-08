#ifndef PATCHES_H
#define PATCHES_H

#include <vector>
#include <cstdint>

namespace nlsar {
    namespace training {
        class data
        {
            private:
                const uint32_t height;
                const uint32_t width;
                const uint32_t dimension;
                float * const covmats;

            public:
                data(const float * const covmats,
                        const uint32_t height,
                        const uint32_t width,
                        const uint32_t dimension);

                data(const data& other);

                ~data();

                uint32_t get_height(void) const;
                uint32_t get_width(void) const;
                uint32_t get_dimension(void) const;
                float * get_covmats(void) const;

                std::vector<data> get_all_patches(const uint32_t patch_size);
                float dissimilarity(const data& other);

            private:
                data get_patch(const uint32_t upper_h,
                        const uint32_t left_w,
                        const uint32_t patch_size);
        };

    }
}

#endif

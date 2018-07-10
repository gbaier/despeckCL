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

#ifndef PATCHES_H
#define PATCHES_H

#include <vector>
#include <cstdint>

namespace nlsar {
    namespace training {
        class data
        {
            private:
                const uint32_t _height;
                const uint32_t _width;
                const uint32_t _dimension;
                float * const covmats;

            public:
                data(const float * const covmats,
                        const uint32_t height,
                        const uint32_t width,
                        const uint32_t dimension);

                data(const data& other);

                ~data();

                uint32_t height(void) const;
                uint32_t width(void) const;
                uint32_t dim(void) const;
                float * get_covmats(void) const;

        };

        data get_patch(const data& training_data,
                       uint32_t upper_h,
                       uint32_t left_w,
                       uint32_t patch_size);

        std::vector<data> get_all_patches(const data& training_data,
                                          uint32_t patch_size);
        float dissimilarity(const data& first, const data& second);
    }
}

#endif

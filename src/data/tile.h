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

#include <stdexcept>
#include <array>

class slice {
    public:
        const int start;
        const int stop;

        slice(const int start, const int stop) : start(start), stop(stop) {
            if (not *this) {
                throw(std::range_error("start must be smaller than stop"));
            }
        }

        slice(const slice& other) =default;

        ~slice() =default;

        slice get_sub (const int sub_start, const int sub_stop) const {
            if (sub_start < 0 || sub_stop > 0) {
                throw(std::range_error("sub_start must be geq than 0 and sub_stop leq than 0"));
            }
            return slice(start + sub_start, stop + sub_stop);
        }

    private:
        explicit operator bool() const {
            return start < stop;
        }
};

template<size_t N>
using tile = std::array<slice, N>;

#endif

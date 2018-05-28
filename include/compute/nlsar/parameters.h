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

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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

        private:
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int /* file_version */)
            {
                ar & patch_size;
                ar & scale_size;
            }
    };
}

#endif

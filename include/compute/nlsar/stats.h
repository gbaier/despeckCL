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

#ifndef STATS_H
#define STATS_H

#include <vector>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace nlsar {
    class stats
    {
        public:
            stats();
            stats(std::vector<float> dissims, unsigned int lut_size);
            stats& operator=(const stats& other);
            unsigned int lut_size;
            float dissims_min;
            float dissims_max;
            std::vector<float> quantilles;
            std::vector<float> chi2cdf_inv;

        private:
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int /* file_version */)
            {
                ar & lut_size;
                ar & dissims_min;
                ar & dissims_max;
                ar & quantilles;
                ar & chi2cdf_inv;
            }

            std::vector<float> get_quantilles(std::vector<float> &dissims);
            std::vector<float> get_chi2cdf_inv(void);
            float get_max_quantilles_error();
    };
}

#endif

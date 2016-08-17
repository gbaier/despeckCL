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

#ifndef INSARSIM_SIMU_H
#define INSARSIM_SIMU_H

#include <vector>
#include <tuple>

namespace nlinsar {
    namespace simu {

        float quantile(std::vector<float> vector, float alpha);
        
        std::tuple<float, float, float> insar_gen(void);

        float quantile_insar(int patch_size, float alpha);
    }
}

#endif

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

#ifndef BEST_PARAMS_H
#define BEST_PARAMS_H

#include <vector>
#include <map>
#include <utility>

#include "parameters.h"
#include "../compute_env.h"

namespace nlsar {

    class get_best_params : public routine_env<get_best_params>
    {
        public:
            std::string routine_name{"get_best_params"};

            void run(std::map<params, std::vector<float>> &enl,
                     std::vector<params>* best_parameters,
                     const int height,
                     const int width);
        private:
            params get_best_pixel_params(std::vector<std::pair<params, float>> params_enl);
     };
}

#endif

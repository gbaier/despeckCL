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

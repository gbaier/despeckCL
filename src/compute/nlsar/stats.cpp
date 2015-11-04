#include "stats.h"

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_cdf.h>
#include <iostream>

#include <algorithm>

nlsar::stats::stats(std::vector<float> dissims, unsigned int patch_size, unsigned int lut_size): lut_size(lut_size)
{
    std::sort(dissims.begin(), dissims.end());
    std::remove_if(dissims.begin(), dissims.end(), [] (float num) { return std::isnan(num);} );
    dissims_min = *std::min_element(dissims.begin(), dissims.end());
    dissims_max = *std::max_element(dissims.begin(), dissims.end());

    for(float dissim = dissims_min; dissim<dissims_max; dissim += (dissims_max - dissims_min)/lut_size) {
        const std::vector<float>::iterator lower_bound = std::lower_bound(dissims.begin(), dissims.end(), dissim);

        const float dissim_resampled_upper = *lower_bound;
        float dissim_resampled_lower = 0;
        if (lower_bound == dissims.begin()) {
            dissim_resampled_lower = dissim_resampled_upper;
        } else {
            dissim_resampled_lower = *(lower_bound-1);
        }

        //std::cout << dissim_resampled_lower << ", " << dissim << ", " << dissim_resampled_upper << std::endl;
        if (dissim - dissim_resampled_lower < dissim_resampled_upper - dissim) {
            const float rel_idx = ((float) (lower_bound - dissims.begin()-1))/dissims.size();
            this->dissims2relidx.push_back(rel_idx);
        } else {
            const float rel_idx = ((float) (lower_bound - dissims.begin()))/dissims.size();
            this->dissims2relidx.push_back(rel_idx);
        }
    }

    for(float d=0; d<1; d += 1./lut_size) {
        // chi square cdf with patch_size*patch_size degrees of freedom
        chi2cdf_inv.push_back(gsl_cdf_chisq_Pinv(d, 49));
    }
}

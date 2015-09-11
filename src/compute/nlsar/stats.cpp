#include "stats.h"

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_cdf.h>
#include <iostream>

#include <algorithm>

float stats::dissim_lookup(float dissim)
{
    std::vector<float>::iterator low = std::lower_bound(dissims.begin(), dissims.end(), dissim);
    return dissim2relidx[*low];
}

float stats::chi2cdf_inv_lookup(float idx)
{
    return chi2cdf_inv[(unsigned int) lut_size*idx];
}

stats::stats(std::vector<float> dissims, unsigned int patch_size, unsigned int lut_size): patch_size(patch_size), c(patch_size*patch_size), lut_size(lut_size)
{
    std::sort(dissims.begin(), dissims.end());
    std::remove_if(dissims.begin(), dissims.end(), [] (float num) { return std::isnan(num);} );
    dissims_min = *std::min_element(dissims.begin(), dissims.end());
    dissims_max = *std::max_element(dissims.begin(), dissims.end());

    std::vector<float> dissims_resampled;

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
            dissims_resampled.push_back(dissim_resampled_lower);
            const float rel_idx = ((float) (lower_bound - dissims.begin()-1))/dissims.size();
            this->dissim2relidx[dissim_resampled_lower] = rel_idx;
        } else {
            dissims_resampled.push_back(dissim_resampled_upper);
            const float rel_idx = ((float) (lower_bound - dissims.begin()))/dissims.size();
            this->dissim2relidx[dissim_resampled_upper] = rel_idx;
        }
    }
    this -> dissims = dissims_resampled;

    for(float d=0; d<1; d += 1./lut_size) {
        // chi square cdf with patch_size*patch_size degrees of freedom
        chi2cdf_inv.push_back(gsl_cdf_chisq_Pinv(d, patch_size*patch_size));
    }
}

float stats::weight(float dissim)
{
    // relative index in the cdf of the dissimilarities idx \in (0,1)
    float idx = dissim_lookup(dissim);
    // find corresponding x value in chi2cdf_inv
    float x = chi2cdf_inv_lookup(idx);
    return exp(-(std::abs(x-c))/h);
}

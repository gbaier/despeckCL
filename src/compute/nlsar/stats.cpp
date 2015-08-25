#include "stats.h"

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_cdf.h>

#include <algorithm>

float stats::dissim_lookup(float dissim)
{
    std::vector<float>::iterator low = std::lower_bound(dissims.begin(), dissims.end(), dissim);
    float idx = low - dissims.begin();
    return idx/dissims.size();
}

float stats::chi2cdf_inv_lookup(float idx)
{
    return chi2cdf_inv[(unsigned int) size*idx];
}

stats::stats(std::vector<float> dissims, unsigned int patch_size): patch_size(patch_size), c(patch_size*patch_size)
{
    std::sort(dissims.begin(), dissims.end());
    std::remove_if(dissims.begin(), dissims.end(), [] (float num) { return std::isnan(num);} );
    this -> dissims = dissims;

    for(float d=0; d<1; d += 1./size) {
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

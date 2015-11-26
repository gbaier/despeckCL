#include "stats.h"

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_cdf.h>

#include <algorithm>
#include <functional>

nlsar::stats::stats(std::vector<float> dissims, unsigned int lut_size): lut_size(lut_size)
{
    std::sort(dissims.begin(), dissims.end());
    std::remove_if(dissims.begin(), dissims.end(), [] (float num) { return std::isnan(num);} );
    dissims_min = *std::min_element(dissims.begin(), dissims.end());
    dissims_max = *std::max_element(dissims.begin(), dissims.end());

    this->quantilles  = get_quantilles(dissims);
    this->chi2cdf_inv = get_chi2cdf_inv();
}

std::vector<float> nlsar::stats::get_quantilles(std::vector<float> &dissims)
{
    const float step_size = (dissims_max - dissims_min)/lut_size;
    
    std::vector<float> quantilles;
    quantilles.reserve(lut_size);

    for(int i = 0; i < lut_size; i++) {
        const float dissim = dissims_min + i*step_size;
        const std::vector<float>::iterator lower_bound = std::lower_bound(dissims.begin(), dissims.end(), dissim);
        quantilles.push_back( ((float) (lower_bound - dissims.begin()))/dissims.size() );
    }
    return quantilles;
}

std::vector<float> nlsar::stats::get_chi2cdf_inv(void)
{
    const float step_size = 1.0f/lut_size;

    std::vector<float> chi2cdf_inv;
    chi2cdf_inv.reserve(lut_size);

    for(int i=0; i < lut_size; i++) {
        const float y = i*step_size;
        // inverse of the chi square cdf with 49 degrees of freedom
        chi2cdf_inv.push_back(gsl_cdf_chisq_Pinv(y, 49));
    }
    return chi2cdf_inv;
}

float nlsar::stats::get_max_quantilles_error(std::vector<float> &quantilles)
{
    std::vector<float> diffs;

    std::transform(quantilles.begin()+1,
                   quantilles.end(),
                   quantilles.begin(),
                   std::back_inserter(diffs),
                   std::minus<float>());

}

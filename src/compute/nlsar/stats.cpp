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

#include "stats.h"

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_cdf.h>

#include <algorithm>
#include <functional>
#include <cmath>

nlsar::stats::stats(std::vector<float> dissims, unsigned int lut_size): lut_size(lut_size)
{
    std::sort(dissims.begin(), dissims.end());
    std::remove_if(dissims.begin(), dissims.end(), [] (float num) { return std::isnan(num);} );
    dissims_min = *std::min_element(dissims.begin(), dissims.end());
    dissims_max = *std::max_element(dissims.begin(), dissims.end());

    this->quantilles  = get_quantilles(dissims);
    this->chi2cdf_inv = get_chi2cdf_inv();
}

nlsar::stats::stats(): lut_size(-1) {}

nlsar::stats& nlsar::stats::operator=(const nlsar::stats& other)
{
  this->dissims_min = other.dissims_min;
  this->dissims_max = other.dissims_max;
  this->quantilles  = other.quantilles;
  this->chi2cdf_inv = other.chi2cdf_inv;
  return *this;
}

std::vector<float> nlsar::stats::get_quantilles(std::vector<float> &dissims)
{
    const float step_size = (dissims_max - dissims_min)/(lut_size-1);
    
    std::vector<float> quantilles;
    quantilles.reserve(lut_size);

    for(size_t i = 0; i < lut_size; i++) {
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

    for(size_t i = 0; i < lut_size-1; i++) {
        const float y = i*step_size;
        // inverse of the chi square cdf with 49 degrees of freedom
        chi2cdf_inv.push_back(gsl_cdf_chisq_Pinv(y, 49));
    }
    chi2cdf_inv.push_back(100000);
    return chi2cdf_inv;
}

float nlsar::stats::get_max_quantilles_error()
{
    std::vector<float> diffs;

    std::transform(quantilles.begin()+1,
                   quantilles.end(),
                   quantilles.begin(),
                   std::back_inserter(diffs),
                   std::minus<float>());

    return *std::max_element(diffs.begin(), diffs.end());
}

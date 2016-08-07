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

#include "insarsim_simu.h"

#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

#include "sim_measures.h"

float nlinsar::simu::quantile(std::vector<float> vector, float alpha)
{
    std::sort(vector.begin(), vector.end());
    return vector[alpha*vector.size()];
}

std::tuple<float, float, float> nlinsar::simu::insar_gen(void)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> randn(0, 1);

    const double R = 1.0; // reflectivity
    const double B = 1.0; // interferometric phase
    const double D = 0.0; // coherence

    double l1 = sqrt(R);
    std::complex<double> l2 = std::polar(l1*D, B);
    double l3 = l1* sqrt(1 - D*D);
    std::complex<double> x1 {randn(gen), randn(gen)};
    std::complex<double> x2 {randn(gen), randn(gen)};
    x1 /= std::sqrt(2);
    x2 /= std::sqrt(2);

    std::complex<double> slc1 = l1*x1;
    std::complex<double> slc2 = l2*x1 + l3*x2;

    const float ampl_master = std::abs(slc1);
    const float ampl_slave  = std::abs(slc2);
    const float phase      = std::arg(slc1*std::conj(slc2));

    return std::make_tuple(ampl_master, ampl_slave, phase);
}

float nlinsar::simu::quantile_insar(int patch_size, float alpha)
{
    constexpr const int size = 2000;
    std::vector<float> similarities(size);
#pragma omp parallel for
    for(int i = 0; i<size; i++) {
        float similarity = 0;
        const float patch_area = std::pow(patch_size, 2);
        for(int j = 0; j<patch_area; j++) {
            float am1, as1, dp1;
            std::tie(am1, as1, dp1) = insar_gen();
            float am2, as2, dp2;
            std::tie(am2, as2, dp2) = insar_gen();
            similarity += pixel_similarity(am1, as1, dp1, am2, as2, dp2);
        }
        similarities[i] = -similarity/patch_area;
    }
    const float q  = quantile(similarities, alpha);
    const float mu = std::accumulate(similarities.begin(), similarities.end(), 0.0f)/similarities.size();
    return q - mu;
}

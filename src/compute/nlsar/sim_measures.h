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

#ifndef NLSAR_SIM_MEASURES_H
#define NLSAR_SIM_MEASURES_H

#ifndef __OPENCL_VERSION__
#include <cmath>
#endif

// square of the absolute value
inline float abs2(float real, float imag) {
    return real*real + imag*imag;
}

inline float det_covmat_2x2(float el_00, float el_01real, float el_01imag, float el_11)
{
    return (el_00*el_11) - (el_01real*el_01real + el_01imag*el_01imag);
}


inline float
pixel_similarity_2x2(float el_00_p1,
                     float el_01real_p1,
                     float el_01imag_p1,
                     float el_11_p1,
                     float el_00_p2,
                     float el_01real_p2,
                     float el_01imag_p2,
                     float el_11_p2,
                     const int nlooks)
{
#ifndef __OPENCL_VERSION__
  using std::isnan;
#endif
  const int dimensions = 2;
  float nom1 = det_covmat_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1);
  float nom2 = det_covmat_2x2(el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2);
  float det  = det_covmat_2x2(el_00_p1 + el_00_p2,
                             el_01real_p1 + el_01real_p2,
                             el_01imag_p1 + el_01imag_p2,
                             el_11_p1 + el_11_p2);

  float similarity = -nlooks * (2 * dimensions * log(2.0f) + log(nom1) +
                                log(nom2) - 2 * log(det));
  if (isnan(similarity)) {
    similarity = 0;
  }
  return similarity;
}


/* computes the docstring of a 3x3 hermitian matrix using Leibniz formula:
 *
 * In general the formula for a 3x3 determinant is
 *
 * a_00*a_11*a_22 - a_00*a_12*a_21 - a_01*a_10*a22 + a_01*a_12*a_20 + a_02*a_10*a_21 - a_02*a_11*a_20
 *
 * For a hermitian matrix this simplifies due to symmetries:
 *
 * a_00*a_11*a_22 - a_00*|a_12|^2 - a_11*|a_02|^2 - a_22*|a_01|^2 + 2Re{a_01*a_12*a_20}
 *
 */
inline float
det_covmat_3x3(float a_00,
               float a_11,
               float a_22,
               float a_01_real,
               float a_01_imag,
               float a_02_real,
               float a_02_imag,
               float a_12_real,
               float a_12_imag)
{
  // a_20_real ==  a_02_real
  // a_20_imag == -a_02_imag
  return a_00 * a_11 * a_22 - a_00 * abs2(a_12_real, a_12_imag) -
         a_11 * abs2(a_02_real, a_02_imag) - a_22 * abs2(a_01_real, a_01_imag) +
         2 * (a_01_real * a_12_real * a_02_real -
              a_01_imag * a_12_imag * a_02_real +
              a_01_real * a_12_imag * a_02_imag +
              a_01_imag * a_12_real * a_02_imag);
}


inline float
pixel_similarity_3x3(float p1_a_00,
                     float p1_a_11,
                     float p1_a_22,
                     float p1_a_01_real,
                     float p1_a_01_imag,
                     float p1_a_02_real,
                     float p1_a_02_imag,
                     float p1_a_12_real,
                     float p1_a_12_imag,
                     float p2_a_00,
                     float p2_a_11,
                     float p2_a_22,
                     float p2_a_01_real,
                     float p2_a_01_imag,
                     float p2_a_02_real,
                     float p2_a_02_imag,
                     float p2_a_12_real,
                     float p2_a_12_imag,
                     const int nlooks)
{
#ifndef __OPENCL_VERSION__
  using std::isnan;
#endif
  const int dimensions = 3;
  float nom1 = det_covmat_3x3(p1_a_00,
                              p1_a_11,
                              p1_a_22,
                              p1_a_01_real,
                              p1_a_01_imag,
                              p1_a_02_real,
                              p1_a_02_imag,
                              p1_a_12_real,
                              p1_a_12_imag);
  float nom2 = det_covmat_3x3(p2_a_00,
                              p2_a_11,
                              p2_a_22,
                              p2_a_01_real,
                              p2_a_01_imag,
                              p2_a_02_real,
                              p2_a_02_imag,
                              p2_a_12_real,
                              p2_a_12_imag);
  float det = det_covmat_3x3(p1_a_00      + p2_a_00,
                             p1_a_11      + p2_a_11,
                             p1_a_22      + p2_a_22,
                             p1_a_01_real + p2_a_01_real,
                             p1_a_01_imag + p2_a_01_imag,
                             p1_a_02_real + p2_a_02_real,
                             p1_a_02_imag + p2_a_02_imag,
                             p1_a_12_real + p2_a_12_real,
                             p1_a_12_imag + p2_a_12_imag);

  float similarity = -nlooks * (2 * dimensions * log(2.0f) + log(nom1) +
                                log(nom2) - 2 * log(det));
  if (isnan(similarity)) {
   similarity = 0;
  }
  return similarity;
}
#endif

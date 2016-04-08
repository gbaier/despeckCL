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

float det_covmat_2x2(float el_00, float el_01real, float el_01imag, float el_11)
{
    return (el_00*el_11) - (el_01real*el_01real + el_01imag*el_01imag);
}

float pixel_similarity_2x2(float el_00_p1, float el_01real_p1, float el_01imag_p1, float el_11_p1,
                           float el_00_p2, float el_01real_p2, float el_01imag_p2, float el_11_p2,
                           const int nlooks)
{
    const int dimensions = 2;
    float nom1 = det_covmat_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1);
    float nom2 = det_covmat_2x2(el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2);
    float det  = det_covmat_2x2(el_00_p1     + el_00_p2,
                                el_01real_p1 + el_01real_p2,
                                el_01imag_p1 + el_01imag_p2,
                                el_11_p1     + el_11_p2);

    float similarity = -nlooks*( 2*dimensions*log(2.0f) +  log(nom1) + log(nom2) - 2*log(det) );
    if (isnan(similarity)) {
        similarity = 0;
    }
    return similarity;
}

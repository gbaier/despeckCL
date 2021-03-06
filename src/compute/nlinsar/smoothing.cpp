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

#include "smoothing.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>

namespace nlinsar {
    struct weight_el {
        float weight;
        int idx;

        // lowest weights get in front of the priority queue
        bool operator<(const weight_el& other) const
        {
            return weight > other.weight;
        }
    };
}

void nlinsar::smoothing::run(cl::CommandQueue cmd_queue,
                             cl::Buffer device_weights,
                             cl::Buffer device_nols,
                             float * ampl_master,
                             const int height_ori,
                             const int width_ori,
                             const int search_window_size,
                             const int patch_size,
                             const int lmin)
{
    const int n_elem_ori = search_window_size * search_window_size * height_ori * width_ori;
    const int wsh = (search_window_size-1)/2;
    const int psh = (patch_size-1)/2;
    const int overlap = wsh + psh;

    float* number_of_looks = (float*) malloc(height_ori*width_ori*sizeof(float));
    float* weights         = (float*) malloc(n_elem_ori*sizeof(float));

    cmd_queue.enqueueReadBuffer(device_weights, CL_TRUE, 0, n_elem_ori*sizeof(float)          , weights,         NULL, NULL);
    cmd_queue.enqueueReadBuffer(device_nols,    CL_TRUE, 0, height_ori*width_ori*sizeof(float), number_of_looks, NULL, NULL);

    for(int h = 0; h < height_ori; h++) {
        for(int w = 0; w < width_ori; w++) {
            float * window_weights = weights + (search_window_size*search_window_size*(h*width_ori+w));
            if (number_of_looks[h*width_ori + w] < lmin) {
                search_window_smoothing(ampl_master, window_weights,
                                        h, w,
                                        width_ori + 2*overlap,
                                        patch_size,
                                        search_window_size,
                                        lmin);
            }
        }
    }
    cmd_queue.enqueueWriteBuffer(device_weights, CL_TRUE, 0, n_elem_ori*sizeof(float), weights, NULL, NULL);
    free(number_of_looks);
    free(weights);
}

void nlinsar::search_window_smoothing(const float * amplitude_master,
                                      float * weights,
                                      const int h,
                                      const int w,
                                      const int width_overlap,
                                      const int patch_size,
                                      const int search_window_size,
                                      const int lmin)
{
    const int psh = (patch_size - 1)/2;
    const int wsh = (search_window_size - 1)/2;
    const int overlap = wsh+psh;

    std::priority_queue<weight_el> ws;

    const double as = amplitude_master[(h+overlap) * width_overlap + (w+overlap)];

    for(int hh=0; hh < search_window_size; hh++) {
        for(int ww=0; ww < search_window_size; ww++) {
            const int window_idx = (h+overlap+hh-wsh) * width_overlap + (w+overlap+ww-wsh);
            const float at = amplitude_master[window_idx];
            if (at < 2.0*as) {
                const int   idx    = hh*search_window_size + ww;
                const float weight = weights[idx];
                // if statement is for priming the priority queue
                if ( ((int) ws.size()) < lmin) {
                    ws.push( weight_el{weight, idx} );
                // else if inserts the higher weights into the queue
                } else if (ws.top().weight < weight) {
                    ws.pop();
                    ws.push( weight_el{weight, idx} );
                }
            }
        }
    }

    std::vector<int> idxs;
    float wsum = 0.0f;

    while(!ws.empty()) {
        const weight_el el = ws.top();
        wsum += el.weight;
        idxs.push_back(el.idx);
        ws.pop();
    }

    wsum /= idxs.size();

    for(auto idx : idxs) {
        weights[idx] = wsum;
    }
}

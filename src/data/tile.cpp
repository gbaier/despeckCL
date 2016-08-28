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

#include "tile.h"
#include "sub_images.h"

tile::tile(insar_data_shared& img_data,
           const int h_low,
           const int w_low,
           const int tile_height,
           const int tile_width,
           const int overlap) : h_low(h_low),
                                w_low(w_low),
                                overlap(overlap),
                                tile_data(copy_tile_data(img_data, tile_height, tile_width))
{
}

insar_data tile::copy_tile_data(insar_data_shared& img_data, const int tile_height, const int tile_width)
{
    float * const a1_sub       = get_sub_image(img_data.a1,       img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    float * const a2_sub       = get_sub_image(img_data.a2,       img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    float * const dp_sub       = get_sub_image(img_data.dp,       img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    float * const ref_filt_sub = get_sub_image(img_data.ref_filt, img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    float * const phi_filt_sub = get_sub_image(img_data.phi_filt, img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    float * const coh_filt_sub = get_sub_image(img_data.coh_filt, img_data.height, img_data.width, h_low, w_low, tile_height, tile_width);
    insar_data sub_image{a1_sub, a2_sub, dp_sub, ref_filt_sub, phi_filt_sub, coh_filt_sub, tile_height, tile_width};
    free(a1_sub);
    free(a2_sub);
    free(dp_sub);
    free(ref_filt_sub);
    free(phi_filt_sub);
    free(coh_filt_sub);
    return sub_image;
}

insar_data& tile::get()
{
    return tile_data;
}

void tile::write(insar_data_shared& img_data)
{
    write_sub_image(img_data.ref_filt, img_data.height, img_data.width, tile_data.ref_filt, h_low, w_low, tile_data.height, tile_data.width, overlap);
    write_sub_image(img_data.phi_filt, img_data.height, img_data.width, tile_data.phi_filt, h_low, w_low, tile_data.height, tile_data.width, overlap);
    write_sub_image(img_data.coh_filt, img_data.height, img_data.width, tile_data.coh_filt, h_low, w_low, tile_data.height, tile_data.width, overlap);
}

#include "tile.h"
#include "sub_images.h"

tile::tile(insar_data_shared& img_data,
           const int h_low,
           const int w_low,
           const int tile_size,
           const int overlap) : h_low(h_low),
                                w_low(w_low),
                                overlap(overlap),
                                tile_data(copy_tile_data(img_data, tile_size))
{
}

insar_data tile::copy_tile_data(insar_data_shared& img_data, const int tile_size)
{
    float * const a1_sub       = get_sub_image(img_data.a1,       img_data.height, img_data.width, h_low, w_low, tile_size);
    float * const a2_sub       = get_sub_image(img_data.a2,       img_data.height, img_data.width, h_low, w_low, tile_size);
    float * const dp_sub       = get_sub_image(img_data.dp,       img_data.height, img_data.width, h_low, w_low, tile_size);
    float * const amp_filt_sub = get_sub_image(img_data.amp_filt, img_data.height, img_data.width, h_low, w_low, tile_size);
    float * const phi_filt_sub = get_sub_image(img_data.phi_filt, img_data.height, img_data.width, h_low, w_low, tile_size);
    float * const coh_filt_sub = get_sub_image(img_data.coh_filt, img_data.height, img_data.width, h_low, w_low, tile_size);
    insar_data sub_image{a1_sub, a2_sub, dp_sub, amp_filt_sub, phi_filt_sub, coh_filt_sub, tile_size, tile_size};
    free(a1_sub);
    free(a2_sub);
    free(dp_sub);
    free(amp_filt_sub);
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
    write_sub_image(img_data.amp_filt, img_data.height, img_data.width, tile_data.amp_filt, h_low, w_low, tile_data.width, overlap);
    write_sub_image(img_data.phi_filt, img_data.height, img_data.width, tile_data.phi_filt, h_low, w_low, tile_data.width, overlap);
    write_sub_image(img_data.coh_filt, img_data.height, img_data.width, tile_data.coh_filt, h_low, w_low, tile_data.width, overlap);
}

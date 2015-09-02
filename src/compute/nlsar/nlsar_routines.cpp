#include "nlsar_routines.h"

nlsar_routines::nlsar_routines(cl::Context context,
                               const int search_window_size,
                               const int window_width,
                               const int dimension,
                               const int block_size) : covmat_create_routine                  (block_size, context),
                                                       covmat_rescale_routine                 (block_size, context),
                                                       covmat_spatial_avg_routine             (block_size, context, window_width),
                                                       compute_pixel_similarities_2x2_routine (block_size, context),
                                                       compute_patch_similarities_routine     (block_size, context),
                                                       compute_number_of_looks_routine        (block_size, context),
                                                       covmat_decompose_routine               (block_size, context),
                                                       weighted_means_routine                 (block_size, context, search_window_size, dimension)
{
}

nlsar_routines::nlsar_routines(const nlsar_routines& other) : covmat_create_routine                  (other.covmat_create_routine),
                                                              covmat_rescale_routine                 (other.covmat_rescale_routine),
                                                              covmat_spatial_avg_routine             (other.covmat_spatial_avg_routine),
                                                              compute_pixel_similarities_2x2_routine (other.compute_pixel_similarities_2x2_routine),
                                                              compute_patch_similarities_routine     (other.compute_patch_similarities_routine),
                                                              compute_number_of_looks_routine        (other.compute_number_of_looks_routine),
                                                              covmat_decompose_routine               (other.covmat_decompose_routine),
                                                              weighted_means_routine                 (other.weighted_means_routine)
{
}

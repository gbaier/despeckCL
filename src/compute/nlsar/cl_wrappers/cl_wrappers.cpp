#include "cl_wrappers.h"

nlsar::cl_wrappers::cl_wrappers(cl::Context context,
                                const int search_window_size,
                                const int dimension,
                                const int block_size) : covmat_create_routine                  (block_size, context),
                                                        covmat_rescale_routine                 (block_size, context),
                                                        covmat_spatial_avg_routine             (block_size, context),
                                                        compute_pixel_similarities_2x2_routine (block_size, context),
                                                        compute_patch_similarities_routine     (context, 16, 4, 4, 4),
                                                        compute_weights_routine                (64, context),
                                                        compute_number_of_looks_routine        (block_size, context),
                                                        compute_nl_statistics_routine          (block_size, context, search_window_size, dimension),
                                                        compute_alphas_routine                 (block_size, context),
                                                        compute_enls_nobias_routine            (block_size, context),
                                                        copy_best_weights_routine              (64, context),
                                                        copy_symm_weights_routine              (-1, context),
                                                        covmat_decompose_routine               (block_size, context),
                                                        weighted_means_routine                 (block_size, context, search_window_size, dimension)
{
}

nlsar::cl_wrappers::cl_wrappers(const cl_wrappers& other) : covmat_create_routine                  (other.covmat_create_routine),
                                                            covmat_rescale_routine                 (other.covmat_rescale_routine),
                                                            covmat_spatial_avg_routine             (other.covmat_spatial_avg_routine),
                                                            compute_pixel_similarities_2x2_routine (other.compute_pixel_similarities_2x2_routine),
                                                            compute_patch_similarities_routine     (other.compute_patch_similarities_routine),
                                                            compute_weights_routine                (other.compute_weights_routine),
                                                            compute_number_of_looks_routine        (other.compute_number_of_looks_routine),
                                                            compute_nl_statistics_routine          (other.compute_nl_statistics_routine),
                                                            compute_alphas_routine                 (other.compute_alphas_routine),
                                                            compute_enls_nobias_routine            (other.compute_enls_nobias_routine),
                                                            copy_best_weights_routine              (other.copy_best_weights_routine),
                                                            copy_symm_weights_routine              (other.copy_symm_weights_routine),
                                                            covmat_decompose_routine               (other.covmat_decompose_routine),
                                                            weighted_means_routine                 (other.weighted_means_routine)
{
}

#include "cl_wrappers.h"

goldstein::cl_wrappers::cl_wrappers(cl::Context context,
                                const int block_size) : raw_interferogram_routine (block_size, context),
                                                        patches_unpack_routine    (block_size, context),
                                                        weighted_multiply_routine (block_size, context),
                                                        patches_pack_routine      (block_size, context),
                                                        slc2real_routine          (block_size, context)
{
}

goldstein::cl_wrappers::cl_wrappers(const cl_wrappers& other) : raw_interferogram_routine (other.raw_interferogram_routine),
                                                                patches_unpack_routine    (other.patches_unpack_routine),
                                                                weighted_multiply_routine (other.weighted_multiply_routine),
                                                                patches_pack_routine      (other.patches_pack_routine),
                                                                slc2real_routine          (other.slc2real_routine)
{
}


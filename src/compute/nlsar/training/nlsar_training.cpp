#include "despeckcl.h"

#include "logging.h"
#include "clcfg.h"
#include "insar_data.h"
#include "get_stats.h"

#include "cl_wrappers.h"

std::map<nlsar::params, nlsar::stats>
despeckcl::nlsar_training(float *ampl_master,
                          float *ampl_slave,
                          float *dphase,
                          const int height,
                          const int width,
                          const std::vector<int> patch_sizes,
                          const std::vector<int> scale_sizes)
{
  logging_setup({});

  float *dummy = (float*) malloc(height*width*sizeof(float));
  insar_data training_data{ampl_master,
                           ampl_slave,
                           dphase,
                           dummy,
                           dummy,
                           dummy,
                           height,
                           width};
  free(dummy);

  size_t dummy_search_window_size = 21;
  size_t dimension = 2;

  cl::Context context = opencl_setup();

  nlsar::cl_wrappers nlsar_cl_wrappers(
      context, dummy_search_window_size, dimension);

  VLOG(0) << "Training weighting kernels";
  return nlsar::training::get_stats(
      patch_sizes, scale_sizes, training_data, context, nlsar_cl_wrappers);
}

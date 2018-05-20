%ignore nlsar_training;

%include "despeckcl_typemaps.i"
%include "std_map.i"

%include "despeckcl.h"
%include "parameters.h"
%include "stats.h"

namespace std {
   %template(nlsar_stats_collections) map<nlsar::params, nlsar::stats>;
}

%feature("docstring") nlsar_train "
    trains the weighting kernel on a homogeneous areas

    :param ndarray ampl_master: the amplitude of the master image
    :param ndarray ampl_slave: the amplitude of the slave image
    :param ndarray phase: the interferometric phase of the master and slave images
    :param [int] patch_sizes: widths of the patches, have to be odd numbers
    :param [int] scale_sizes: widths of the scales, have to be odd numbers
    :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: dissimilarity statistics of the homogeneous training area
    :rtype: wrapped std\:\:map of parameters to dissimilarity statistics
    "

%feature("docstring") store_nlsar_stats_collection "
    stores the NL-SAR statistics in a file

    :param nsc: wrapped std\:\:map of parameters to dissimilarity statistics
    :param [string] filename: filename where to store the NL-SAR dissimilarity statistics
    "

%feature("docstring") load_nlsar_stats_collection "
    loads the NL-SAR statistics from a file

    :param [string] filename: filename where to store the NL-SAR dissimilarity statistics
    :return: dissimilarity statistics
    :rtype: wrapped std\:\:map of parameters to dissimilarity statistics
    "

%inline %{
std::map<nlsar::params, nlsar::stats> nlsar_train(float* ampl_master, int h1, int w1,
                                                  float* ampl_slave,  int h2, int w2,
                                                  float* phase,       int h3, int w3,
                                                  const std::vector<int> patch_sizes,
                                                  const std::vector<int> scale_sizes,
                                                  const std::vector<std::string> enabled_log_levels = {"error", "warning", "fatal"})
{
  return despeckcl::nlsar_training(ampl_master,
                                   ampl_slave,
                                   phase,
                                   h1,
                                   w1,
                                   patch_sizes,
                                   scale_sizes,
                                   enabled_log_levels);
}

void store_nlsar_stats_collection(std::map<nlsar::params, nlsar::stats> nsc, std::string filename)
{
   return despeckcl::store_nlsar_stats_collection(nsc, filename);

}

std::map<nlsar::params, nlsar::stats> load_nlsar_stats_collection(std::string filename)
{
   return despeckcl::load_nlsar_stats_collection(filename);

}

/* NLSAR declaration and wrap */
void _nlsar_c_wrap(float* ampl_master, int h1, int w1,
                   float* ampl_slave,  int h2, int w2,
                   float* phase,      int h3, int w3,
                   float* ref_filt,   int h4, int w4,
                   float* phase_filt, int h5, int w5,
                   float* coh_filt,    int h6, int w6,
                   const int search_window_size,
                   const std::vector<int> patch_sizes,
                   const std::vector<int> scale_sizes,
                   std::map<nlsar::params, nlsar::stats> nlsar_stats,
                   const std::vector<std::string> enabled_log_levels)
{
    despeckcl::nlsar(ampl_master,
                     ampl_slave,
                     phase,
                     ref_filt,
                     phase_filt,
                     coh_filt,
                     h1,
                     w1,
                     search_window_size,
                     patch_sizes,
                     scale_sizes,
                     nlsar_stats,
                     enabled_log_levels);
}
%}

%pythoncode{
import numpy as np

def nlsar(ampl_master,
          ampl_slave,
          phase,
          search_window_size,
          patch_sizes,
          scale_sizes,
          nlsar_stats,
          enabled_log_levels = ['error', 'warning', 'fatal']):
    """
    filters the input with the nlsar filter

    :param ndarray ampl_master: the amplitude of the master image
    :param ndarray ampl_slave: the amplitude of the slave image
    :param ndarray phase: the interferometric phase of the master and slave images
    :param int search_window_size: width of the search window, has to be an odd number
    :param [int] patch_sizes: widths of the patches, have to be odd numbers
    :param [int] scale_sizes: widths of the scales, have to be odd numbers
    :param wrapped std\:\:map nlsar_stats: statistics of a homogenous training area produced by **nlsar_train**
    :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivity, phase and coherence estimates
    :rtype: tuple of ndarrays

    """

    ref_filt   = np.zeros_like(ampl_master)
    phase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)

    _despeckcl._nlsar_c_wrap(ampl_master,
                             ampl_slave,
                             phase,
                             ref_filt,
                             phase_filt,
                             coh_filt,
                             search_window_size,
                             patch_sizes,
                             scale_sizes,
                             nlsar_stats,
                             enabled_log_levels)

    return (ref_filt, phase_filt, coh_filt)
}

%ignore nlsar_training;
%ignore nlsar;

%include "despeckcl_typemaps.i"
%include "std_map.i"
%include "std_string.i"

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
std::map<nlsar::params, nlsar::stats> nlsar_train_insar(float* ampl_master, int h1, int w1,
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


std::map<nlsar::params, nlsar::stats> _nlsar_train_c_wrap(float* covmat_raw, int d1, int h1, int w1,
                                                          int dim,
                                                          const std::vector<int> patch_sizes,
                                                          const std::vector<int> scale_sizes,
                                                          const std::vector<std::string> enabled_log_levels = {"error", "warning", "fatal"})
{
  return despeckcl::nlsar_training(covmat_raw,
                                   h1,
                                   w1,
                                   dim,
                                   patch_sizes,
                                   scale_sizes,
                                   enabled_log_levels);
}

/* NLSAR declaration and wrap */
void _nlsar_c_wrap_insar(float* ampl_master, int h1, int w1,
                         float* ampl_slave,  int h2, int w2,
                         float* phase,      int h3, int w3,
                         float* ref_filt,   int h4, int w4,
                         float* phase_filt, int h5, int w5,
                         float* coh_filt,    int h6, int w6,
                         const int search_window_size,
                         const std::vector<int> patch_sizes,
                         const std::vector<int> scale_sizes,
                         std::map<nlsar::params, nlsar::stats> nlsar_stats,
                         const float h_param,
                         const float c_param,
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
                     h_param,
                     c_param,
                     enabled_log_levels);
}


void _nlsar_c_wrap(float* covmat_raw,  int d1, int h1, int w1, 
                   float* covmat_filt, int d2, int h2, int w2,
                   int dim,
                   const int search_window_size,
                   const std::vector<int> patch_sizes,
                   const std::vector<int> scale_sizes,
                   std::map<nlsar::params, nlsar::stats> nlsar_stats,
                   const float h_param,
                   const float c_param,
                   const std::vector<std::string> enabled_log_levels)
{
    despeckcl::nlsar(covmat_raw,
                     covmat_filt,
                     h1,
                     w1,
                     dim,
                     search_window_size,
                     patch_sizes,
                     scale_sizes,
                     nlsar_stats,
                     h_param,
                     c_param,
                     enabled_log_levels);
}
%}

%pythoncode{
import numpy as np


def nlsar_train(covmat_raw, 
                patch_sizes,
                scale_sizes,
                enabled_log_levels = ['error', 'warning', 'fatal']):

    d, dd, h, w = covmat_raw.shape
    real = covmat_raw.real.reshape((-1, h, w))
    imag = covmat_raw.imag.reshape((-1, h, w))
    covmat_raw_interlace = np.empty((2*d*d, h, w), np.float32)
    covmat_raw_interlace[::2] = real
    covmat_raw_interlace[1::2] = imag

    return _despeckcl._nlsar_train_c_wrap(covmat_raw_interlace, d, patch_sizes, scale_sizes, enabled_log_levels)



def nlsar(covmat_raw,
          search_window_size,
          patch_sizes,
          scale_sizes,
          nlsar_stats,
          h=15.0,
          c=49.0,
          enabled_log_levels = ['error', 'warning', 'fatal']):
    """
    filters the input with the nlsar filter

    :param ndarray covmat_raw: unfiltered covariance/scattering matrix
    :param int search_window_size: width of the search window, has to be an odd number
    :param [int] patch_sizes: widths of the patches, have to be odd numbers
    :param [int] scale_sizes: widths of the scales, have to be odd numbers
    :param wrapped std\:\:map nlsar_stats: statistics of a homogenous training area produced by **nlsar_train**
    :param float h: nonlocal smoothing parameter
    :param float c: degrees of freedom of Chi-squared distribution
    :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivity, phase and coherence estimates
    :rtype: tuple of ndarrays

    """

    d, dd, height, width = covmat_raw.shape
    real = covmat_raw.real.reshape((-1, height, width))
    imag = covmat_raw.imag.reshape((-1, height, width))
    covmat_raw_interlace = np.empty((2*d*d, height, width), np.float32)
    covmat_raw_interlace[::2] = real
    covmat_raw_interlace[1::2] = imag

    del real, imag

    covmat_filt_interlace = np.zeros_like(covmat_raw_interlace)

    _despeckcl._nlsar_c_wrap(covmat_raw_interlace,
                             covmat_filt_interlace,
                             d,
                             search_window_size,
                             patch_sizes,
                             scale_sizes,
                             nlsar_stats,
                             h,
                             c,
                             enabled_log_levels)

    del covmat_raw_interlace

    real_filt = covmat_filt_interlace[::2].reshape((d, d, height, width))
    imag_filt = covmat_filt_interlace[1::2].reshape((d, d, height, width))

    del covmat_filt_interlace

    return real_filt + 1j*imag_filt

def nlsar_insar(ampl_master,
                ampl_slave,
                phase,
                search_window_size,
                patch_sizes,
                scale_sizes,
                nlsar_stats,
                h=15.0,
                c=49.0,
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
    :param float h: nonlocal smoothing parameter
    :param float c: degrees of freedom of Chi-squared distribution
    :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivity, phase and coherence estimates
    :rtype: tuple of ndarrays

    """

    ref_filt   = np.zeros_like(ampl_master)
    phase_filt = np.zeros_like(ampl_master)
    coh_filt   = np.zeros_like(ampl_master)

    _despeckcl._nlsar_c_wrap_insar(ampl_master,
                                   ampl_slave,
                                   phase,
                                   ref_filt,
                                   phase_filt,
                                   coh_filt,
                                   search_window_size,
                                   patch_sizes,
                                   scale_sizes,
                                   nlsar_stats,
                                   h,
                                   c,
                                   enabled_log_levels)

    return (ref_filt, phase_filt, coh_filt)
}

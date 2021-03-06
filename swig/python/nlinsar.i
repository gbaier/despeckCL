%include "despeckcl_typemaps.i"
%include "despeckcl.h"

%inline %{
void _nlinsar_c_wrap(float* ampl_master, int h1, int w1,
                     float* ampl_slave,  int h2, int w2,
                     float* phase,      int h3, int w3,
                     float* ref_filt,   int h4, int w4,
                     float* phase_filt, int h5, int w5,
                     float* coh_filt,    int h6, int w6,
                     const int search_window_size,
                     const int patch_size,
                     const int niter,
                     const int lmin,
                     const std::vector<std::string> enabled_log_levels)
{
    despeckcl::nlinsar(ampl_master,
                       ampl_slave,
                       phase,
                       ref_filt,
                       phase_filt,
                       coh_filt,
                       h1,
                       w1,
                       search_window_size,
                       patch_size,
                       niter,
                       lmin,
                       enabled_log_levels);
}
%}

%pythoncode{
import numpy as np

def nlinsar(ampl_master,
            ampl_slave,
            phase,
            search_window_size,
            patch_size,
            niter,
            lmin,
            enabled_log_levels = ['error', 'warning', 'fatal']):
    """Filters the input with the NLInSAR filter
 
    :param ndarray ampl_master: the amplitude of the master image
    :param ndarray ampl_slave: the amplitude of the slave image
    :param ndarray phase: the interferometric phase of the master and slave images
    :param int search_window_size: width of the search window, has to be an odd number
    :param int patch_size: width of the patch, has to be an odd number
    :param int niter: number of iterations
    :param int lmin: minimum number of looks for the smoothing step
    :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivity, phase and coherence estimates
    :rtype: tuple of ndarrays

    """

    ref_filt   = np.zeros_like(ampl_master)
    phase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)

    _despeckcl._nlinsar_c_wrap(ampl_master,
                               ampl_slave,
                               phase,
                               ref_filt,
                               phase_filt,
                               coh_filt,
                               search_window_size,
                               patch_size,
                               niter,
                               lmin,
                               enabled_log_levels)
    return (ref_filt, phase_filt, coh_filt)
}

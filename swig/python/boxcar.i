%include "despeckcl_typemaps.i"
%include "despeckcl.h"

%ignore boxcar_routines;

/* Boxcar declarations and definitions */
%inline %{
void _boxcar_c_wrap(float* ampl_master, int h1, int w1,
                    float* ampl_slave,  int h2, int w2,
                    float* phase,      int h3, int w3,
                    float* ref_filt,   int h4, int w4,
                    float* phase_filt, int h5, int w5,
                    float* coh_filt,    int h6, int w6,
                    const int window_width,
                    const std::vector<std::string> enabled_log_levels)
{
    despeckcl::boxcar(ampl_master,
                      ampl_slave,
                      phase,
                      ref_filt,
                      phase_filt,
                      coh_filt,
                      h1,
                      w1,
                      window_width,
                      enabled_log_levels);
    return;
}
%}

%pythoncode{
import numpy as np

def boxcar(ampl_master,
           ampl_slave,
           phase,
           window_width,
           enabled_log_levels = ['error', 'warning', 'fatal']):
    """Filters the input with a boxcar filter

    :param ndarray ampl_master: the amplitude of the master image
    :param ndarray ampl_slave: the amplitude of the slave image
    :param ndarray phase: the interferometric phase of the master and slave images
    :param int window_width: the window width of the boxcar window, has to be an odd number
    :return: a tuple containing the reflectivity, phase and coherence estimates
    :rtype: tuple of ndarrays
    """

    ref_filt   = np.zeros_like(ampl_master)
    phase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)
    _despeckcl._boxcar_c_wrap(ampl_master,
                              ampl_slave,
                              phase,
                              ref_filt,
                              phase_filt,
                              coh_filt,
                              window_width,
                              enabled_log_levels);
    return (ref_filt, phase_filt, coh_filt)
}

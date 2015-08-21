%module (docstring="despeckle Toolbox for InSAR written in OpenCL") despeckcl

%{
    #define SWIG_FILE_WITH_INIT
    #include "boxcar.h"
    #include "nlinsar.h"
    #include "nlsar.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply( float* IN_ARRAY2,     int DIM1, int DIM2) {(float* master_amplitude,   int h1, int w1)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2) {(float* slave_amplitude,    int h2, int w2)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2) {(float* dphase,             int h3, int w3)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* amplitude_filtered, int h4, int w4)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* dphase_filtered,    int h5, int w5)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* coherence_filtered, int h6, int w6)}

%ignore boxcar_routines;
%feature("autodoc", "2");


/* Boxcar declarations and definitions */
%include "boxcar.h"


%inline %{
void _boxcar_c_wrap(float* master_amplitude,   int h1, int w1,
                   float* slave_amplitude,    int h2, int w2,
                   float* dphase,             int h3, int w3,
                   float* amplitude_filtered, int h4, int w4,
                   float* dphase_filtered,    int h5, int w5,
                   float* coherence_filtered, int h6, int w6,
                   const int window_width)
{
    std::vector<el::Level> enabled_log_levels {
                                               el::Level::Info,
                                               el::Level::Verbose,
                                               el::Level::Warning,
                                               el::Level::Error,
                                               el::Level::Fatal,
                                               };
    boxcar(master_amplitude, slave_amplitude, dphase,
            amplitude_filtered, dphase_filtered, coherence_filtered,
            h1,
            w1,
            window_width,
            enabled_log_levels);
    return;
}
%}


%pythoncode{
import numpy as np

def boxcar(master_amplitude,
           slave_amplitude,
           dphase,
           window_width):
    amplitude_filtered = np.zeros_like(master_amplitude)
    dphase_filtered    = np.zeros_like(master_amplitude)
    coherence_filtered = np.zeros_like(master_amplitude)
    _despeckcl._boxcar_c_wrap(master_amplitude, slave_amplitude, dphase,
                          amplitude_filtered, dphase_filtered, coherence_filtered,
                          window_width);
    return (amplitude_filtered, dphase_filtered, coherence_filtered)
}

/* NLInSAR declaration and wrap */

%include "nlinsar.h"
%inline %{
void _nlinsar_c_wrap(float* master_amplitude,   int h1, int w1,
                     float* slave_amplitude,    int h2, int w2,
                     float* dphase,             int h3, int w3,
                     float* amplitude_filtered, int h4, int w4,
                     float* dphase_filtered,    int h5, int w5,
                     float* coherence_filtered, int h6, int w6,
                     const int search_window_size,
                     const int patch_size,
                     const int niter,
                     const int lmin)
{
    std::vector<el::Level> enabled_log_levels {
                                               el::Level::Info,
                                               el::Level::Verbose,
                                               el::Level::Warning,
                                               el::Level::Error,
                                               el::Level::Fatal,
                                               };

    nlinsar(master_amplitude, slave_amplitude, dphase,
            amplitude_filtered, dphase_filtered, coherence_filtered,
            h1,
            w1,
            search_window_size,
            patch_size,
            niter,
            lmin,
            enabled_log_levels);
    return;
}
%}

%pythoncode{
import numpy as np

def nlinsar(master_amplitude,
            slave_amplitude,
            dphase,
            search_window_size,
            patch_size,
            niter,
            lmin):
    amplitude_filtered = np.zeros_like(master_amplitude)
    dphase_filtered    = np.zeros_like(master_amplitude)
    coherence_filtered = np.zeros_like(master_amplitude)
    _despeckcl._nlinsar_c_wrap(master_amplitude, slave_amplitude, dphase,
                               amplitude_filtered, dphase_filtered, coherence_filtered,
                               search_window_size, patch_size, niter, lmin)
    return (amplitude_filtered, dphase_filtered, coherence_filtered)
}

/* NLSAR declaration and wrap */

%include "nlsar.h"
%inline %{
void _nlsar_c_wrap(float* master_amplitude,   int h1, int w1,
                   float* slave_amplitude,    int h2, int w2,
                   float* dphase,             int h3, int w3,
                   float* amplitude_filtered, int h4, int w4,
                   float* dphase_filtered,    int h5, int w5,
                   float* coherence_filtered, int h6, int w6,
                   const int search_window_size,
                   const int patch_size)
{
    std::vector<el::Level> enabled_log_levels {
                                               el::Level::Info,
                                               el::Level::Verbose,
                                               el::Level::Warning,
                                               el::Level::Error,
                                               el::Level::Fatal,
                                               };

    nlsar(master_amplitude, slave_amplitude, dphase,
          amplitude_filtered, dphase_filtered, coherence_filtered,
          h1,
          w1,
          search_window_size,
          patch_size,
          enabled_log_levels);
    return;
}
%}

%pythoncode{
import numpy as np

def nlsar(master_amplitude,
          slave_amplitude,
          dphase,
          search_window_size,
          patch_size):
    amplitude_filtered = np.zeros_like(master_amplitude)
    dphase_filtered    = np.zeros_like(master_amplitude)
    coherence_filtered = np.zeros_like(master_amplitude)
    _despeckcl._nlsar_c_wrap(master_amplitude, slave_amplitude, dphase,
                             amplitude_filtered, dphase_filtered, coherence_filtered,
                             search_window_size, patch_size)
    return (amplitude_filtered, dphase_filtered, coherence_filtered)
}

%module (docstring="despeckle Toolbox for InSAR written in OpenCL") despeckcl

%{
    #define SWIG_FILE_WITH_INIT
    #include "despeckcl.h"
%}

%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"
%include "despeckcl.h"
%include "bbox.h"

%nodefaultctor;

namespace std {
   %template(IntVector) vector<int>;
   %template(StringVector) vector<string>;
}

%init %{
    import_array();
%}

%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* ampl_master, int h1, int w1)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* ampl_slave,  int h2, int w2)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* dphase,      int h3, int w3)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* ampl_filt,   int h4, int w4)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* dphase_filt, int h5, int w5)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* coh_filt,    int h6, int w6)}

%ignore boxcar_routines;
%feature("autodoc", "2");


/* Boxcar declarations and definitions */
%inline %{
void _boxcar_c_wrap(float* ampl_master, int h1, int w1,
                    float* ampl_slave,  int h2, int w2,
                    float* dphase,      int h3, int w3,
                    float* ampl_filt,   int h4, int w4,
                    float* dphase_filt, int h5, int w5,
                    float* coh_filt,    int h6, int w6,
                    const int window_width,
                    const std::vector<std::string> enabled_log_levels)
{
    despeckcl::boxcar(ampl_master,
                      ampl_slave,
                      dphase,
                      ampl_filt,
                      dphase_filt,
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
           dphase,
           window_width,
           enabled_log_levels = ['error', 'warning', 'fatal']):

    ampl_filt   = np.zeros_like(ampl_master)
    dphase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)
    _despeckcl._boxcar_c_wrap(ampl_master,
                              ampl_slave,
                              dphase,
                              ampl_filt,
                              dphase_filt,
                              coh_filt,
                              window_width,
                              enabled_log_levels);
    return (ampl_filt, dphase_filt, coh_filt)
}

/* NLInSAR declaration and wrap */
%inline %{
void _nlinsar_c_wrap(float* ampl_master, int h1, int w1,
                     float* ampl_slave,  int h2, int w2,
                     float* dphase,      int h3, int w3,
                     float* ampl_filt,   int h4, int w4,
                     float* dphase_filt, int h5, int w5,
                     float* coh_filt,    int h6, int w6,
                     const int search_window_size,
                     const int patch_size,
                     const int niter,
                     const int lmin,
                     const std::vector<std::string> enabled_log_levels)
{
    despeckcl::nlinsar(ampl_master,
                       ampl_slave,
                       dphase,
                       ampl_filt,
                       dphase_filt,
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
            dphase,
            search_window_size,
            patch_size,
            niter,
            lmin,
            enabled_log_levels = ['error', 'warning', 'fatal']):

    ampl_filt   = np.zeros_like(ampl_master)
    dphase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)

    _despeckcl._nlinsar_c_wrap(ampl_master,
                               ampl_slave,
                               dphase,
                               ampl_filt,
                               dphase_filt,
                               coh_filt,
                               search_window_size,
                               patch_size,
                               niter,
                               lmin,
                               enabled_log_levels)
    return (ampl_filt, dphase_filt, coh_filt)
}

/* NLSAR declaration and wrap */
%inline %{
void _nlsar_c_wrap(float* ampl_master, int h1, int w1,
                   float* ampl_slave,  int h2, int w2,
                   float* dphase,      int h3, int w3,
                   float* ampl_filt,   int h4, int w4,
                   float* dphase_filt, int h5, int w5,
                   float* coh_filt,    int h6, int w6,
                   const int search_window_size,
                   const std::vector<int> patch_sizes,
                   const std::vector<int> scale_sizes,
                   const bbox training_dims,
                   const std::vector<std::string> enabled_log_levels)
{
    despeckcl::nlsar(ampl_master,
                     ampl_slave,
                     dphase,
                     ampl_filt,
                     dphase_filt,
                     coh_filt,
                     h1,
                     w1,
                     search_window_size,
                     patch_sizes,
                     scale_sizes,
                     training_dims,
                     enabled_log_levels);
}
%}

%pythoncode{
import numpy as np

def nlsar(ampl_master,
          ampl_slave,
          dphase,
          search_window_size,
          patch_sizes,
          scale_sizes,
          training_dims,
          enabled_log_levels = ['error', 'warning', 'fatal']):

    ampl_filt   = np.zeros_like(ampl_master)
    dphase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)

    _despeckcl._nlsar_c_wrap(ampl_master,
                             ampl_slave,
                             dphase,
                             ampl_filt,
                             dphase_filt,
                             coh_filt,
                             search_window_size,
                             patch_sizes,
                             scale_sizes,
                             training_dims,
                             enabled_log_levels)

    return (ampl_filt, dphase_filt, coh_filt)
}

/* Goldstein declaration and wrap */
%inline %{
void _goldstein_c_wrap(float* ampl_master, int h1, int w1,
                       float* ampl_slave,  int h2, int w2,
                       float* dphase,      int h3, int w3,
                       float* ampl_filt,   int h4, int w4,
                       float* dphase_filt, int h5, int w5,
                       float* coh_filt,    int h6, int w6,
                       const int patch_size,
                       const int overlap,
                       const float alpha,
                       const std::vector<std::string> enabled_log_levels)
{
    despeckcl::goldstein(ampl_master,
                         ampl_slave,
                         dphase,
                         ampl_filt,
                         dphase_filt,
                         coh_filt,
                         h1,
                         w1,
                         patch_size,
                         overlap,
                         alpha,
                         enabled_log_levels);
}
%}

%pythoncode{
import numpy as np

def goldstein(ampl_master,
              ampl_slave,
              dphase,
              patch_size,
              overlap,
              alpha,
              enabled_log_levels = ['error', 'warning', 'fatal']):

    ampl_filt   = np.zeros_like(ampl_master)
    dphase_filt = np.zeros_like(ampl_master)
    coh_filt    = np.zeros_like(ampl_master)

    _despeckcl._goldstein_c_wrap(ampl_master,
                                 ampl_slave,
                                 dphase,
                                 ampl_filt,
                                 dphase_filt,
                                 coh_filt,
                                 patch_size,
                                 overlap,
                                 alpha,
                                 enabled_log_levels)

    return (ampl_filt, dphase_filt, coh_filt)
}

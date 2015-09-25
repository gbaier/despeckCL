%module despeckcl
%{
    #include "boxcar.h"
    #include "nlinsar.h"
%}

%typemap(in) (float* in_array, int dim_h, int dim_w) (Matrix mat) {
  mat = $input.matrix_value();
  $2 = $input.dims()(0);
  $3 = $input.dims()(1);
  $1 = (float*) malloc($2 * $3 * sizeof(float));
  for (int i = 0; i < $2; i++) {
    for (int j = 0; j < $3; j++) {
      $1[i*$3 + j] = mat(i,j);
    }
  }
}

%typemap(in, numinputs=0) (float** out_array, int* dim_h, int* dim_w) {
    float* ptr = (float*) malloc(sizeof(float));
    int height;
    int width;
    $1 = &ptr;
    $2 = &height;
    $3 = &width;
}

%typemap(argout) (float** out_array, int* dim_h, int* dim_w) {
  Matrix mat(*$2, *$3);
  for (int i = 0; i < *$2; i++) {
    for (int j = 0; j < *$3; j++) {
      mat(i,j) = (*$1)[i* (*$3) + j];
    }
  }
  $result->append(mat);
}

%typemap(freearg) (float** out_array, int* dim_h, int* dim_w) {
   free(*$1);
}

%apply (float* in_array, int dim_h, int dim_w) {(float* master_amplitude, int h1, int w1)}
%apply (float* in_array, int dim_h, int dim_w) {(float* slave_amplitude,  int h2, int w2)}
%apply (float* in_array, int dim_h, int dim_w) {(float* dphase,           int h3, int w3)}

%apply (float** out_array, int* dim_h, int* dim_w) {(float** amplitude_filtered, int* h4, int* w4)}
%apply (float** out_array, int* dim_h, int* dim_w) {(float** dphase_filtered,    int* h5, int* w5)}
%apply (float** out_array, int* dim_h, int* dim_w) {(float** coherence_filtered, int* h6, int* w6)}

%ignore boxcar_routines;

/*
%rename (boxcar) my_boxcar;
*/

%inline %{
void boxcar(float* master_amplitude,   int h1, int w1,
            float* slave_amplitude,    int h2, int w2,
            float* dphase,             int h3, int w3,
            float** amplitude_filtered, int* h4, int* w4,
            float** dphase_filtered,    int* h5, int* w5,
            float** coherence_filtered, int* h6, int* w6,
            const int window_width)
{
    std::vector<el::Level> enabled_log_levels {
                   //                            el::Level::Info,
                   //                            el::Level::Verbose,
                                               el::Level::Warning,
                                               el::Level::Error,
                                               el::Level::Fatal,
                                               };
    const int height = h1;
    const int width = w1;

    *h4 = height;
    *h5 = height;
    *h6 = height;
    
    *w4 = width;
    *w5 = width;
    *w6 = width;

    *amplitude_filtered = (float*) malloc(height * width * sizeof(float));
    *dphase_filtered    = (float*) malloc(height * width * sizeof(float));
    *coherence_filtered = (float*) malloc(height * width * sizeof(float));

    boxcar(master_amplitude, slave_amplitude, dphase,
           *amplitude_filtered, *dphase_filtered, *coherence_filtered,
           height,
           width,
           window_width,
           enabled_log_levels);
    return;
}
%}

%inline %{
namespace nlinsar {
    void nlinsar(float* master_amplitude,   int h1, int w1,
                 float* slave_amplitude,    int h2, int w2,
                 float* dphase,             int h3, int w3,
                 float** amplitude_filtered, int* h4, int* w4,
                 float** dphase_filtered,    int* h5, int* w5,
                 float** coherence_filtered, int* h6, int* w6,
                 const int search_window_size,
                 const int patch_size,
                 const int niter,
                 const int lmin)
    {
        std::vector<el::Level> enabled_log_levels {
                       //                            el::Level::Info,
                       //                            el::Level::Verbose,
                                                   el::Level::Warning,
                                                   el::Level::Error,
                                                   el::Level::Fatal,
                                                   };
        const int height = h1;
        const int width = w1;

        *h4 = height;
        *h5 = height;
        *h6 = height;
        
        *w4 = width;
        *w5 = width;
        *w6 = width;

        *amplitude_filtered = (float*) malloc(height * width * sizeof(float));
        *dphase_filtered    = (float*) malloc(height * width * sizeof(float));
        *coherence_filtered = (float*) malloc(height * width * sizeof(float));

        nlinsar::nlinsar(master_amplitude, slave_amplitude, dphase,
                         *amplitude_filtered, *dphase_filtered, *coherence_filtered,
                         h1, w1,
                         search_window_size,
                         patch_size,
                         niter,
                         lmin,
                         enabled_log_levels);
        return;
    }
}
%}

%ignore nlinsar_routines;

#include "../include/boxcar.h"

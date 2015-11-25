%module despeckcl
%{
    #include "despeckcl.h"
%}

%include "std_vector.i"
%include "std_string.i"

namespace std {
   %template(IntVector) vector<int>;
   %template(StringVector) vector<string>;
}

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

%apply (float* in_array, int dim_h, int dim_w) {(float* ampl_master, int h1, int w1)}
%apply (float* in_array, int dim_h, int dim_w) {(float* ampl_slave,  int h2, int w2)}
%apply (float* in_array, int dim_h, int dim_w) {(float* dphase,      int h3, int w3)}

%apply (float** out_array, int* dim_h, int* dim_w) {(float** ampl_filt,   int* h4, int* w4)}
%apply (float** out_array, int* dim_h, int* dim_w) {(float** dphase_filt, int* h5, int* w5)}
%apply (float** out_array, int* dim_h, int* dim_w) {(float** coh_filt,    int* h6, int* w6)}

%ignore boxcar_routines;

%inline %{
void boxcar(float* ampl_master,  int h1, int w1,
            float* ampl_slave,   int h2, int w2,
            float* dphase,       int h3, int w3,
            float** ampl_filt,   int* h4, int* w4,
            float** dphase_filt, int* h5, int* w5,
            float** coh_filt,    int* h6, int* w6,
            const int window_width)
{
    std::vector<std::string> enabled_log_levels {"warning", "error", "fatal"};

    const int height = h1;
    const int width = w1;

    *h4 = height;
    *h5 = height;
    *h6 = height;
    
    *w4 = width;
    *w5 = width;
    *w6 = width;

    *ampl_filt   = (float*) malloc(height * width * sizeof(float));
    *dphase_filt = (float*) malloc(height * width * sizeof(float));
    *coh_filt    = (float*) malloc(height * width * sizeof(float));

    despeckcl::boxcar(ampl_master,
                      ampl_slave,
                      dphase,
                      *ampl_filt,
                      *dphase_filt,
                      *coh_filt,
                      height,
                      width,
                      window_width,
                      enabled_log_levels);
}
%}

%inline %{
void nlinsar(float* ampl_master,  int h1, int w1,
             float* ampl_slave,   int h2, int w2,
             float* dphase,       int h3, int w3,
             float** ampl_filt,   int* h4, int* w4,
             float** dphase_filt, int* h5, int* w5,
             float** coh_filt,    int* h6, int* w6,
             const int search_window_size,
             const int patch_size,
             const int niter,
             const int lmin)
{
    std::vector<std::string> enabled_log_levels {"warning", "error", "fatal"};

    const int height = h1;
    const int width = w1;

    *h4 = height;
    *h5 = height;
    *h6 = height;
    
    *w4 = width;
    *w5 = width;
    *w6 = width;

    *ampl_filt   = (float*) malloc(height * width * sizeof(float));
    *dphase_filt = (float*) malloc(height * width * sizeof(float));
    *coh_filt    = (float*) malloc(height * width * sizeof(float));

    despeckcl::nlinsar(ampl_master,
                       ampl_slave,
                       dphase,
                       *ampl_filt,
                       *dphase_filt,
                       *coh_filt,
                       h1,
                       w1,
                       search_window_size,
                       patch_size,
                       niter,
                       lmin,
                       enabled_log_levels);
}
%}

%ignore nlinsar_routines;

%inline %{
void nlsar(float* ampl_master,  int h1, int w1,
           float* ampl_slave,   int h2, int w2,
           float* dphase,       int h3, int w3,
           float** ampl_filt,   int* h4, int* w4,
           float** dphase_filt, int* h5, int* w5,
           float** coh_filt,    int* h6, int* w6,
           const int search_window_size,
           const std::vector<int> patch_sizes,
           const std::vector<int> scale_sizes,
           const int training_dim_h_low,
           const int training_dim_w_low,
           const int training_dim_size)
{
    std::vector<std::string> enabled_log_levels {"warning", "error", "fatal"};

    const int height = h1;
    const int width = w1;

    *h4 = height;
    *h5 = height;
    *h6 = height;
    
    *w4 = width;
    *w5 = width;
    *w6 = width;

    *ampl_filt   = (float*) malloc(height * width * sizeof(float));
    *dphase_filt = (float*) malloc(height * width * sizeof(float));
    *coh_filt    = (float*) malloc(height * width * sizeof(float));

    despeckcl::nlsar(ampl_master,
                     ampl_slave,
                     dphase,
                     *ampl_filt,
                     *dphase_filt,
                     *coh_filt,
                     h1,
                     w1,
                     search_window_size,
                     patch_sizes,
                     scale_sizes,
                     std::make_tuple(training_dim_h_low,
                                     training_dim_w_low,
                                     training_dim_size),
                     enabled_log_levels);
}
%}

%ignore nlsar_routines;

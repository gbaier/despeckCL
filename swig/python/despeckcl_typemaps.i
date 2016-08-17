%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(IntVector) vector<int>;
   %template(StringVector) vector<string>;
}

%init %{
    import_array();
%}

%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* ampl_master, int h1, int w1)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* ampl_slave,  int h2, int w2)}
%apply( float* IN_ARRAY2,     int DIM1, int DIM2)  {(float* phase,      int h3, int w3)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* ref_filt,   int h4, int w4)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* phase_filt, int h5, int w5)}
%apply( float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* coh_filt,    int h6, int w6)}

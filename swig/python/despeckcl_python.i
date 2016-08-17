%module (docstring="A despeckling/denoising Toolbox for SAR/InSAR written in OpenCL") despeckcl

%{
    #define SWIG_FILE_WITH_INIT
    #include "despeckcl.h"
%}

%include "nlsar.i"
%include "nlinsar.i"
%include "boxcar.i"
%include "goldstein.i"

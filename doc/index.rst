.. despeckCL documentation master file, created by
   sphinx-quickstart on Tue Sep 29 19:23:09 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to despeckCL's documentation!
=====================================

A toolbox for despeckling/denoising SAR/InSAR images written in OpenCL

Implemented Filters
-------------------

So far the following filters are implemented:

* **Boxcar**: The simple boxcar filter for InSAR data
* **NL-InSAR**: A nonlocal InSAR filter introduced in *Deledalle, C.-A.; Denis, L.; Tupin, F., "NL-InSAR: Nonlocal Interferogram Estimation," Geoscience and Remote Sensing, IEEE Transactions on , vol.49, no.4, pp.1441,1452, April 2011*
* **NL-SAR**: A nonlocal filter for SAR, InSAR and PolSAR, introduced in *Deledalle, C.-A.; Denis, L.; Tupin, F.; Reigber, A.; Jäger, M., "NL-SAR: A Unified Nonlocal Framework for Resolution-Preserving (Pol)(In)SAR Denoising," Geoscience and Remote Sensing, IEEE Transactions on , vol.53, no.4, pp.2021,2038, April 2015*

  * currently this implementation uses only a single search window of fixed size in contrast to the original paper, which uses multiple search window sizes.

* **Goldstein**: The Goldstein InSAR filter, *Goldstein, R. M.; Werner, C. L., "Radar interferogram filtering for geophysical applications," Geophysical Research Letters,vol. 25, no. 21, pp.4035,4038, 1998*


API documentation
-----------------

.. toctree::
   :maxdepth: 2

   cpp_api.rst
   python_api.rst
   octave_api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


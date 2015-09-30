C++ API
=======

.. cpp:function:: void boxcar(float* master_amplitude, float* slave_amplitude, float* dphase, float* ampl_filt, float* dphase_filt, float* coh_filt, const int height, const int width, int window_width, std::vector<std::string> enabled_log_levels)

   Filters the input with a boxcar filter

   :param master_amplitude: the amplitude of the master image
   :param slave_amplitude: the amplitude of the slave image
   :param dphase: the interferometric phase of the master and slave images
   :param ampl_filt: the filtered amplitude estimate
   :param dphase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param window_width: the window width of the boxcar window, has to be an odd number
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info


.. cpp:function:: int nlinsar(float* master_amplitude, float* slave_amplitude, float* dphase, float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered, const int height, const int width, const int search_window_size, const int patch_size, const int niter, const int lmin, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLInSAR filter

   :param master_amplitude: the amplitude of the master image
   :param slave_amplitude: the amplitude of the slave image
   :param dphase: the interferometric phase of the master and slave images
   :param ampl_filt: the filtered amplitude estimate
   :param dphase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_size: width of the patch, has to be an odd number
   :param niter: number of iterations
   :param lmin: minimum number of looks for the smoothing step
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

.. cpp:function:: int nlsar(float* master_amplitude, float* slave_amplitude, float* dphase, float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered, const int height, const int width, const int search_window_size, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLSAR filter

   :param master_amplitude: the amplitude of the master image
   :param slave_amplitude: the amplitude of the slave image
   :param dphase: the interferometric phase of the master and slave images
   :param ampl_filt: the filtered amplitude estimate
   :param dphase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_sizes: widths of the patches, have to be odd numbers
   :param scale_sizes: widths of the scales, have to be odd numbers
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

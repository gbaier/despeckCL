C++ API
=======

.. cpp:function:: void boxcar(float* ampl_master, float* ampl_slave, float* dphase, float* ampl_filt, float* dphase_filt, float* coh_filt, const int height, const int width, int window_width, std::vector<std::string> enabled_log_levels)

   Filters the input with a boxcar filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
   :param dphase: the interferometric phase of the master and slave images
   :param ampl_filt: the filtered amplitude estimate
   :param dphase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param window_width: the window width of the boxcar window, has to be an odd number
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info


.. cpp:function:: int nlinsar(float* ampl_master, float* ampl_slave, float* dphase, float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered, const int height, const int width, const int search_window_size, const int patch_size, const int niter, const int lmin, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLInSAR filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
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

.. cpp:function:: int nlsar(float* ampl_master, float* ampl_slave, float* dphase, float* amplitude_filtered, float* dphase_filtered, float* coherence_filtered, const int height, const int width, const int search_window_size, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes, const std::tuple<int, int, int> training_dims, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLSAR filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
   :param dphase: the interferometric phase of the master and slave images
   :param ampl_filt: the filtered amplitude estimate
   :param dphase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_sizes: widths of the patches, have to be odd numbers
   :param scale_sizes: widths of the scales, have to be odd numbers
   :param training_dims (x,y,w): location (x, y) of area used for training, where w is the width both directions
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

.. cpp:function:: int goldstein(float* ampl_master, float* ampl_slave, float* dphase, float* ampl_filt, float* dphase_filt, float* coh_filt, const unsigned int height, const unsigned int width, const unsigned int patch_size, const unsigned int overlap, const float alpha, std::vector<std::string> enabled_log_levels)

    Filters the input with the Goldstein filter

    :param ampl_master: the amplitude of the master image
    :param ampl_slave: the amplitude of the slave image
    :param dphase: the interferometric phase of the master and slave images
    :param patch_size: width of the patch for each 2D FFT
    :param overlap: overlap of the patches
    :param alpha: strength of filtering
    :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivy, phase and coherence estimates
    :rtype: tuple of ndarrays

C++ API
=======

.. cpp:function:: void boxcar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, int window_width, std::vector<std::string> enabled_log_levels)

   Filters the input with a boxcar filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
   :param phase: the interferometric phase of the master and slave images
   :param ref_filt: the filtered reflectivity estimate
   :param phase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param window_width: the window width of the boxcar window, has to be an odd number
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info


.. cpp:function:: int nlinsar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, const int search_window_size, const int patch_size, const int niter, const int lmin, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLInSAR filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
   :param phase: the interferometric phase of the master and slave images
   :param ref_filt: the filtered reflectivity estimate
   :param phase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_size: width of the patch, has to be an odd number
   :param niter: number of iterations
   :param lmin: minimum number of looks for the smoothing step
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

.. cpp:function:: int nlsar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, const int search_window_size, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes, std::map<nlsar::params, nlsar::stats> nlsar_stats, std::vector<std::string> enabled_log_levels)

   Filters the input with the NLSAR filter

   :param ampl_master: the amplitude of the master image
   :param ampl_slave: the amplitude of the slave image
   :param phase: the interferometric phase of the master and slave images
   :param ref_filt: the filtered reflectivity estimate
   :param phase_filt: the interferometric phase estimate
   :param coh_filt: the coherence estimate
   :param height: height of the images
   :param width: width of the images
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_sizes: widths of the patches, have to be odd numbers
   :param scale_sizes: widths of the scales, have to be odd numbers
   :param std\:\:map\<nlsar\:\:params, nlsar\:\:stats\> nlsar_stats: statistics computed on homogenous training area for all parameters
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

.. cpp:function:: std::map<params, stats> nlsar_training(float *ampl_master, float *ampl_slave, float *phase, const int height, const int width, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes)

    Trains the NL-SAR weighting kernel on a homogeneous area

    :param ampl_master: the amplitude of the master image
    :param ampl_slave: the amplitude of the slave image
    :param phase: the interferometric phase of the master and slave images
    :param patch_sizes: widths of the patches, have to be odd numbers
    :param scale_sizes: widths of the scales, have to be odd numbers

.. cpp:function:: int goldstein(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const unsigned int height, const unsigned int width, const unsigned int patch_size, const unsigned int overlap, const float alpha, std::vector<std::string> enabled_log_levels)

    Filters the input with the Goldstein filter

    :param ampl_master: the amplitude of the master image
    :param ampl_slave: the amplitude of the slave image
    :param phase: the interferometric phase of the master and slave images
    :param ref_filt: the filtered reflectivity estimate
    :param phase_filt: the interferometric phase estimate
    :param coh_filt: the coherence estimate
    :param patch_size: width of the patch for each 2D FFT
    :param overlap: overlap of the patches
    :param alpha: strength of filtering
    :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
    :return: a tuple containing the reflectivy, phase and coherence estimates
    :rtype: tuple of ndarrays

C++ API
=======

I strongly advise to use the Python bindings instead of calling the C++ code directly.
Primarily because this is the way I call and test the code all the time.

In any case, just include **despeckcl.h** and call the following functions for denoising a SAR interferogram, or, in the case of NL-SAR, any SAR covariance matrix.

Boxcar
------

.. cpp:function:: int boxcar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, const int window_width, std::vector<std::string> enabled_log_levels)

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

NL-InSAR
--------

.. cpp:function:: int nlinsar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, const int search_window_size, const int patch_size, const int niter, const int lmin, std::vector<std::string> enabled_log_levels)

   Filters the input with the NL-InSAR filter

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


NL-SAR
------

The functions for NL-SAR are overloaded and can be called either passing in the master and slave acquisitions' amplitudes and their interferometric phase for InSAR, or point to the covariance matrix for InSAR and PolSAR.
There is slight changes to Charles Deledalle's original implementation (https://www.math.u-bordeaux.fr/~cdeledal/nlsar.php).
This implementation only uses a quadratic search window of fixed size.


.. cpp:type:: nlsar_stats_collection = std::map<nlsar::params, nlsar::stats>

   Stores for each parameter set, i.e. the combinations of scale size and patch size, the corresponding statistics calculated on a homogeneous area.

.. cpp:class:: nlsar::params

   Stores one parameter set

    .. cpp:member:: int patch_size
    .. cpp:member:: int scale_size

.. cpp:class:: nlsar::stats

    Computes and stores the statitics gathered over a homogenous terrain

    .. cpp:function:: nlsar::stats::stats(std::vector<float> dissims, unsigned int lut_size)

       :param dissims: computed dissimilarities on the homogenous area
       :param lut_size: size of the quantilles' look-up-table


.. cpp:function:: nlsar_stats_collection nlsar_training(float *ampl_master, float *ampl_slave, float *phase, const int height, const int width, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes)

    Gathers the statistics for NL-SAR's weighting kernel on a homogeneous area.
    The area should be small.
    Around 30 by 30 pixels.

    :param ampl_master: the amplitude of the master image
    :param ampl_slave: the amplitude of the slave image
    :param phase: the interferometric phase of the master and slave images
    :param patch_sizes: widths of the patches, have to be odd numbers
    :param scale_sizes: widths of the scales, have to be odd numbers


.. cpp:function:: nlsar_stats_collection nlsar_training(float *covmat_raw, const int height, const int width, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes)

    Gathers the statistics for NL-SAR's weighting kernel on a homogeneous area.
    The area should be small.
    Around 30 by 30 pixels.

    :param covmat_raw: the unfiltered covariance matrix
    :param patch_sizes: widths of the patches, have to be odd numbers
    :param scale_sizes: widths of the scales, have to be odd numbers


.. cpp:function:: void store_nlsar_stats_collection(nlsar_stats_collection nsc, std::string filename)

    Stores the computed statistics in a file for later use.

.. cpp:function:: nlsar_stats_collection load_nlsar_stats_collection(std::string filename)

    Loads previously computed statistics from a file.

.. cpp:function:: int nlsar(float* ampl_master, float* ampl_slave, float* phase, float* ref_filt, float* phase_filt, float* coh_filt, const int height, const int width, const int search_window_size, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes, std::map<nlsar::params, nlsar::stats> nlsar_stats, const float h_param, const float c_param, std::vector<std::string> enabled_log_levels)

   Filters the input with the NL-SAR filter

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
   :param h_param: parameter h of the weighting kernel. In my experience 3 is a good default value.
   :param c_param: parameter c of the weighting kernel. Can most probably always be set to 49, as in the original paper.
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

.. cpp:function:: int nlsar(float* covmat_raw, float* covmat_filt, const int height, const int width, const int search_window_size, const std::vector<int> patch_sizes, const std::vector<int> scale_sizes, std::map<nlsar::params, nlsar::stats> nlsar_stats, const float h_param, const float c_param, std::vector<std::string> enabled_log_levels)

   Filters the input with the NL-SAR filter

   :param covmat_raw: the unfilterered covariance matrix
   :param covmat_filt: the filtered covariance matrix
   :param height: height of the image
   :param width: width of the image
   :param search_window_size: width of the search window, has to be an odd number
   :param patch_sizes: widths of the patches, have to be odd numbers
   :param scale_sizes: widths of the scales, have to be odd numbers
   :param std\:\:map\<nlsar\:\:params, nlsar\:\:stats\> nlsar_stats: statistics computed on homogenous training area for all parameters
   :param h_param: parameter h of the weighting kernel. In my experience 3 is a good default value.
   :param c_param: parameter c of the weighting kernel. Can most probably always be set to 49, as in the original paper.
   :param enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info

Goldstein
---------

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

Python API
==========

.. py:function:: boxcar(master_amplitude, slave_amplitude, dphase, window_width)

   Filters the input with a boxcar filter

   :param ndarray master_amplitude: the amplitude of the master image
   :param ndarray slave_amplitude: the amplitude of the slave image
   :param ndarray dphase: the interferometric phase of the master and slave images
   :param int window_width: the window width of the boxcar window, has to be an odd number
   :return: a tuple containing the reflectivy, phase and coherence estimates
   :rtype: tuple of ndarrays

.. py:function:: nlinsar(master_amplitude, slave_amplitude, dphase, search_window_size, patch_size, niter, lmin)

   Filters the input with the NLInSAR filter

   :param ndarray master_amplitude: the amplitude of the master image
   :param ndarray slave_amplitude: the amplitude of the slave image
   :param ndarray dphase: the interferometric phase of the master and slave images
   :param int search_window_size: width of the search window, has to be an odd number
   :param int patch_size: width of the patch, has to be an odd number
   :param int niter: number of iterations
   :param int lmin: minimum number of looks for the smoothing step
   :return: a tuple containing the reflectivy, phase and coherence estimates
   :rtype: tuple of ndarrays

.. py:function:: nlsar(master_amplitude, slave_amplitude, dphase, search_window_size, patch_sizes, scale_sizes[, enabled_log_levels=['error', 'warning', 'fatal']])

   Filters the input with the NLSAR filter

   :param ndarray master_amplitude: the amplitude of the master image
   :param ndarray slave_amplitude: the amplitude of the slave image
   :param ndarray dphase: the interferometric phase of the master and slave images
   :param int search_window_size: width of the search window, has to be an odd number
   :param [int] patch_sizes: widths of the patches, have to be odd numbers
   :param [int] scale_sizes: widths of the scales, have to be odd numbers
   :param [string] enabled_log_levels: enabled log levels, log levels are: error, fatal, warning, debug, info
   :return: a tuple containing the reflectivy, phase and coherence estimates
   :rtype: tuple of ndarrays

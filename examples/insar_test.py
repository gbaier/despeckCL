#!/usr/bin/python3
import sys
sys.path.append('/opt/despeckcl/bindings')
import despeckcl
import numpy as np
from scipy.io import loadmat

data = loadmat('insar_test_data.mat')
ampl_master = np.ascontiguousarray(data['ampl_master'])
ampl_slave  = np.ascontiguousarray(data['ampl_slave'])
dphase      = np.ascontiguousarray(data['dphase'])

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

ampl_plotopts = {'cmap':plt.get_cmap('gray'), 'vmin':0, 'vmax':70}

methods = {}

##################################
#
# Boxcar Filtering
#
#################################

window_size = 3
methods[despeckcl.boxcar] = (window_size,)

##################################
#
# NLSAR Filtering
#
#################################

search_window_size = 21
patch_sizes = [3, 5, 7, 9, 11]
scale_sizes = [1, 3, 5]
log_levels = ['debug', 'verbose', 'warning', 'fatal', 'error', 'info']
log_levels = ['verbose', 'info', 'warning', 'fatal', 'error']
log_levels = ['info', 'warning', 'fatal', 'error']
training_dim = (0, 0, 25)
methods[despeckcl.nlsar] = (search_window_size, patch_sizes, scale_sizes, training_dim, log_levels)

##################################
#
# NLInSAR Filtering
#
#################################

search_window_size = 21
patch_size = 7
niter = 3
lmin = 10
log_levels = ['debug', 'verbose', 'warning', 'fatal', 'error', 'info']
log_levels = ['info', 'warning', 'fatal', 'error']
methods[despeckcl.nlinsar] = (search_window_size, patch_size, niter, lmin, log_levels)

##################################
#
# Plotting
#
#################################


fig = plt.figure(1, (7., 11.))
print(len(methods) + 1)
grid = ImageGrid(fig, 111, # similar to subplot(111)
                 nrows_ncols = (len(methods) + 1, 3),
                 axes_pad=0.2, # pad between axes in inch.
                 cbar_mode='each',
                 share_all=True,
                 )

# input data
im = grid[0].imshow(20*np.log10(ampl_master), **ampl_plotopts)
grid.cbar_axes[0].colorbar(im) 

im = grid[1].imshow(20*np.log10(ampl_slave), cmap=plt.get_cmap('gray'))
grid.cbar_axes[1].colorbar(im) 

im = grid[2].imshow(dphase)
grid.cbar_axes[2].colorbar(im) 

for idx, (method, args) in enumerate(methods.items()):
    print(method.__name__)
    ampl_filt, dphase_filt, coh_filt = method(ampl_master,
                                              ampl_slave,
                                              dphase,
                                              *args)

    im = grid[3*(idx+1)].imshow(20*np.log10(ampl_filt), **ampl_plotopts)
    grid.cbar_axes[3*(idx+1)].colorbar(im) 
    im = grid[3*(idx+1) + 1].imshow(coh_filt, cmap=plt.get_cmap('gray'))
    grid.cbar_axes[3*(idx+1) + 1].colorbar(im) 
    im = grid[3*(idx+1) + 2].imshow(dphase_filt)
    grid.cbar_axes[3*(idx+1)+2].colorbar(im) 

plt.show()

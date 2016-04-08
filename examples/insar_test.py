#!/usr/bin/python3

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import sys
sys.path.insert(0, '../build/swig/python')
import despeckcl
print(despeckcl.__file__)


data = loadmat('insar_test_data.mat')
ampl_master = np.ascontiguousarray(data['ampl_master'])
ampl_slave = np.ascontiguousarray(data['ampl_slave'])
dphase = np.ascontiguousarray(data['dphase'])

ampl_plotopts = {'cmap': plt.get_cmap('bone')}

log_levels = ['debug', 'verbose', 'warning', 'fatal', 'error', 'info']
log_levels = ['warning', 'fatal', 'error']

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

nlsar_stats = despeckcl.nlsar_train(ampl_master[0:25, 0:25],
                                    ampl_slave[0:25, 0:25],
                                    dphase[0:25, 0:25],
                                    patch_sizes,
                                    scale_sizes)

methods[despeckcl.nlsar] = (search_window_size,
                            patch_sizes,
                            scale_sizes,
                            nlsar_stats,
                            log_levels)

##################################
#
# NLInSAR Filtering
#
#################################

search_window_size = 21
patch_size = 7
niter = 3
lmin = 10
methods[despeckcl.nlinsar] = (search_window_size,
                              patch_size,
                              niter,
                              lmin,
                              log_levels)

##################################
#
# Goldstein Filtering
#
#################################

patch_size = 32
overlap = 4
alpha = 0.5
methods[despeckcl.goldstein] = (patch_size, overlap, alpha, log_levels)

##################################
#
# Plotting
#
#################################

fig = plt.figure(1, (15., 9.))

nw = len(methods) + 1
nh = 3
grid = ImageGrid(fig, 111,
                 nrows_ncols=(nh, nw),
                 axes_pad=0.5,
                 cbar_pad=0.1,
                 cbar_mode='each',
                 share_all=True)

# input data
im = grid[0].imshow(20*np.log10(ampl_master), **ampl_plotopts)
grid.cbar_axes[0].colorbar(im)
grid[0].set_title('master amplitude')

im = grid[nw].imshow(20*np.log10(ampl_slave), **ampl_plotopts)
grid.cbar_axes[nw].colorbar(im)
grid[nw].set_title('slave amplitude')

im = grid[2*nw].imshow(dphase)
grid.cbar_axes[2*nw].colorbar(im)
grid[2*nw].set_title('phase')

for idx, (method, args) in enumerate(methods.items(), 1):
    print(method.__name__)
    ampl_filt, dphase_filt, coh_filt = method(ampl_master,
                                              ampl_slave,
                                              dphase,
                                              *args)

    im = grid[idx].imshow(20*np.log10(ampl_filt), **ampl_plotopts)
    grid[idx].set_title(method.__name__)
    grid.cbar_axes[idx].colorbar(im)

    im = grid[nw+idx].imshow(coh_filt, cmap=plt.get_cmap('gray'))
    grid.cbar_axes[nw+idx].colorbar(im)

    im = grid[2*nw+idx].imshow(dphase_filt)
    grid.cbar_axes[2*nw+idx].colorbar(im)

for ax in grid:
    ax.set_axis_off()

plt.show()

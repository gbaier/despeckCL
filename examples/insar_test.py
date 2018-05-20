#!/usr/bin/python3

import numpy as np
from scipy.io import loadmat
from collections import OrderedDict

# use Qt4Agg for X11 forwarding
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# add the build directory to the python search paths for finding the module
# without installing it
import sys
sys.path.insert(0, '../build/swig/python')
import despeckcl

# read input data
data = loadmat('insar_test_data.mat')
ampl_master = np.ascontiguousarray(data['ampl_master'])
ampl_slave = np.ascontiguousarray(data['ampl_slave'])
phase = np.ascontiguousarray(data['dphase'])


methods = OrderedDict()

log_levels = ['warning', 'fatal', 'error']
####################
#                  #
# Boxcar Filtering #
#                  #
####################

window_size = 5
methods[despeckcl.boxcar] = (window_size,)

####################
#                  #
# NL-SAR Filtering #
#                  #
####################

search_window_size = 21
patch_sizes = [3, 5, 7, 9, 11]
scale_sizes = [1, 3, 5]

training_area = np.s_[50:75, 50:75]

nlsar_stats = despeckcl.nlsar_train(ampl_master[training_area],
                                    ampl_slave[training_area],
                                    phase[training_area],
                                    patch_sizes,
                                    scale_sizes,
                                    log_levels)

# store and load NL-SAR statistics
despeckcl.store_nlsar_stats_collection(nlsar_stats, "nlsar_stats.txt")
nlsar_stats_res = despeckcl.load_nlsar_stats_collection("nlsar_stats.txt")

methods[despeckcl.nlsar] = (search_window_size,
                            patch_sizes,
                            scale_sizes,
                            nlsar_stats_res,
                            log_levels)

######################
#                    #
# NL-InSAR Filtering #
#                    #
######################

search_window_size = 21
patch_size = 7
niter = 5
lmin = 10
methods[despeckcl.nlinsar] = (search_window_size,
                              patch_size,
                              niter,
                              lmin,
                              log_levels)

#######################
#                     #
# Goldstein Filtering #
#                     #
#######################

patch_size = 32
overlap = 4
alpha = 0.5
methods[despeckcl.goldstein] = (patch_size, overlap, alpha, log_levels)

############
#          #
# Plotting #
#          #
############

fig = plt.figure(1, (15., 9.))

# plot options for amplitude, coherence, and phase
ampl_plotopts = {'cmap': plt.get_cmap('bone'), 'vmin': 0, 'vmax': 80}
coh_plotopts = {'cmap': plt.get_cmap('gray'), 'vmin': 0, 'vmax': 1}
phi_plotopts = {'cmap': plt.get_cmap('hsv'), 'vmin': -np.pi, 'vmax': np.pi}

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

im = grid[2*nw].imshow(phase, **phi_plotopts)
grid.cbar_axes[2*nw].colorbar(im)
grid[2*nw].set_title('phase')

# filtering results
print('starting filtering')
for idx, (method, args) in enumerate(methods.items(), 1):
    print(method.__name__)
    ref_filt, phase_filt, coh_filt = method(ampl_master,
                                            ampl_slave,
                                            phase,
                                            *args)

    im = grid[idx].imshow(10*np.log10(ref_filt), **ampl_plotopts)
    grid[idx].set_title(method.__name__)
    grid.cbar_axes[idx].colorbar(im)

    im = grid[nw+idx].imshow(coh_filt, **coh_plotopts)
    grid.cbar_axes[nw+idx].colorbar(im)

    im = grid[2*nw+idx].imshow(phase_filt, **phi_plotopts)
    grid.cbar_axes[2*nw+idx].colorbar(im)

for ax in grid:
    ax.set_axis_off()

plt.show()

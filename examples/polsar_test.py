""" example the downloads some fully polarimetric data from ESA's
PolSAR test data set and filters it with NL-SAR """

import os
import urllib.request
import zipfile

import gdal
import matplotlib.pyplot as plt
import numpy as np

# Add build directory to the python search paths for finding the module
# without installing it
import sys
sys.path.insert(0, '../build/swig/python')
import despeckcl

###############################
#                             #
# Get some test data from ESA #
#                             #
###############################

URL = 'https://earth.esa.int/documents/653194/658149/'

FILENAME = 'AIRSAR_Flevoland'
DATANAME = 'FLEVOL.STK'

# extracts data to use for training
TRAIN_SUB = np.s_[:, :, 200:230, 200:230]

# extracts data to be filtered and plotted
AREA_SUB = np.s_[:, :, :400, :600]


def stk_reader(stk_filename):
    """ see http://gdal.org/frmt_airsar.html for description """
    data = gdal.Open(stk_filename)
    data = data.ReadAsArray()
    mat = np.empty((3, 3, *data.shape[1:]), dtype=np.complex64)
    mat[0, 0] = data[0]
    mat[0, 1] = data[1]
    mat[1, 0] = data[1].conj()
    mat[0, 2] = data[2]
    mat[2, 0] = data[2].conj()
    mat[1, 1] = data[3]
    mat[1, 2] = data[4]
    mat[2, 1] = data[4].conj()
    mat[2, 2] = data[5]
    return mat


try:
    COVMAT = stk_reader(DATANAME)
except FileNotFoundError:
    urllib.request.urlretrieve(URL + FILENAME, FILENAME + '.zip')
    with zipfile.ZipFile(FILENAME + '.zip') as zf:
        zf.extract(DATANAME)
    COVMAT = stk_reader(DATANAME)

#############
#           #
# Filtering #
#           #
#############

PARAMS = {
    'search_window_size': 21,
    'patch_sizes': [3, 5, 7],
    'scale_sizes': [1, 3],
    'h': 3.0,
    'c': 49,
    'enabled_log_levels': ['warning', 'fatal', 'error'],  #, 'debug', 'info']
}

# store and load NL-SAR statistics
STATS_FILENAME = 'polsar_stats.txt'

print('getting similarity statistics')
if os.path.isfile(STATS_FILENAME):
    print('found saved statistics... restoring')
    NLSAR_STATS = despeckcl.load_nlsar_stats_collection(STATS_FILENAME)
else:
    print('computing statistics')
    NLSAR_STATS = despeckcl.nlsar_train(
        COVMAT[TRAIN_SUB], PARAMS['patch_sizes'], PARAMS['scale_sizes'])
    print('storing statistics')
    despeckcl.store_nlsar_stats_collection(NLSAR_STATS, STATS_FILENAME)

print('filtering')
COVMAT_FILT = despeckcl.nlsar(
    COVMAT[AREA_SUB], nlsar_stats=NLSAR_STATS, **PARAMS)

############
#          #
# Plotting #
#          #
############

fig = plt.figure()
ax = None

for nr, (data, title) in enumerate(
        zip([COVMAT[AREA_SUB], COVMAT_FILT], ['input', 'filtered']), 1):
    # extract diagonal elements
    diag = np.abs(np.diagonal(data)) + 0.000001

    # conversion to dB and normalization
    rgb_comp = 10 * np.log10(diag)
    rgb_comp_norm = rgb_comp - rgb_comp.min()
    rgb_comp_norm /= rgb_comp_norm.max()

    ax = fig.add_subplot(1, 2, nr, sharex=ax, sharey=ax)
    ax.imshow(rgb_comp_norm)
    ax.set_title(title)

plt.show()

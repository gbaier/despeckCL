#!/usr/bin/python3

import zipfile
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import gdal

# add the build directory to the python search paths for finding the module
# without installing it
import sys
sys.path.insert(0, '../build/swig/python')
import despeckcl

url = 'https://earth.esa.int/documents/653194/658149/'

filename = 'AIRSAR_Flevoland'
dataname = 'FLEVOL.STK'

train_sub = np.s_[:, :, 200:230, 200:230]
area_sub = np.s_[:, :, :300, :400]
dim_sub = np.s_[:2, :2]


def extract_from_archive(filename, dataname):
    with zipfile.ZipFile(filename) as zf:
        zf.extract(dataname)


def stk_reader(stk_filename):
    """ see http://gdal.org/frmt_airsar.html for description """
    data = gdal.Open(stk_filename)
    data = data.ReadAsArray()
    covmat = np.empty((3, 3, *data.shape[1:]), dtype=np.complex64)
    covmat[0, 0] = data[0]
    covmat[0, 1] = data[1]
    covmat[1, 0] = data[1].conj()
    covmat[0, 2] = data[2]
    covmat[2, 0] = data[2].conj()
    covmat[1, 1] = data[3]
    covmat[1, 2] = data[4]
    covmat[2, 1] = data[4].conj()
    covmat[2, 2] = data[5]
    return covmat


try:
    covmat = stk_reader(dataname)
except FileNotFoundError:
    urllib.request.urlretrieve(url + filename, filename + '.zip')
    extract_from_archive(filename + '.zip', dataname)
    covmat = stk_reader(dataname)

#############
#           #
# Filtering #
#           #
#############
covmat = covmat[dim_sub]

search_window_size = 21
patch_sizes = [3, 5, 7]
scale_sizes = [1, 3]
log_levels = ['warning', 'fatal', 'error']#, 'debug', 'info']

from IPython import embed
print('computing similarity statistics')
nlsar_stats = despeckcl.nlsar_train(covmat[train_sub], patch_sizes,
                                    scale_sizes)

print('filtering')
covmat_filt = despeckcl.nlsar(covmat[area_sub], search_window_size,
                              patch_sizes, scale_sizes, nlsar_stats, log_levels)


def plot_covmats(covmats, fig):
    nrows = len(covmats)
    ncols = covmats[0].shape[0]
    ax = None
    for nr, covmat in enumerate(covmats):
        # diagonal
        diag = np.abs(np.diagonal(covmat)) + 0.000001
        rgb_comp = 20 * np.log10(diag)
        #rgb_comp_norm = rgb_comp - rgb_comp.min()
        #rgb_comp_norm /= rgb_comp_norm.max()
        for nc in range(ncols):
            ax = fig.add_subplot(
                nrows, ncols, nr * ncols + nc + 1, sharex=ax, sharey=ax)
            #ax.imshow(rgb_comp_norm[:, :, nc], vmin=0, vmax=1)
            ax.imshow(rgb_comp[:, :, nc], vmin=-70, vmax=-10)
            ax.set_title('channel {}'.format(nc + 1))


fig = plt.figure()
plot_covmats([covmat[area_sub], covmat_filt], fig)
plt.show()
